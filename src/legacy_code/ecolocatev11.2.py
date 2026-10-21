from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import math
import shutil
import sys
import time
from typing import List, Optional, Tuple

try:
    import numpy as np
    import rasterio
    from affine import Affine
    from pyproj import Geod
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform as warp_transform
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy, rasterio, affine, and pyproj. Install them "
        "in the Python environment used to run this script, then run Step2.py again.\n"
        f"Missing dependency: {exc}"
    )


LANDSAT_INPUT_NAME = "Landsat_B10_merged.tif"
ECO_INPUT_NAME = "ECO_repro.tif"
ECO_TRANSLATED_NAME = "ECO_translated.tif"
TRANSLATION_LOG_NAME = "ECO_translation.csv"
GLOBAL_MAX_ECO_SHIFT_PIXELS = 300
COARSE_GLOBAL_DOWNSAMPLE = 16
NATIVE_REFINE_RADIUS_PIXELS = 50
COARSE_SEARCH_RADIUS = 4
REFINE_SEARCH_RADIUS = 3
NATIVE_CRS_EPSG = 4326
LOCAL_WARP_GRID = 5
LOCAL_WARP_RESIDUAL_RADIUS = 12
LOCAL_WARP_MIN_CONTROLS = 6
ALIGNMENT_METHOD = (
    f"nan-aware gradient magnitude + {COARSE_GLOBAL_DOWNSAMPLE}x coarse masked phase catch "
    "+ native phase refinement + globally prealigned affine local warp"
)

GEOD = Geod(ellps="WGS84")


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes:02d}:{sec:05.2f}"


class ConsoleLog:
    def __init__(self):
        self.start = time.perf_counter()
        self.width = max(78, min(118, shutil.get_terminal_size((100, 20)).columns))

    def elapsed(self) -> str:
        return format_elapsed(time.perf_counter() - self.start)

    def line(self, char: str = "-"):
        print(char * self.width, flush=True)

    def banner(self, title: str):
        self.line("=")
        print(title.center(self.width), flush=True)
        self.line("=")

    def section(self, title: str):
        print("", flush=True)
        print(f"[{self.elapsed()}] {title}", flush=True)

    def info(self, message: str):
        print(f"[{self.elapsed()}] {message}", flush=True)

    def detail(self, label: str, value):
        print(f"    {label:<28} {value}", flush=True)

    def success(self, message: str):
        print(f"[{self.elapsed()}] OK  {message}", flush=True)


@dataclass
class PyramidLevel:
    factor: int
    reference: np.ndarray
    moving: np.ndarray


@dataclass
class ShiftResult:
    row_shift: float
    col_shift: float
    score: float
    valid_pixels: int


@dataclass
class LocalShift:
    row: float
    col: float
    row_shift: float
    col_shift: float
    score: float
    valid_pixels: int


@dataclass
class WarpModel:
    row_offset: float
    row_col_coef: float
    row_row_coef: float
    col_offset: float
    col_col_coef: float
    col_row_coef: float
    control_points: int
    rmse_px: float
    mean_score: float
    method: str

    @classmethod
    def from_translation(cls, shift: ShiftResult, method: str = "translation fallback"):
        return cls(
            float(shift.row_shift), 0.0, 0.0,
            float(shift.col_shift), 0.0, 0.0,
            0, 0.0, float(shift.score), method,
        )

    def shift_at(self, row: float, col: float) -> Tuple[float, float]:
        row_shift = self.row_offset + (self.row_col_coef * col) + (self.row_row_coef * row)
        col_shift = self.col_offset + (self.col_col_coef * col) + (self.col_row_coef * row)
        return float(row_shift), float(col_shift)

    def pixel_affine(self) -> Affine:
        return Affine(
            1.0 + self.col_col_coef,
            self.col_row_coef,
            self.col_offset,
            self.row_col_coef,
            1.0 + self.row_row_coef,
            self.row_offset,
        )


@dataclass
class RasterGeometry:
    transform: Affine
    crs: object
    width: int
    height: int


def find_case_insensitive(folder: Path, filename: str) -> Path:
    target = filename.casefold()
    for path in folder.iterdir():
        if path.is_file() and path.name.casefold() == target:
            return path
    raise RuntimeError(f"Required input was not found: {filename}")


def nodata_to_nan(data: np.ndarray, nodata_value) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    invalid = ~np.isfinite(out) | (out == 0)
    if nodata_value is not None:
        try:
            nodata_float = float(nodata_value)
            if math.isnan(nodata_float):
                invalid |= np.isnan(out)
            else:
                invalid |= out == nodata_float
        except Exception:
            pass
    out[invalid] = np.nan
    return out


def read_single_band_as_nan(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 1:
            raise RuntimeError(f"{path.name} has no raster bands.")
        return nodata_to_nan(src.read(1), src.nodata)


def require_crs(src, path: Path):
    if src.crs is None:
        raise RuntimeError(f"{path.name} does not have a CRS.")
    return src.crs


def file_size_text(path: Path) -> str:
    try:
        size = float(path.stat().st_size)
    except Exception:
        return "unknown size"
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def array_summary(data: np.ndarray) -> str:
    valid = np.isfinite(data) & (data != 0)
    valid_count = int(valid.sum())
    total = int(data.size)
    if valid_count <= 0:
        return f"valid=0 / {total} (0.00%)"
    pct = 100.0 * valid_count / max(1, total)
    return f"valid={valid_count:,} / {total:,} ({pct:.1f}%)"


def gradient_magnitude_for_alignment(data: np.ndarray) -> np.ndarray:
    grad = np.full(data.shape, np.nan, dtype=np.float32)
    if min(data.shape) < 3:
        return grad

    center = data[1:-1, 1:-1]
    north = data[:-2, 1:-1]
    south = data[2:, 1:-1]
    west = data[1:-1, :-2]
    east = data[1:-1, 2:]
    valid = (
        np.isfinite(center)
        & np.isfinite(north)
        & np.isfinite(south)
        & np.isfinite(west)
        & np.isfinite(east)
    )
    gx = (east - west) * 0.5
    gy = (south - north) * 0.5
    mag = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32)
    inner = grad[1:-1, 1:-1]
    inner[valid] = mag[valid]
    inner[~valid] = np.nan
    grad[~np.isfinite(grad) | (grad == 0)] = np.nan
    return grad


def reproject_landsat_to_eco_grid(landsat_path: Path, eco_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(eco_path) as eco_src:
        eco_crs = require_crs(eco_src, eco_path)
        eco_transform = eco_src.transform
        eco_shape = (eco_src.height, eco_src.width)
        eco_data = nodata_to_nan(eco_src.read(1), eco_src.nodata)

    with rasterio.open(landsat_path) as landsat_src:
        require_crs(landsat_src, landsat_path)
        landsat_data = nodata_to_nan(landsat_src.read(1), landsat_src.nodata)
        landsat_on_eco = np.full(eco_shape, np.nan, dtype=np.float32)
        reproject(
            source=landsat_data,
            destination=landsat_on_eco,
            src_transform=landsat_src.transform,
            src_crs=landsat_src.crs,
            src_nodata=np.nan,
            dst_transform=eco_transform,
            dst_crs=eco_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    landsat_on_eco[~np.isfinite(landsat_on_eco) | (landsat_on_eco == 0)] = np.nan
    eco_data[~np.isfinite(eco_data) | (eco_data == 0)] = np.nan
    return landsat_on_eco, eco_data


def nanmean_downsample_2x(data: np.ndarray) -> np.ndarray:
    height, width = data.shape
    padded_height = height + (height % 2)
    padded_width = width + (width % 2)
    padded = np.full((padded_height, padded_width), np.nan, dtype=np.float32)
    padded[:height, :width] = data
    blocks = padded.reshape(padded_height // 2, 2, padded_width // 2, 2)
    valid = np.isfinite(blocks)
    totals = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    counts = valid.sum(axis=(1, 3))
    out = np.full(totals.shape, np.nan, dtype=np.float32)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def nanmean_downsample_power_of_two(data: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor == 1:
        return data
    out = data
    current = 1
    while current < factor:
        out = nanmean_downsample_2x(out)
        current *= 2
    return out


def build_pyramid(reference: np.ndarray, moving: np.ndarray, max_shift_pixels: int) -> List[PyramidLevel]:
    levels = [PyramidLevel(1, reference, moving)]
    factor = 1
    ref_level = reference
    mov_level = moving

    while math.ceil(max_shift_pixels / factor) > COARSE_SEARCH_RADIUS:
        if min(ref_level.shape) < 16:
            break
        ref_level = nanmean_downsample_2x(ref_level)
        mov_level = nanmean_downsample_2x(mov_level)
        factor *= 2
        levels.append(PyramidLevel(factor, ref_level, mov_level))

    return list(reversed(levels))


def shifted_slices(shape: Tuple[int, int], row_shift: int, col_shift: int):
    height, width = shape
    if abs(row_shift) >= height or abs(col_shift) >= width:
        return None

    if row_shift >= 0:
        ref_y0, ref_y1 = row_shift, height
        mov_y0, mov_y1 = 0, height - row_shift
    else:
        ref_y0, ref_y1 = 0, height + row_shift
        mov_y0, mov_y1 = -row_shift, height

    if col_shift >= 0:
        ref_x0, ref_x1 = col_shift, width
        mov_x0, mov_x1 = 0, width - col_shift
    else:
        ref_x0, ref_x1 = 0, width + col_shift
        mov_x0, mov_x1 = -col_shift, width

    if ref_y1 <= ref_y0 or ref_x1 <= ref_x0:
        return None
    return (slice(ref_y0, ref_y1), slice(ref_x0, ref_x1)), (slice(mov_y0, mov_y1), slice(mov_x0, mov_x1))


def min_valid_pixels_for_level(reference: np.ndarray, moving: np.ndarray) -> int:
    finite_limit = min(int(np.isfinite(reference).sum()), int(np.isfinite(moving).sum()))
    return min(1000, max(30, int(finite_limit * 0.002)))


def normalized_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    row_shift: int,
    col_shift: int,
    min_valid: int,
):
    slices = shifted_slices(reference.shape, row_shift, col_shift)
    if slices is None:
        return -math.inf, 0
    ref_slice, mov_slice = slices
    ref_view = reference[ref_slice]
    mov_view = moving[mov_slice]
    valid = np.isfinite(ref_view) & np.isfinite(mov_view)
    valid_count = int(valid.sum())
    if valid_count < min_valid:
        return -math.inf, valid_count

    ref_values = ref_view[valid].astype(np.float64)
    mov_values = mov_view[valid].astype(np.float64)
    ref_values -= ref_values.mean()
    mov_values -= mov_values.mean()
    denom = math.sqrt(float(np.dot(ref_values, ref_values) * np.dot(mov_values, mov_values)))
    if denom <= 0.0 or not math.isfinite(denom):
        return -math.inf, valid_count
    score = float(np.dot(ref_values, mov_values) / denom)
    return score, valid_count


def better_candidate(score, count, row_shift, col_shift, best: Optional[ShiftResult]) -> bool:
    if best is None:
        return True
    if score > best.score + 1e-12:
        return True
    if abs(score - best.score) <= 1e-12:
        new_distance = row_shift * row_shift + col_shift * col_shift
        old_distance = best.row_shift * best.row_shift + best.col_shift * best.col_shift
        if new_distance < old_distance:
            return True
        if new_distance == old_distance and count > best.valid_pixels:
            return True
    return False


def search_level(
    level: PyramidLevel,
    center_row: int,
    center_col: int,
    radius: int,
    max_shift_pixels: int,
) -> ShiftResult:
    best = None
    min_valid = min_valid_pixels_for_level(level.reference, level.moving)
    for row_shift in range(center_row - radius, center_row + radius + 1):
        for col_shift in range(center_col - radius, center_col + radius + 1):
            if abs(row_shift * level.factor) > max_shift_pixels:
                continue
            if abs(col_shift * level.factor) > max_shift_pixels:
                continue
            score, count = normalized_correlation(
                level.reference,
                level.moving,
                row_shift,
                col_shift,
                min_valid,
            )
            if not math.isfinite(score):
                continue
            if better_candidate(score, count, row_shift, col_shift, best):
                best = ShiftResult(row_shift, col_shift, score, count)
    if best is None:
        raise RuntimeError(
            f"No valid correlation candidate at pyramid factor {level.factor}. "
            "Check that the two rasters overlap and have enough non-nodata pixels."
        )
    return best


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def valid_search_crop(reference: np.ndarray, moving: np.ndarray, pad: int):
    valid = np.isfinite(reference) | np.isfinite(moving)
    ys, xs = np.where(valid)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("No valid pixels were found for phase correlation.")
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(reference.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(reference.shape[1], int(xs.max()) + pad + 1)
    return reference[y0:y1, x0:x1], moving[y0:y1, x0:x1]


def hann_window(shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    wy = np.hanning(height).astype(np.float32) if height > 1 else np.ones(1, dtype=np.float32)
    wx = np.hanning(width).astype(np.float32) if width > 1 else np.ones(1, dtype=np.float32)
    return wy[:, None] * wx[None, :]


def standardize_for_phase(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(data)
    if int(valid.sum()) < 30:
        raise RuntimeError("Not enough valid pixels for phase correlation.")
    values = data[valid].astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if not math.isfinite(std) or std <= 0.0:
        std = 1.0
    out = np.zeros(data.shape, dtype=np.float32)
    out[valid] = ((data[valid] - mean) / std).astype(np.float32)
    out *= hann_window(data.shape)
    return out, valid


def lag_to_index(lag: int, size: int) -> int:
    return int(lag) if lag >= 0 else int(size + lag)


def masked_phase_correlation_peak(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_pixels: int,
    center_row: int = 0,
    center_col: int = 0,
    search_radius: Optional[int] = None,
) -> ShiftResult:
    ref_crop, mov_crop = valid_search_crop(reference, moving, max_shift_pixels + 4)
    ref_phase, ref_valid = standardize_for_phase(ref_crop)
    mov_phase, mov_valid = standardize_for_phase(mov_crop)

    height, width = ref_phase.shape
    fft_shape = (
        next_power_of_two(2 * height - 1),
        next_power_of_two(2 * width - 1),
    )

    ref_fft = np.fft.rfft2(ref_phase, s=fft_shape)
    mov_fft = np.fft.rfft2(mov_phase, s=fft_shape)
    cross_power = ref_fft * np.conj(mov_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    phase_surface = np.fft.irfft2(cross_power, s=fft_shape)

    ref_mask_fft = np.fft.rfft2(ref_valid.astype(np.float32), s=fft_shape)
    mov_mask_fft = np.fft.rfft2(mov_valid.astype(np.float32), s=fft_shape)
    overlap_surface = np.fft.irfft2(ref_mask_fft * np.conj(mov_mask_fft), s=fft_shape)

    min_valid = min_valid_pixels_for_level(ref_crop, mov_crop)
    best = None
    if search_radius is None:
        row_min, row_max = -max_shift_pixels, max_shift_pixels
        col_min, col_max = -max_shift_pixels, max_shift_pixels
    else:
        row_min = max(-max_shift_pixels, int(center_row) - int(search_radius))
        row_max = min(max_shift_pixels, int(center_row) + int(search_radius))
        col_min = max(-max_shift_pixels, int(center_col) - int(search_radius))
        col_max = min(max_shift_pixels, int(center_col) + int(search_radius))

    for row_shift in range(row_min, row_max + 1):
        row_idx = lag_to_index(row_shift, fft_shape[0])
        for col_shift in range(col_min, col_max + 1):
            col_idx = lag_to_index(col_shift, fft_shape[1])
            valid_count = int(round(float(overlap_surface[row_idx, col_idx])))
            if valid_count < min_valid:
                continue
            score = float(phase_surface[row_idx, col_idx])
            if better_candidate(score, valid_count, row_shift, col_shift, best):
                best = ShiftResult(float(row_shift), float(col_shift), score, valid_count)

    if best is None:
        raise RuntimeError(
            "Masked phase correlation did not find a valid shift. "
            "Check raster overlap and nodata coverage."
        )
    return best


def valid_bbox(data_a: np.ndarray, data_b: np.ndarray):
    valid = np.isfinite(data_a) | np.isfinite(data_b)
    ys, xs = np.where(valid)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def shift_array_nearest(data: np.ndarray, row_shift: float, col_shift: float) -> np.ndarray:
    row_shift = int(round(float(row_shift)))
    col_shift = int(round(float(col_shift)))
    out = np.full(data.shape, np.nan, dtype=np.float32)
    slices = shifted_slices(data.shape, row_shift, col_shift)
    if slices is None:
        return out
    dst_slice, src_slice = slices
    out[dst_slice] = data[src_slice]
    return out


def local_window_slices(reference: np.ndarray, moving: np.ndarray, max_shift_pixels: int):
    bbox = valid_bbox(reference, moving)
    if bbox is None:
        return []
    y0, y1, x0, x1 = bbox
    height, width = reference.shape
    usable_h = y1 - y0
    usable_w = x1 - x0
    window = int(max(160, min(384, max_shift_pixels * 5)))
    window = min(window, height, width)
    if window < max(48, max_shift_pixels * 2):
        window = min(height, width)
    half = max(1, window // 2)

    cy_min = min(max(y0 + half, half), height - half)
    cy_max = max(min(y1 - half, height - half), cy_min)
    cx_min = min(max(x0 + half, half), width - half)
    cx_max = max(min(x1 - half, width - half), cx_min)

    if LOCAL_WARP_GRID <= 1:
        centers_y = [0.5 * (cy_min + cy_max)]
        centers_x = [0.5 * (cx_min + cx_max)]
    else:
        centers_y = np.linspace(cy_min, cy_max, LOCAL_WARP_GRID)
        centers_x = np.linspace(cx_min, cx_max, LOCAL_WARP_GRID)

    windows = []
    seen = set()
    for cy in centers_y:
        for cx in centers_x:
            cy_i = int(round(float(cy)))
            cx_i = int(round(float(cx)))
            wy0 = max(0, cy_i - half)
            wy1 = min(height, wy0 + window)
            wy0 = max(0, wy1 - window)
            wx0 = max(0, cx_i - half)
            wx1 = min(width, wx0 + window)
            wx0 = max(0, wx1 - window)
            key = (wy0, wy1, wx0, wx1)
            if key in seen or wy1 - wy0 < 32 or wx1 - wx0 < 32:
                continue
            seen.add(key)
            windows.append((wy0, wy1, wx0, wx1))
    return windows


def fit_warp_model(control_points: List[LocalShift], method: str) -> WarpModel:
    if len(control_points) < 3:
        raise RuntimeError("At least three local shifts are required to fit a linear warp.")

    a = np.array(
        [[1.0, point.col, point.row] for point in control_points],
        dtype=np.float64,
    )
    row_y = np.array([point.row_shift for point in control_points], dtype=np.float64)
    col_y = np.array([point.col_shift for point in control_points], dtype=np.float64)
    row_coef = np.linalg.lstsq(a, row_y, rcond=None)[0]
    col_coef = np.linalg.lstsq(a, col_y, rcond=None)[0]

    row_pred = a @ row_coef
    col_pred = a @ col_coef
    residuals = np.sqrt((row_pred - row_y) ** 2 + (col_pred - col_y) ** 2)
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    mean_score = float(np.mean([point.score for point in control_points])) if control_points else math.nan

    return WarpModel(
        float(row_coef[0]), float(row_coef[1]), float(row_coef[2]),
        float(col_coef[0]), float(col_coef[1]), float(col_coef[2]),
        len(control_points), rmse, mean_score, method,
    )


def robust_fit_warp_model(control_points: List[LocalShift]) -> WarpModel:
    if len(control_points) < LOCAL_WARP_MIN_CONTROLS:
        raise RuntimeError("Not enough local shifts for a robust linear warp.")

    model = fit_warp_model(control_points, "affine local warp")
    a = np.array([[1.0, point.col, point.row] for point in control_points], dtype=np.float64)
    row_y = np.array([point.row_shift for point in control_points], dtype=np.float64)
    col_y = np.array([point.col_shift for point in control_points], dtype=np.float64)
    row_pred = (
        model.row_offset
        + model.row_col_coef * a[:, 1]
        + model.row_row_coef * a[:, 2]
    )
    col_pred = (
        model.col_offset
        + model.col_col_coef * a[:, 1]
        + model.col_row_coef * a[:, 2]
    )
    residuals = np.sqrt((row_pred - row_y) ** 2 + (col_pred - col_y) ** 2)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    threshold = max(3.0, median + (3.0 * 1.4826 * mad))
    keep = residuals <= threshold
    filtered = [point for point, keep_point in zip(control_points, keep) if bool(keep_point)]
    if len(filtered) >= LOCAL_WARP_MIN_CONTROLS and len(filtered) < len(control_points):
        return fit_warp_model(filtered, "affine local warp (robust)")
    return model


def estimate_linear_warp(
    reference: np.ndarray,
    moving: np.ndarray,
    global_shift: ShiftResult,
    local_max_shift_pixels: int,
    logger: Optional[ConsoleLog] = None,
) -> WarpModel:
    base_row_shift = int(round(float(global_shift.row_shift)))
    base_col_shift = int(round(float(global_shift.col_shift)))
    prealigned_moving = shift_array_nearest(moving, base_row_shift, base_col_shift)
    windows = local_window_slices(reference, prealigned_moving, local_max_shift_pixels)
    controls = []

    for wy0, wy1, wx0, wx1 in windows:
        ref_win = reference[wy0:wy1, wx0:wx1]
        mov_win = prealigned_moving[wy0:wy1, wx0:wx1]
        try:
            phase = masked_phase_correlation_peak(
                ref_win,
                mov_win,
                local_max_shift_pixels,
                center_row=0,
                center_col=0,
                search_radius=LOCAL_WARP_RESIDUAL_RADIUS,
            )
            local_best = search_level(
                PyramidLevel(1, ref_win, mov_win),
                int(round(phase.row_shift)),
                int(round(phase.col_shift)),
                1,
                local_max_shift_pixels,
            )
            refined = subpixel_refine_with_ncc(
                ref_win,
                mov_win,
                int(round(local_best.row_shift)),
                int(round(local_best.col_shift)),
                local_max_shift_pixels,
            )
        except Exception:
            continue

        if not math.isfinite(refined.score):
            continue
        controls.append(
            LocalShift(
                row=0.5 * (wy0 + wy1 - 1),
                col=0.5 * (wx0 + wx1 - 1),
                row_shift=base_row_shift + refined.row_shift,
                col_shift=base_col_shift + refined.col_shift,
                score=refined.score,
                valid_pixels=refined.valid_pixels,
            )
        )

    if logger is not None:
        logger.info(
            f"Linear warp controls: {len(controls)}/{len(windows)} local windows "
            f"(residual search +/-{LOCAL_WARP_RESIDUAL_RADIUS}px)"
        )

    if len(controls) < LOCAL_WARP_MIN_CONTROLS:
        if logger is not None:
            logger.info("Linear warp fallback: not enough reliable local controls")
        return WarpModel.from_translation(global_shift)

    try:
        model = robust_fit_warp_model(controls)
    except Exception as exc:
        if logger is not None:
            logger.info(f"Linear warp fallback: {exc}")
        return WarpModel.from_translation(global_shift)

    if logger is not None:
        logger.info(
            f"Linear warp fit: controls={model.control_points}; "
            f"rmse={model.rmse_px:.2f}px; mean score={model.mean_score:.5f}"
        )
    return model


def parabolic_subpixel_offset(left: float, center: float, right: float) -> float:
    if not all(math.isfinite(value) for value in (left, center, right)):
        return 0.0
    denom = left - (2.0 * center) + right
    if denom >= -1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not math.isfinite(offset):
        return 0.0
    return max(-0.5, min(0.5, float(offset)))


def subpixel_refine_with_ncc(
    reference: np.ndarray,
    moving: np.ndarray,
    row_shift: int,
    col_shift: int,
    max_shift_pixels: int,
) -> ShiftResult:
    min_valid = min_valid_pixels_for_level(reference, moving)

    def score_at(row, col):
        if abs(row) > max_shift_pixels or abs(col) > max_shift_pixels:
            return -math.inf, 0
        return normalized_correlation(reference, moving, int(row), int(col), min_valid)

    center_score, center_count = score_at(row_shift, col_shift)
    if not math.isfinite(center_score):
        raise RuntimeError("Could not compute normalized correlation at the phase-correlation peak.")

    row_minus, _ = score_at(row_shift - 1, col_shift)
    row_plus, _ = score_at(row_shift + 1, col_shift)
    col_minus, _ = score_at(row_shift, col_shift - 1)
    col_plus, _ = score_at(row_shift, col_shift + 1)

    row_offset = parabolic_subpixel_offset(row_minus, center_score, row_plus)
    col_offset = parabolic_subpixel_offset(col_minus, center_score, col_plus)
    refined_row = max(-max_shift_pixels, min(max_shift_pixels, float(row_shift) + row_offset))
    refined_col = max(-max_shift_pixels, min(max_shift_pixels, float(col_shift) + col_offset))
    return ShiftResult(refined_row, refined_col, center_score, center_count)


def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    global_max_shift_pixels: int,
    native_refine_radius_pixels: int,
    coarse_downsample: int,
    logger: Optional[ConsoleLog] = None,
    describe_shift=None,
) -> ShiftResult:
    if reference.shape != moving.shape:
        raise RuntimeError("Reference and moving arrays must already be on the same grid.")

    coarse_downsample = max(1, int(coarse_downsample))
    if coarse_downsample > 1:
        coarse_reference = nanmean_downsample_power_of_two(reference, coarse_downsample)
        coarse_moving = nanmean_downsample_power_of_two(moving, coarse_downsample)
        coarse_max = int(math.ceil(global_max_shift_pixels / coarse_downsample))
        coarse_peak = masked_phase_correlation_peak(
            coarse_reference,
            coarse_moving,
            coarse_max,
        )
        coarse_row_native = coarse_peak.row_shift * coarse_downsample
        coarse_col_native = coarse_peak.col_shift * coarse_downsample
        if logger is not None:
            coarse_shift = (
                describe_shift(coarse_row_native, coarse_col_native)
                if describe_shift is not None
                else f"row={coarse_row_native:+.0f}, col={coarse_col_native:+.0f}"
            )
            logger.info(
                f"Coarse {coarse_downsample}x catch: {coarse_shift}; "
                f"phase={coarse_peak.score:.5f}; valid={coarse_peak.valid_pixels:,}"
            )
    else:
        coarse_row_native = 0.0
        coarse_col_native = 0.0

    phase_peak = masked_phase_correlation_peak(
        reference,
        moving,
        global_max_shift_pixels,
        center_row=int(round(coarse_row_native)),
        center_col=int(round(coarse_col_native)),
        search_radius=native_refine_radius_pixels,
    )
    if logger is not None:
        shift_text = (
            describe_shift(phase_peak.row_shift, phase_peak.col_shift)
            if describe_shift is not None
            else f"row={phase_peak.row_shift:+.0f}, col={phase_peak.col_shift:+.0f}"
        )
        logger.info(
            f"Native phase refine: {shift_text}; "
            f"phase={phase_peak.score:.5f}; valid={phase_peak.valid_pixels:,}"
        )

    integer_best = search_level(
        PyramidLevel(1, reference, moving),
        int(round(phase_peak.row_shift)),
        int(round(phase_peak.col_shift)),
        2,
        global_max_shift_pixels,
    )
    if logger is not None:
        shift_text = (
            describe_shift(integer_best.row_shift, integer_best.col_shift)
            if describe_shift is not None
            else f"row={integer_best.row_shift:+.0f}, col={integer_best.col_shift:+.0f}"
        )
        logger.info(
            f"Local NCC check: {shift_text}; "
            f"score={integer_best.score:.5f}; valid={integer_best.valid_pixels:,}"
        )

    refined = subpixel_refine_with_ncc(
        reference,
        moving,
        int(round(integer_best.row_shift)),
        int(round(integer_best.col_shift)),
        global_max_shift_pixels,
    )
    if logger is not None:
        shift_text = (
            describe_shift(refined.row_shift, refined.col_shift)
            if describe_shift is not None
            else f"row={refined.row_shift:+.2f}, col={refined.col_shift:+.2f}"
        )
        logger.info(
            f"Sub-pixel fit: {shift_text}; "
            f"score={refined.score:.5f}; valid={refined.valid_pixels:,}"
        )
    return refined


def shifted_transform(transform: Affine, row_shift: float, col_shift: float) -> Affine:
    return transform * Affine.translation(col_shift, row_shift)


def load_raster_geometry(path: Path) -> RasterGeometry:
    with rasterio.open(path) as src:
        crs = require_crs(src, path)
        return RasterGeometry(src.transform, crs, src.width, src.height)


def apply_translation_like_geoviewer(eco_path: Path, output_path: Path, row_shift: float, col_shift: float):
    with rasterio.open(eco_path) as src:
        require_crs(src, eco_path)
        source_data = nodata_to_nan(src.read(1), src.nodata)
        destination = np.full(source_data.shape, np.nan, dtype=np.float32)
        new_source_transform = shifted_transform(src.transform, row_shift, col_shift)
        reproject(
            source=source_data,
            destination=destination,
            src_transform=new_source_transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=np.nan,
            fill_value=np.nan,
            # Keep the translated scene snapped to the original ECO grid so
            # shifted and unshifted outputs line up cell-for-cell.
            resampling=Resampling.nearest,
        )
        destination[~np.isfinite(destination) | (destination == 0)] = np.nan
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = src.tags(1)

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination.astype(np.float32), 1)
        if tags:
            dst.update_tags(**tags)
        if band_tags:
            dst.update_tags(1, **band_tags)


def apply_warp_like_geoviewer(eco_path: Path, output_path: Path, warp_model: WarpModel):
    with rasterio.open(eco_path) as src:
        require_crs(src, eco_path)
        source_data = nodata_to_nan(src.read(1), src.nodata)
        destination = np.full(source_data.shape, np.nan, dtype=np.float32)
        new_source_transform = src.transform * warp_model.pixel_affine()
        reproject(
            source=source_data,
            destination=destination,
            src_transform=new_source_transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=np.nan,
            fill_value=np.nan,
            resampling=Resampling.nearest,
        )
        destination[~np.isfinite(destination) | (destination == 0)] = np.nan
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = src.tags(1)

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination.astype(np.float32), 1)
        if tags:
            dst.update_tags(**tags)
        if band_tags:
            dst.update_tags(1, **band_tags)


def raster_center(transform: Affine, width: int, height: int):
    return transform * (width / 2.0, height / 2.0)


def lonlat_from_crs(crs, x, y):
    if crs and crs.to_epsg() == NATIVE_CRS_EPSG:
        return float(x), float(y)
    lon, lat = warp_transform(crs, f"EPSG:{NATIVE_CRS_EPSG}", [x], [y])
    return float(lon[0]), float(lat[0])


def translation_geodesic_from_geometry(geometry: RasterGeometry, row_shift: float, col_shift: float):
    original_transform = geometry.transform
    moved_transform = shifted_transform(geometry.transform, row_shift, col_shift)
    x0, y0 = raster_center(original_transform, geometry.width, geometry.height)
    x1, y1 = raster_center(moved_transform, geometry.width, geometry.height)

    lon0, lat0 = lonlat_from_crs(geometry.crs, x0, y0)
    lon1, lat1 = lonlat_from_crs(geometry.crs, x1, y1)
    azimuth, _back_azimuth, distance = GEOD.inv(lon0, lat0, lon1, lat1)
    if abs(float(row_shift)) < 1e-12 and abs(float(col_shift)) < 1e-12:
        azimuth = 0.0
    else:
        azimuth %= 360.0
    dx = moved_transform.c - original_transform.c
    dy = moved_transform.f - original_transform.f
    return {
        "center_lon_before": lon0,
        "center_lat_before": lat0,
        "center_lon_after": lon1,
        "center_lat_after": lat1,
        "x_shift_map_units": dx,
        "y_shift_map_units": dy,
        "azimuth_deg": azimuth,
        "distance_m": distance,
    }


def translation_geodesic(eco_path: Path, row_shift: float, col_shift: float):
    return translation_geodesic_from_geometry(load_raster_geometry(eco_path), row_shift, col_shift)


def format_shift_report(row_shift: float, col_shift: float, geodesic_info: dict) -> str:
    return (
        f"row={float(row_shift):+.2f}px col={float(col_shift):+.2f}px | "
        f"{geodesic_info['distance_m']:.2f} m @ {geodesic_info['azimuth_deg']:.3f} deg"
    )


def write_translation_csv(
    csv_path: Path,
    landsat_path: Path,
    eco_path: Path,
    output_path: Path,
    result: ShiftResult,
    geodesic_info: dict,
    warp_model: WarpModel,
):
    fields = [
        "timestamp",
        "reference_file",
        "moving_file",
        "translated_file",
        "alignment_method",
        "max_search_pixels",
        "coarse_downsample",
        "native_refine_radius_pixels",
        "row_shift_px",
        "col_shift_px",
        "x_shift_map_units",
        "y_shift_map_units",
        "azimuth_deg",
        "distance_m",
        "correlation_score",
        "valid_pixels_at_best_shift",
        "warp_method",
        "warp_control_points",
        "warp_rmse_px",
        "warp_mean_score",
        "row_offset",
        "row_col_coef",
        "row_row_coef",
        "col_offset",
        "col_col_coef",
        "col_row_coef",
        "pixel_affine_a",
        "pixel_affine_b",
        "pixel_affine_c",
        "pixel_affine_d",
        "pixel_affine_e",
        "pixel_affine_f",
    ]
    pixel_affine = warp_model.pixel_affine()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": landsat_path.name,
        "moving_file": eco_path.name,
        "translated_file": output_path.name,
        "alignment_method": ALIGNMENT_METHOD,
        "max_search_pixels": GLOBAL_MAX_ECO_SHIFT_PIXELS,
        "coarse_downsample": COARSE_GLOBAL_DOWNSAMPLE,
        "native_refine_radius_pixels": NATIVE_REFINE_RADIUS_PIXELS,
        "row_shift_px": result.row_shift,
        "col_shift_px": result.col_shift,
        "x_shift_map_units": geodesic_info["x_shift_map_units"],
        "y_shift_map_units": geodesic_info["y_shift_map_units"],
        "azimuth_deg": geodesic_info["azimuth_deg"],
        "distance_m": geodesic_info["distance_m"],
        "correlation_score": result.score,
        "valid_pixels_at_best_shift": result.valid_pixels,
        "warp_method": warp_model.method,
        "warp_control_points": warp_model.control_points,
        "warp_rmse_px": warp_model.rmse_px,
        "warp_mean_score": warp_model.mean_score,
        "row_offset": warp_model.row_offset,
        "row_col_coef": warp_model.row_col_coef,
        "row_row_coef": warp_model.row_row_coef,
        "col_offset": warp_model.col_offset,
        "col_col_coef": warp_model.col_col_coef,
        "col_row_coef": warp_model.col_row_coef,
        "pixel_affine_a": pixel_affine.a,
        "pixel_affine_b": pixel_affine.b,
        "pixel_affine_c": pixel_affine.c,
        "pixel_affine_d": pixel_affine.d,
        "pixel_affine_e": pixel_affine.e,
        "pixel_affine_f": pixel_affine.f,
    }
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def raster_file_summary(path: Path) -> dict:
    with rasterio.open(path) as src:
        crs_text = str(src.crs) if src.crs else "None"
        return {
            "name": path.name,
            "size": file_size_text(path),
            "shape": f"{src.width} columns x {src.height} rows",
            "dtype": ", ".join(src.dtypes),
            "crs": crs_text,
            "nodata": src.nodata,
            "pixel_size": f"{abs(src.transform.a):.10f} x {abs(src.transform.e):.10f}",
        }


def log_raster_summary(logger: ConsoleLog, title: str, path: Path):
    summary = raster_file_summary(path)
    logger.info(title)
    logger.detail("File", summary["name"])
    logger.detail("Size", summary["size"])
    logger.detail("Grid", summary["shape"])
    logger.detail("Data type", summary["dtype"])
    logger.detail("CRS", summary["crs"])
    logger.detail("Pixel size", summary["pixel_size"])
    logger.detail("Declared nodata", summary["nodata"])


def main():
    logger = ConsoleLog()
    logger.banner("STEP 2 - ECOSTRESS TO LANDSAT GRADIENT PHASE TRANSLATION")

    folder = Path(__file__).resolve().parent
    landsat_path = find_case_insensitive(folder, LANDSAT_INPUT_NAME)
    eco_path = find_case_insensitive(folder, ECO_INPUT_NAME)
    output_path = folder / ECO_TRANSLATED_NAME
    csv_path = folder / TRANSLATION_LOG_NAME

    logger.section("Inputs")
    logger.info(f"Folder: {folder}")
    logger.info(
        f"Truth: {landsat_path.name} ({file_size_text(landsat_path)}) | "
        f"Moving: {eco_path.name} ({file_size_text(eco_path)})"
    )
    logger.info(
        f"Search <= +/-{GLOBAL_MAX_ECO_SHIFT_PIXELS} ECO px via {COARSE_GLOBAL_DOWNSAMPLE}x coarse catch; "
        f"native refine +/-{NATIVE_REFINE_RADIUS_PIXELS} px"
    )
    logger.info("Nodata ignored: nan, declared nodata, 0")
    logger.info("Alignment: nan-aware gradient magnitude; warp: affine/linear")

    logger.section("Prepare grid")
    step_start = time.perf_counter()
    landsat_on_eco, eco_data = reproject_landsat_to_eco_grid(landsat_path, eco_path)

    overlap = np.isfinite(landsat_on_eco) & np.isfinite(eco_data)
    if int(overlap.sum()) < 30:
        raise RuntimeError("The Landsat and ECOSTRESS rasters do not have enough overlapping valid pixels.")

    landsat_gradient = gradient_magnitude_for_alignment(landsat_on_eco)
    eco_gradient = gradient_magnitude_for_alignment(eco_data)
    gradient_overlap = np.isfinite(landsat_gradient) & np.isfinite(eco_gradient)
    if int(gradient_overlap.sum()) < 30:
        raise RuntimeError("The Landsat and ECOSTRESS gradients do not have enough overlapping valid pixels.")

    logger.success(
        f"Prepared in {format_elapsed(time.perf_counter() - step_start)} | "
        f"raw overlap={int(overlap.sum()):,} px | "
        f"gradient overlap={int(gradient_overlap.sum()):,} px"
    )

    eco_geometry = load_raster_geometry(eco_path)

    def describe_shift(row_shift, col_shift):
        return format_shift_report(
            row_shift,
            col_shift,
            translation_geodesic_from_geometry(eco_geometry, row_shift, col_shift),
        )

    logger.section("Search")
    step_start = time.perf_counter()
    global_result = estimate_translation(
        landsat_gradient,
        eco_gradient,
        GLOBAL_MAX_ECO_SHIFT_PIXELS,
        NATIVE_REFINE_RADIUS_PIXELS,
        COARSE_GLOBAL_DOWNSAMPLE,
        logger=logger,
        describe_shift=describe_shift,
    )
    warp_model = estimate_linear_warp(
        landsat_gradient,
        eco_gradient,
        global_result,
        NATIVE_REFINE_RADIUS_PIXELS,
        logger=logger,
    )
    center_row = (eco_geometry.height - 1) / 2.0
    center_col = (eco_geometry.width - 1) / 2.0
    center_row_shift, center_col_shift = warp_model.shift_at(center_row, center_col)
    result = ShiftResult(
        center_row_shift,
        center_col_shift,
        warp_model.mean_score if math.isfinite(warp_model.mean_score) else global_result.score,
        global_result.valid_pixels,
    )
    logger.info(f"Warp center shift: {describe_shift(result.row_shift, result.col_shift)}")
    logger.success(f"Gradient phase + affine warp completed in {format_elapsed(time.perf_counter() - step_start)}")

    logger.section("Write outputs")
    logger.info(f"Applying {warp_model.method} with NN resampling on the original ECO grid")
    step_start = time.perf_counter()
    apply_warp_like_geoviewer(eco_path, output_path, warp_model)
    geodesic_info = translation_geodesic_from_geometry(eco_geometry, result.row_shift, result.col_shift)
    write_translation_csv(csv_path, landsat_path, eco_path, output_path, result, geodesic_info, warp_model)
    logger.success(
        f"Wrote {output_path.name} and {csv_path.name} in "
        f"{format_elapsed(time.perf_counter() - step_start)}"
    )
    logger.info(
        f"Final center: {describe_shift(result.row_shift, result.col_shift)} | "
        f"score={result.score:.5f}; valid={result.valid_pixels:,}"
    )
    logger.success(f"Done. Total elapsed: {logger.elapsed()}")
    logger.line("=")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)