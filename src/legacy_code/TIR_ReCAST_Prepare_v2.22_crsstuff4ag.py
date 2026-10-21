"""TIR-ReCAST Prepare v2.22: retain each source group's native CRS."""

from pathlib import Path
import math
import re
import sys

try:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy and rasterio. Install them in the Python "
        "environment used to run this script, then run Step1.py again.\n"
        f"Missing dependency: {exc}"
    )


LANDSAT_SCALE = 0.00341802
LANDSAT_OFFSET = 149.0
LANDSAT_OUTPUT_NAME = "Landsat_B10_merged.tif"


def is_landsat_b10(path: Path) -> bool:
    name = path.name.upper()
    return (
        path.suffix.upper() in {".TIF", ".TIFF"}
        and name.startswith(("LC08_", "LC09_"))
        and name.endswith(("_ST_B10.TIF", "_ST_B10.TIFF"))
    )


def is_ecostress_lst(path: Path) -> bool:
    name = path.name.upper()
    return (
        path.suffix.upper() in {".TIF", ".TIFF"}
        and name.startswith("ECOV002_")
        and "_PROCESSED" not in name
    )


def ecostress_datetime_key(path: Path) -> str:
    name = path.stem

    match = re.search(r"(?<!\d)((?:19|20)\d{6})[Tt](\d{6})(?!\d)", name)
    if match:
        return f"{match.group(1)}T{match.group(2)}"

    match = re.search(r"(?<!\d)((?:19|20)\d{6})[_.-](\d{6})(?!\d)", name)
    if match:
        return f"{match.group(1)}T{match.group(2)}"

    match = re.search(r"(?<!\d)((?:19|20)\d{12})(?!\d)", name)
    if match:
        stamp = match.group(1)
        return f"{stamp[:8]}T{stamp[8:]}"

    raise RuntimeError(
        f"Could not identify YYYYMMDDTHHMMSS datetime in ECOSTRESS filename: {path.name}"
    )


def find_inputs(folder: Path):
    tif_paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.upper() in {".TIF", ".TIFF"}
    ]
    landsat_paths = sorted(path for path in tif_paths if is_landsat_b10(path))
    ecostress_paths = sorted(path for path in tif_paths if is_ecostress_lst(path))
    return landsat_paths, ecostress_paths


def _nodata_mask(data: np.ndarray, nodata_value) -> np.ndarray:
    mask = ~np.isfinite(data) | (data == 0)
    if nodata_value is None:
        return mask
    try:
        nodata_float = float(nodata_value)
    except Exception:
        return mask
    if math.isnan(nodata_float):
        return mask | np.isnan(data)
    return mask | (data == nodata_float)


def read_band_with_nan(src, *, scale=1.0, offset=0.0) -> np.ndarray:
    data = src.read(1).astype(np.float32)
    invalid = _nodata_mask(data, src.nodata)
    data[invalid] = np.nan
    valid = ~invalid
    data[valid] = data[valid] * scale + offset
    return data


def _metadata_float(tags, keys):
    normalized = {str(key).strip().lower(): value for key, value in tags.items()}
    for key in keys:
        if key not in normalized:
            continue
        try:
            value = float(normalized[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value, key
    return None, None


def ecostress_scale_offset(src):
    scale = 1.0
    offset = 0.0
    scale_source = None
    offset_source = None

    try:
        native_scale = float(src.scales[0])
        if math.isfinite(native_scale) and native_scale != 1.0:
            scale = native_scale
            scale_source = "GeoTIFF band scale"
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    try:
        native_offset = float(src.offsets[0])
        if math.isfinite(native_offset) and native_offset != 0.0:
            offset = native_offset
            offset_source = "GeoTIFF band offset"
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    tags = {}
    tags.update(src.tags())
    tags.update(src.tags(1))

    if scale_source is None:
        tagged_scale, key = _metadata_float(
            tags,
            ("scale_factor", "scalefactor", "scale", "slope", "multiplier"),
        )
        if tagged_scale is not None:
            scale = tagged_scale
            scale_source = f"metadata tag '{key}'"

    if offset_source is None:
        tagged_offset, key = _metadata_float(
            tags,
            ("add_offset", "addoffset", "offset", "intercept"),
        )
        if tagged_offset is not None:
            offset = tagged_offset
            offset_source = f"metadata tag '{key}'"

    return scale, offset, scale_source, offset_source


def band_units(src):
    try:
        if src.units and src.units[0]:
            return src.units[0]
    except (AttributeError, IndexError):
        pass

    tags = {}
    tags.update(src.tags())
    tags.update(src.tags(1))
    for key, value in tags.items():
        if str(key).strip().lower() in {"units", "unit"} and value:
            return str(value)
    return None


def require_crs(src, path: Path):
    if src.crs is None:
        raise RuntimeError(f"{path.name} does not have a CRS.")
    return src.crs


def target_grid_for(paths):
    """Retain the source CRS; align mosaics only within that CRS."""
    bounds = []
    resolutions = []
    native_crs = None
    first_grid = None
    identical_grids = True
    for path in paths:
        with rasterio.open(path) as src:
            src_crs = require_crs(src, path)
            if native_crs is None:
                native_crs = src_crs
            elif src_crs != native_crs:
                raise RuntimeError(
                    f"Cannot merge {path.name}: its CRS ({src_crs}) differs from "
                    f"the group's original CRS ({native_crs}). Process different "
                    "CRSs in separate folders; no CRS conversion is performed."
                )
            grid = (src.transform, src.width, src.height)
            if first_grid is None:
                first_grid = grid
            else:
                identical_grids = identical_grids and grid == first_grid
            bounds.append(tuple(src.bounds))
            xres = math.hypot(src.transform.a, src.transform.d)
            yres = math.hypot(src.transform.b, src.transform.e)
            if math.isfinite(xres) and math.isfinite(yres) and xres > 0 and yres > 0:
                resolutions.append((xres, yres))
            else:
                raise RuntimeError(f"{path.name} has an invalid pixel resolution.")

    if not resolutions or first_grid is None:
        raise RuntimeError("Could not determine an output resolution.")

    if identical_grids:
        return (*first_grid, native_crs)

    left = min(item[0] for item in bounds)
    bottom = min(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    top = max(item[3] for item in bounds)
    xres = min(item[0] for item in resolutions)
    yres = min(item[1] for item in resolutions)
    width = max(1, int(math.ceil((right - left) / xres - 1e-9)))
    height = max(1, int(math.ceil((top - bottom) / yres - 1e-9)))
    return from_origin(left, top, xres, yres), width, height, native_crs


def reproject_to_grid(source_data, src, dst_transform, dst_width, dst_height, dst_crs):
    if src.crs != dst_crs:
        raise RuntimeError("Grid alignment requires matching source and output CRSs.")
    if (src.transform, src.width, src.height) == (dst_transform, dst_width, dst_height):
        return source_data
    dest = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=source_data,
        destination=dest,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    dest[~np.isfinite(dest) | (dest == 0)] = np.nan
    return dest


def write_float_tif(path: Path, data, transform, crs, *, units=None):
    data = data.astype(np.float32, copy=False)
    data[~np.isfinite(data) | (data == 0)] = np.nan
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if units:
            dst.update_tags(1, units=units)


def merge_landsat_to_kelvin_native_crs(landsat_paths, output_path: Path):
    transform, width, height, native_crs = target_grid_for(landsat_paths)
    print(f"  Retaining original CRS: {native_crs}")
    value_sum = np.zeros((height, width), dtype=np.float64)
    value_count = np.zeros((height, width), dtype=np.uint16)

    for path in landsat_paths:
        print(f"Processing Landsat: {path.name}")
        with rasterio.open(path) as src:
            require_crs(src, path)
            kelvin = read_band_with_nan(src, scale=LANDSAT_SCALE, offset=LANDSAT_OFFSET)
            repro = reproject_to_grid(kelvin, src, transform, width, height, native_crs)
        valid = np.isfinite(repro) & (repro != 0)
        value_sum[valid] += repro[valid]
        value_count[valid] += 1

    merged = np.full((height, width), np.nan, dtype=np.float32)
    valid = value_count > 0
    merged[valid] = (value_sum[valid] / value_count[valid]).astype(np.float32)
    write_float_tif(output_path, merged, transform, native_crs, units="Kelvin")
    print(f"Wrote {output_path.name}")


def merge_ecostress_datetime_native_crs(ecostress_paths, output_path: Path):
    transform, width, height, native_crs = target_grid_for(ecostress_paths)
    print(f"  Retaining original CRS: {native_crs}")
    value_sum = np.zeros((height, width), dtype=np.float64)
    value_count = np.zeros((height, width), dtype=np.uint16)
    units = None

    for path in ecostress_paths:
        print(f"Processing ECOSTRESS: {path.name}")
        with rasterio.open(path) as src:
            require_crs(src, path)
            scale, offset, scale_source, offset_source = ecostress_scale_offset(src)
            if scale_source or offset_source:
                scale_text = f"{scale:g}" if scale_source else "1 (default)"
                offset_text = f"{offset:g}" if offset_source else "0 (default)"
                print(f"  Applying ECOSTRESS scale={scale_text}, offset={offset_text}")
                if scale_source:
                    print(f"    Scale source: {scale_source}")
                if offset_source:
                    print(f"    Offset source: {offset_source}")
            else:
                print("  No non-default ECOSTRESS scale or offset found; values unchanged.")

            if units is None:
                units = band_units(src)

            data = read_band_with_nan(src, scale=scale, offset=offset)
            repro = reproject_to_grid(
                data, src, transform, width, height, native_crs
            )

        valid = np.isfinite(repro) & (repro != 0)
        value_sum[valid] += repro[valid]
        value_count[valid] += 1

    merged = np.full((height, width), np.nan, dtype=np.float32)
    valid = value_count > 0
    merged[valid] = (value_sum[valid] / value_count[valid]).astype(np.float32)

    write_float_tif(output_path, merged, transform, native_crs, units=units)
    print(f"Wrote {output_path.name}")


def main():
    folder = Path(__file__).resolve().parent
    print(f"Working folder: {folder}")
    landsat_paths, ecostress_paths = find_inputs(folder)
    if landsat_paths:
        print("Found Landsat files:")
        for path in landsat_paths:
            print(f"  {path.name}")
        merge_landsat_to_kelvin_native_crs(
            landsat_paths, folder / LANDSAT_OUTPUT_NAME
        )
    else:
        print("No Landsat LC08/LC09 *_ST_B10.TIF files found; skipping Landsat processing.")

    if ecostress_paths:
        print("Found ECOSTRESS files:")
        for path in ecostress_paths:
            print(f"  {path.name}")

        ecostress_groups = {}
        for path in ecostress_paths:
            datetime_key = ecostress_datetime_key(path)
            ecostress_groups.setdefault(datetime_key, []).append(path)

        for datetime_key, group_paths in sorted(ecostress_groups.items()):
            output_path = folder / f"ECOV002_{datetime_key}_processed.tif"
            print(
                f"Stitching {len(group_paths)} ECOSTRESS file(s) for "
                f"{datetime_key} -> {output_path.name}"
            )
            merge_ecostress_datetime_native_crs(group_paths, output_path)
    else:
        print("No ECOSTRESS ECOV002_*.TIF files found; skipping ECOSTRESS processing.")

    if not landsat_paths and not ecostress_paths:
        print("No matching Landsat or ECOSTRESS files were found.")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
