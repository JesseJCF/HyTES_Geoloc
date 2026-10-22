"""
Synthetic warp: build one HyTES-like flightline from the basemap, "messed up"
by a semi-random polynomial warp, so a registration algorithm can be tested
against a known answer.

The strip copies a real HyTES line (--template): same CRS, grid, size, dtype,
file layout and swath footprint (template fill 0 -> 0, template NaN -> NaN).
Its content is the basemap, in its own units (F), resampled onto that grid,
but taken from DISPLACED positions, so the strip's georeference is wrong by a
known amount. That displacement is the ground truth written alongside it.

Warp model (backward map, in strip pixel coordinates):
    u = (col - cx) / cx,  v = (row - cy) / cy        (u, v in [-1, 1])
    terms = P_i(u) * P_j(v) for every i + j <= --order
    src_col = col + sum(cx_k * terms_k)
    src_row = row + sum(cy_k * terms_k)
P is the Chebyshev polynomial T_n (default) or the plain power x**n. Plain
powers of high order bend almost only near the strip edges; T_n oscillates
evenly n times across [-1, 1], so higher orders give wiggles everywhere.
Strip pixel (row, col) takes the basemap value at the ground position of strip
pixel (src_row, src_col), sampled bilinearly. Coefficients are drawn at random
(seeded), weighted 1 for the translation and 0.6 * decay**(k - 1) for degree
k >= 1, then scaled so the largest displacement over the strip equals
--max-shift-m. Warps that would fold the strip (Jacobian determinant <= 0.2)
are rejected.

The strip only carries the basemap's ~10 m detail upsampled to the HyTES grid
(~3 m), so it is smoother than real HyTES; no noise is added.

Outputs (in synthetic_tests/data, as inputs for the registration tests):
    synthwarp_o<order>_s<seed>.tif               warped strip (basemap units, F; fill 0)
    synthwarp_o<order>_s<seed>_displacement.tif  band 1 = src_col - col,
                                                 band 2 = src_row - row (strip px)
    synthwarp_o<order>_s<seed>_truth.json        seed, coefficients, model, stats

Usage:
    python synthetic_warp.py                            # random seed
    python synthetic_warp.py --seed 7 --max-shift-m 80
    python synthetic_warp.py --order 6                  # wiggly (Chebyshev)
    python synthetic_warp.py --order 6 --decay 1.0      # wigglier: high orders as strong as low
    python synthetic_warp.py --order 2 --basis power    # the original 2nd-order model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BASEMAP = DATA_DIR / "ECOSTRESS_LST_10m_2020-07-01_to_2020-07-30_plus_2022_2023_2024_LST.tif"
BASEMAP_UNITS = "F"               # strip keeps the basemap units (no conversion)
DEFAULT_TEMPLATE = (
    ROOT.parent / "data" / "20210524t215658_LosAngelesCA_L2_B103_V01_L2_LST_geo_UTM.tif"
)
HYTES_FILL = 0.0

BLOCK_ROWS = 128
SCALE_GRID = 201          # points per axis for the max-shift and fold checks
MIN_JACOBIAN_DET = 0.2    # below this the warp (nearly) folds the strip


def term_list(order: int) -> list:
    """(i, j) degree pairs, i + j <= order, ordered 1, u, v, u*u, u*v, v*v, ..."""
    return [(i, k - i) for k in range(order + 1) for i in range(k, -1, -1)]


def term_weights(terms: list, decay: float) -> np.ndarray:
    return np.array([1.0 if i + j == 0 else 0.6 * decay ** (i + j - 1) for i, j in terms])


def basis_1d(x: np.ndarray, order: int, basis: str) -> np.ndarray:
    """P_0..P_order evaluated at x, shape (order + 1, x.size)."""
    out = np.empty((order + 1, x.size))
    out[0] = 1.0
    if order >= 1:
        out[1] = x
    for n in range(2, order + 1):
        out[n] = 2.0 * x * out[n - 1] - out[n - 2] if basis == "chebyshev" else x * out[n - 1]
    return out


def displacement(coef_x, coef_y, terms, basis, rows: np.ndarray, cols: np.ndarray, cx: float, cy: float):
    """dx, dy on the grid rows x cols (1-D arrays), shape (rows.size, cols.size)."""
    order = max(i + j for i, j in terms)
    pu = basis_1d((np.asarray(cols, dtype=np.float64) - cx) / cx, order, basis)
    pv = basis_1d((np.asarray(rows, dtype=np.float64) - cy) / cy, order, basis)
    dx = np.zeros((pv.shape[1], pu.shape[1]))
    dy = np.zeros_like(dx)
    for (i, j), ax, ay in zip(terms, coef_x, coef_y):
        term = np.outer(pv[j], pu[i])
        dx += ax * term
        dy += ay * term
    return dx, dy


def random_coefficients(rng, terms, basis, decay, width, height, max_shift_m, px_m, py_m):
    """Random coefficients scaled so max displacement over the strip = max_shift_m.

    Also returns (min, max) of the Jacobian determinant of the backward map.
    """
    weights = term_weights(terms, decay)
    coef_x = rng.normal(size=len(terms)) * weights
    coef_y = rng.normal(size=len(terms)) * weights
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    rows = np.linspace(0, height - 1, SCALE_GRID)
    cols = np.linspace(0, width - 1, SCALE_GRID)
    dx, dy = displacement(coef_x, coef_y, terms, basis, rows, cols, cx, cy)
    scale = max_shift_m / float(np.max(np.hypot(dx * px_m, dy * py_m)))
    coef_x, coef_y = coef_x * scale, coef_y * scale
    dx, dy = dx * scale, dy * scale
    ddx_dr, ddx_dc = np.gradient(dx, rows, cols)
    ddy_dr, ddy_dc = np.gradient(dy, rows, cols)
    det = (1.0 + ddx_dc) * (1.0 + ddy_dr) - ddx_dr * ddy_dc
    return coef_x, coef_y, (float(det.min()), float(det.max()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a warped HyTES-like strip from the basemap.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                        help="real HyTES line whose grid, footprint and file layout are copied")
    parser.add_argument("--basemap", type=Path, default=BASEMAP)
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    parser.add_argument("--max-shift-m", type=float, default=50.0,
                        help="largest displacement anywhere in the strip, in metres")
    parser.add_argument("--order", type=int, default=2,
                        help="polynomial degree; higher = more bends (default 2)")
    parser.add_argument("--basis", choices=("chebyshev", "power"), default="chebyshev",
                        help="chebyshev spreads wiggles evenly; power bends mostly at the edges")
    parser.add_argument("--decay", type=float, default=2.0 / 3.0,
                        help="weight falloff per degree; 1.0 = high orders as strong as low")
    args = parser.parse_args()
    if args.order < 0:
        parser.error("--order must be >= 0")

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(100000))
    rng = np.random.default_rng(seed)

    with rasterio.open(args.template) as tpl:
        profile = tpl.profile.copy()
        tags = tpl.tags()
        template = tpl.read(1)
        t_transform, t_crs = tpl.transform, tpl.crs
        t_bounds = tpl.bounds
    if t_crs is None or t_crs.is_geographic:
        raise SystemExit("The template must be in a projected CRS (metres), like HyTES UTM lines.")
    height, width = template.shape
    px_m = math.hypot(t_transform.a, t_transform.d)
    py_m = math.hypot(t_transform.b, t_transform.e)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    terms = term_list(args.order)
    coef_x, coef_y, (det_min, det_max) = random_coefficients(
        rng, terms, args.basis, args.decay, width, height, args.max_shift_m, px_m, py_m
    )
    if det_min <= MIN_JACOBIAN_DET:
        raise SystemExit(
            f"seed {seed}: this warp would fold the strip (min Jacobian determinant "
            f"{det_min:.3f} <= {MIN_JACOBIAN_DET}). Use a lower --order, --max-shift-m "
            f"or --decay, or another --seed."
        )

    with rasterio.open(args.basemap) as src:
        # Only the basemap around the strip, padded by the max shift.
        pad = args.max_shift_m + 2.0 * max(px_m, py_m)
        left, bottom, right, top = transform_bounds(
            t_crs, src.crs, t_bounds.left - pad, t_bounds.bottom - pad,
            t_bounds.right + pad, t_bounds.top + pad,
        )
        window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        window = Window(window.col_off - 2, window.row_off - 2, window.width + 4, window.height + 4)
        window = window.intersection(Window(0, 0, src.width, src.height))
        base = src.read(1, window=window).astype(np.float64)
        if src.nodata is not None and not math.isnan(src.nodata):
            base[base == src.nodata] = np.nan
        b_inverse = ~src.window_transform(window)
        to_base_crs = Transformer.from_crs(t_crs, src.crs, always_xy=True)

    strip = np.empty((height, width), dtype=np.float32)
    disp = np.empty((2, height, width), dtype=np.float32)
    cols = np.arange(width, dtype=np.float64)
    for r0 in range(0, height, BLOCK_ROWS):
        r1 = min(height, r0 + BLOCK_ROWS)
        rows = np.arange(r0, r1, dtype=np.float64)
        dx, dy = displacement(coef_x, coef_y, terms, args.basis, rows, cols, cx, cy)
        rows_b, cols_b = np.meshgrid(rows, cols, indexing="ij")
        # Ground position of the displaced pixel centre -> basemap pixel index.
        c_src, r_src = cols_b + dx + 0.5, rows_b + dy + 0.5
        x = t_transform.a * c_src + t_transform.b * r_src + t_transform.c
        y = t_transform.d * c_src + t_transform.e * r_src + t_transform.f
        bx, by = to_base_crs.transform(x, y)
        b_col = b_inverse.a * bx + b_inverse.b * by + b_inverse.c
        b_row = b_inverse.d * bx + b_inverse.e * by + b_inverse.f
        strip[r0:r1] = map_coordinates(
            base, [b_row - 0.5, b_col - 0.5], order=1, mode="constant", cval=np.nan, prefilter=False
        )
        disp[0, r0:r1] = dx
        disp[1, r0:r1] = dy

    footprint = np.isfinite(template) & (template != HYTES_FILL)
    lost = footprint & ~np.isfinite(strip)
    strip[~np.isfinite(strip)] = HYTES_FILL
    strip[template == HYTES_FILL] = HYTES_FILL
    strip[np.isnan(template)] = np.nan

    out_base = DATA_DIR / f"synthwarp_o{args.order}_s{seed}"
    with rasterio.open(f"{out_base}.tif", "w", **profile) as dst:
        dst.write(strip, 1)
        dst.update_tags(**tags)
        dst.update_tags(SYNTHETIC_WARP=f"order={args.order}; basis={args.basis}; seed={seed}; "
                                       f"see {out_base.name}_truth.json")
    profile.update(count=2)
    with rasterio.open(f"{out_base}_displacement.tif", "w", **profile) as dst:
        dst.write(disp)
        dst.set_band_description(1, "dx: source col - strip col (strip px)")
        dst.set_band_description(2, "dy: source row - strip row (strip px)")

    magnitude_px = np.hypot(disp[0], disp[1])[footprint]
    magnitude_m = np.hypot(disp[0] * px_m, disp[1] * py_m)[footprint]
    centre_dx, centre_dy = (
        float(d[0, 0]) for d in displacement(coef_x, coef_y, terms, args.basis, np.array([cy]), np.array([cx]), cx, cy)
    )
    truth = {
        "seed": seed,
        "template": str(args.template),
        "basemap": str(args.basemap),
        "output": f"{out_base}.tif",
        "model": "backward map in strip pixels: src = strip + d(u,v); u=(col-cx)/cx, "
                 "v=(row-cy)/cy; d = sum(coef_k * P_i(u) * P_j(v)) over terms (i, j); "
                 "P = Chebyshev T_n or power x**n per 'basis'",
        "order": args.order,
        "basis": args.basis,
        "decay": args.decay,
        "cx": cx,
        "cy": cy,
        "terms": [list(t) for t in terms],
        "coef_dx_px": coef_x.tolist(),
        "coef_dy_px": coef_y.tolist(),
        "max_shift_m_requested": args.max_shift_m,
        "pixel_size_m": [px_m, py_m],
        "centre_shift_px": [centre_dx, centre_dy],
        "jacobian_det_min": det_min,
        "jacobian_det_max": det_max,
        "shift_px_mean": float(magnitude_px.mean()),
        "shift_px_max": float(magnitude_px.max()),
        "shift_m_mean": float(magnitude_m.mean()),
        "shift_m_max": float(magnitude_m.max()),
        "units": BASEMAP_UNITS,
        "fill_value": HYTES_FILL,
        "fill_fraction": float((strip == HYTES_FILL).mean()),
        "nan_fraction": float(np.isnan(strip).mean()),
        "footprint_pixels_lost_to_basemap_nodata": int(lost.sum()),
    }
    Path(f"{out_base}_truth.json").write_text(json.dumps(truth, indent=2))

    print(f"seed={seed} order={args.order} basis={args.basis} decay={args.decay:.3f} "
          f"terms={len(terms)} | Jacobian det {det_min:.3f}..{det_max:.3f}")
    print(f"shift over footprint: px mean {truth['shift_px_mean']:.2f}, max {truth['shift_px_max']:.2f} | "
          f"m mean {truth['shift_m_mean']:.1f}, max {truth['shift_m_max']:.1f}")
    print(f"centre shift (dx, dy) strip px: ({centre_dx:+.2f}, {centre_dy:+.2f})")
    print(f"fill {truth['fill_fraction']:.1%}, footprint pixels lost to basemap no-data: {int(lost.sum())}")
    print(f"wrote {out_base}.tif (+ _displacement.tif, _truth.json)")


if __name__ == "__main__":
    main()
