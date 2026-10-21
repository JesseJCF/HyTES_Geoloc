from pathlib import Path
import math
import sys

try:
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy and rasterio. Install them in the Python "
        "environment used to run this script, then run Step1.py again.\n"
        f"Missing dependency: {exc}"
    )


LANDSAT_SCALE = 0.00341802
LANDSAT_OFFSET = 149.0
DST_CRS = CRS.from_epsg(4326)
LANDSAT_OUTPUT_NAME = "Landsat_B10_merged.tif"
ECOSTRESS_OUTPUT_NAME = "ECO_repro.tif"


def is_landsat_b10(path: Path) -> bool:
    name = path.name.upper()
    return (
        path.suffix.upper() in {".TIF", ".TIFF"}
        and name.startswith(("LC08_", "LC09_"))
        and name.endswith(("_ST_B10.TIF", "_ST_B10.TIFF"))
    )


def is_ecostress_lst(path: Path) -> bool:
    name = path.name.upper()
    stem = path.stem.upper()
    return (
        path.suffix.upper() in {".TIF", ".TIFF"}
        and name.startswith("ECO")
        and stem.endswith("_LST")
    )


def find_inputs(folder: Path):
    tif_paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.upper() in {".TIF", ".TIFF"}
    ]
    landsat_paths = sorted(path for path in tif_paths if is_landsat_b10(path))
    ecostress_paths = sorted(path for path in tif_paths if is_ecostress_lst(path))
    if not landsat_paths:
        raise RuntimeError("No LC08/LC09 *_ST_B10.TIF files were found.")
    if not ecostress_paths:
        raise RuntimeError("No ECOSTRESS ECO*_LST.tif file was found.")
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


def read_band_with_nan(src, *, landsat_kelvin: bool) -> np.ndarray:
    data = src.read(1).astype(np.float32)
    invalid = _nodata_mask(data, src.nodata)
    data[invalid] = np.nan
    if landsat_kelvin:
        valid = ~invalid
        data[valid] = data[valid] * LANDSAT_SCALE + LANDSAT_OFFSET
    return data


def require_crs(src, path: Path):
    if src.crs is None:
        raise RuntimeError(f"{path.name} does not have a CRS.")
    return src.crs


def target_grid_for(paths, dst_crs):
    bounds = []
    resolutions = []
    for path in paths:
        with rasterio.open(path) as src:
            src_crs = require_crs(src, path)
            bounds.append(transform_bounds(src_crs, dst_crs, *src.bounds, densify_pts=21))
            dst_transform, _dst_width, _dst_height = calculate_default_transform(
                src_crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
            )
            xres = abs(dst_transform.a)
            yres = abs(dst_transform.e)
            if xres > 0 and yres > 0:
                resolutions.append((xres, yres))

    if not resolutions:
        raise RuntimeError("Could not determine an output resolution.")

    left = min(item[0] for item in bounds)
    bottom = min(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    top = max(item[3] for item in bounds)
    xres = min(item[0] for item in resolutions)
    yres = min(item[1] for item in resolutions)
    width = max(1, int(math.ceil((right - left) / xres)))
    height = max(1, int(math.ceil((top - bottom) / yres)))
    return from_origin(left, top, xres, yres), width, height


def reproject_to_grid(source_data, src, dst_transform, dst_width, dst_height, dst_crs):
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


def merge_landsat_to_kelvin_4326(landsat_paths, output_path: Path):
    transform, width, height = target_grid_for(landsat_paths, DST_CRS)
    value_sum = np.zeros((height, width), dtype=np.float64)
    value_count = np.zeros((height, width), dtype=np.uint16)

    for path in landsat_paths:
        print(f"Processing Landsat: {path.name}")
        with rasterio.open(path) as src:
            require_crs(src, path)
            kelvin = read_band_with_nan(src, landsat_kelvin=True)
            repro = reproject_to_grid(kelvin, src, transform, width, height, DST_CRS)
        valid = np.isfinite(repro) & (repro != 0)
        value_sum[valid] += repro[valid]
        value_count[valid] += 1

    merged = np.full((height, width), np.nan, dtype=np.float32)
    valid = value_count > 0
    merged[valid] = (value_sum[valid] / value_count[valid]).astype(np.float32)
    write_float_tif(output_path, merged, transform, DST_CRS, units="Kelvin")
    print(f"Wrote {output_path.name}")


def merge_ecostress_to_4326(ecostress_paths, output_path: Path):
    transform, width, height = target_grid_for(ecostress_paths, DST_CRS)
    value_sum = np.zeros((height, width), dtype=np.float64)
    value_count = np.zeros((height, width), dtype=np.uint16)

    for path in ecostress_paths:
        print(f"Processing ECOSTRESS: {path.name}")
        with rasterio.open(path) as src:
            require_crs(src, path)
            data = read_band_with_nan(src, landsat_kelvin=False)
            repro = reproject_to_grid(data, src, transform, width, height, DST_CRS)
        valid = np.isfinite(repro) & (repro != 0)
        value_sum[valid] += repro[valid]
        value_count[valid] += 1

    merged = np.full((height, width), np.nan, dtype=np.float32)
    valid = value_count > 0
    merged[valid] = (value_sum[valid] / value_count[valid]).astype(np.float32)
    write_float_tif(output_path, merged, transform, DST_CRS)
    print(f"Wrote {output_path.name}")


def main():
    folder = Path(__file__).resolve().parent
    print(f"Working folder: {folder}")
    landsat_paths, ecostress_paths = find_inputs(folder)
    print("Found Landsat files:")
    for path in landsat_paths:
        print(f"  {path.name}")
    print("Found ECOSTRESS files:")
    for path in ecostress_paths:
        print(f"  {path.name}")

    merge_landsat_to_kelvin_4326(landsat_paths, folder / LANDSAT_OUTPUT_NAME)
    merge_ecostress_to_4326(ecostress_paths, folder / ECOSTRESS_OUTPUT_NAME)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)