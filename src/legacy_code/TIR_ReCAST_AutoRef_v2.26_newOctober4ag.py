from __future__ import annotations

"""
TIR-ReCAST v2.26: validated native-grid precision refinement.

A bounded continuous refinement follows the selected affine correction.
Disjoint fit and validation windows test residual translation and mild affine
distortion on original native-grid features. Unsupported candidates retain
the selected correction. Regional matching also refines fractional peaks
directly on original features. Denser regional controls retain smaller spatial
changes; small-motion gates require additional independent support.
TIR_BRIDGE_PRECISION=0 disables these additions.

Fractional affines use the same pixel-center bilinear sampling for iterative
refinement, residual assessment, and final output. Residuals are measured after
the actual fractional correction, avoiding nearest-neighbor rounding being
counted again as motion. Affine validation compares identical valid support.
Local affine controls use source pixel centers in pixel-corner coordinates.
Final processed and error rasters each sample their original data once.

Each scene retains its original CRS, pixel grid, and dimensions. Reference
data are sampled directly onto that scene grid. Projected displacement uses
native map coordinates, converted to metres using the CRS linear units;
azimuth is clockwise from grid north. No geographic CRS conversion is used.
Already-geographic inputs retain their own CRS and ellipsoid for reporting.

Matching *_err TIFFs receive the final correction selected for *_processed:
the same affine, regional displacement field, and resampling mode. Error
outputs append _err to the processed output stem in ./Polished, preserving
the confidence/retry markers. Matching uses the original input filename.
Both inputs must share CRS, dimensions, and pixel transform. Unchanged
processed passthroughs also copy their error rasters unchanged. Missing,
incompatible, or failed companions are reported in the existing CSV and
console. Valid error zeros are retained unless explicitly declared nodata.
Existing coarse search ranges and confidence thresholds are retained.

After the existing registration, distributed local matches estimate a smooth
displacement field. Regions may move differently; no scene-wide translation
consensus is required for this regional pass. Withheld spatial windows
validate regional improvements. Unsupported regions retain the baseline field.
Accepted regional corrections and fine-texture translations use bilinear
sampling from each original band once, so fractional corrections affect pixels.
All other accepted affines also use bilinear output. No intermediate rasters or JSON files.
TIR_BRIDGE_LOCAL_WARP=0 disables regional refinement (default: 1).
The CSV stores the base affine and a reconstructable local field. A nonlinear
warp cannot be represented by six coefficients: total-affine columns are blank
when a local warp is written; center-shift metrics include the local correction.

Weak/uncertain legacy matches receive an additional +/-12-native-pixel search.
Two denoised, locally normalized gradient scales retain field boundaries.
Disjoint fit/validation cores must support one translation; validation uses
native bilinear coordinates on common support, never a coarse grid.
Strong, coherent legacy results bypass the fine translation pass. Rejected texture
candidates retain the original registration, assessment, and retry behavior.
Accepted texture results bypass broad-feature polishing and affine fitting.
SciPy enables the optional CPU refinement passes; no intermediate rasters are written.
Small/fragmented footprints without distributed usable cores retain the
existing result when neither fine-texture route has sufficient evidence.
TIR_BRIDGE_FINE_TEXTURE=0 disables the fine translation pass. Default: 1.
Set FINE_TEXTURE, LOCAL_WARP, and PRECISION switches to 0 to disable optional refinement.
Texture diagnostics append to the existing CSV. Residuals are measured again
on the final resampled raster with the fine features for accepted candidates.

Existing coarse matching, retry rules, and FFT dimensions are retained.
Rolling scene slots avoid within-chunk idle time. Exact CPU kernels fuse only
per-element preparation; reference-only phase/FFT caches retain original math.
TIR_BRIDGE_ROLLING_LANES=auto (default): rolling GPU slots; original CPU chunks.
Set 0 for original chunks everywhere or 1 for rolling slots everywhere.
TIR_BRIDGE_EXACT_CPU_KERNELS=0 disables the additional native preparation.
TIR_BRIDGE_REFERENCE_PHASE_MB=0 disables reference phase reuse.
Native CPU array preparation, shared reference features, reusable GPU NCC
buffers, and bounded multi-process scheduling. No search or precision reduction.
Optional Numba/threadpoolctl support: install_recast_speed_support.py.
One buffered CSV covers all datasets; no JSON report or Python bytecode files.
Native kernels retain a compiled-code cache outside the workspace by default.
Windows default: %LOCALAPPDATA%/TIR_ReCAST/numba. Linux/macOS use the user cache.
TIR_BRIDGE_NATIVE_DISK_CACHE=0 disables this optional compiled-code cache.
TIR_BRIDGE_NATIVE_CACHE_DIR overrides its location; NUMBA_CACHE_DIR is respected.

CSV conventions:
- For affine-only rows, coefficients map source -> corrected pixel coordinates:
  col_out = a*col_in + b*row_in + c; row_out = d*col_in + e*row_in + f.
- These are the exact coefficients used in src_transform = source_transform * A.
- Translation is net displacement at (width/2, height/2), in pixel-corner
  coordinates, evaluated through source_transform * A in the source CRS.
- Projected distances are grid distances in metres; azimuth is clockwise from
  grid north in [0, 360). Geographic inputs use their native ellipsoid and true
  north. CSV fields identify the basis; north and zero motion use 0.
- The six affine coefficients already include translation; do not apply it twice.
- Passthroughs have identity coefficients and zero translation. OUTPUT_FAILED
  rows have blank final corrections. Invalid movement metrics are flagged.
- Source/reference paths identify duplicate filenames across dataset folders.
- Residual columns follow the reference path: before/after P90 magnitudes in
  native scene pixels, their difference (positive = improvement), paired-window
  count, and support quality. After includes registration and final polishing.
- Before is reconstructed from the inverse final warp and measured final
  local residuals on paired windows; it is not an independent pre-fit match.
- Unavailable residual estimates stay blank, not zero. Unmeasured passthroughs
  retain blank magnitudes and zero applied improvement. No output means no
  reported after residual or improvement.
- Completed rows are queued in memory without CSV serialization during matching.
- Each dataset's rows are written once at its end, in completion order.
- input_index retains discovery order. One CSV is replaced on each run.
- Handled exits flush queued rows; a forced termination can lose the current batch.
- CSV serialization/write timing includes opening, flushing, and closing.
- Native warm-up and CSV writing remain included in the throughput metric.
- TIR_BRIDGE_PROFILE=1 adds stage timings to this same CSV; default is off.
Existing cache folders/JSON files are not recursively removed.

The script reads immediate child dataset folders containing ECOSTRESS
*_processed TIFFs and one Landsat reference per folder. Dataset folders are
processed sequentially while their scenes retain within-folder parallelism. It
estimates a guarded affine registration, assesses residual error, composes a
polishing translation, and resamples each original raster only once. Each
dataset receives its own ./Polished folder. One workspace-level CSV,
command-line progress, and final statistics cover all folders as one batch. If no child dataset exists,
the script retains its original single-folder behavior. Optional CuPy
acceleration moves the FFT-heavy registration and residual-matching kernels to
an NVIDIA GPU.
Multiple non-blocking CUDA streams can overlap independent scene work, while an
optimized SciPy/NumPy CPU path remains available automatically. Per-scene
preprocessing is reused by low-confidence retries. Identical native searches
reuse their exact results only for unchanged prepared arrays, identical search
settings, and a stable compute backend. Deterministic downstream stages are
memoized only for the exact same global shift and matcher mode. Retry outputs
carry an _01_ marker. Coarse matching, FFT setup, and QA window selection retain
the existing search ranges and decision thresholds.
"""

import sys
import os

# Disable bytecode for this process and spawned children.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache, wraps
from operator import itemgetter
from pathlib import Path
import csv
import atexit
import multiprocessing
import pickle
from queue import Empty
import math
import shutil
import tempfile
import threading
import time
import weakref
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_NUMPY_PRELOADED = "numpy" in sys.modules

# Prevent nested BLAS/OpenMP oversubscription. Explicit FFT and GDAL thread
# counts are configured below. Existing user settings remain authoritative.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

try:
    import numpy as np
    import rasterio
    from affine import Affine
    from pyproj import CRS as PyprojCRS
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.warp import reproject
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy, rasterio, affine, and pyproj. Install them "
        "in the Python environment used to run this script, then rerun this script.\n"
        f"Missing dependency: {exc}"
    )

try:
    from scipy import fft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    from scipy import ndimage as scipy_ndimage
except ImportError:
    scipy_ndimage = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from threadpoolctl import ThreadpoolController
except ImportError:
    ThreadpoolController = None

GPU_REQUEST_MODE = os.environ.get("TIR_BRIDGE_GPU", "auto").strip().casefold()
GPU_DEVICE_ID = int(os.environ.get("TIR_BRIDGE_GPU_DEVICE", "0"))
cp = None
cp_ndimage = None
CUPY_IMPORT_ERROR = ""
if GPU_REQUEST_MODE not in {"0", "off", "false", "cpu", "none"}:
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cp_ndimage
    except Exception as exc:
        cp = None
        cp_ndimage = None
        CUPY_IMPORT_ERROR = str(exc)


LANDSAT_INPUT_NAME = "Landsat_B10_merged.tif"
ECO_INPUT_SUFFIX = "_processed"
ECO_ERROR_SUFFIX = "_err"
OUTPUT_FOLDER_NAME = "Polished"
OUTPUT_CSV_NAME = "TIR_ReCAST_Corrections.csv"
LEGACY_OUTPUT_CSV_NAME = "TIR_BRIDGE_Polished.csv"
CSV_BUFFER_BYTES = 1024 * 1024
PERFORMANCE_ENABLED = os.environ.get("TIR_BRIDGE_PROFILE", "0").casefold() not in {"0", "off", "false", "no"}
CLEAN_PREVIOUS_SCENE_OUTPUTS = True
OVERWRITE_EXISTING = True
ZERO_IS_NODATA = True

# Confidence geometry: scene convex-hull support versus total center translation.
CONFIDENCE_HULL_FRACTION_SATURATION = 0.55
CONFIDENCE_SHIFT_DECAY_PER_HULL_RADIUS = 4.0
CONFIDENCE_GEOMETRY_EXPONENT = 0.70

ECO_NAN_DILATION_PIXELS_PER_PERCENT = 0.5

# Performance controls. These can also be overridden with environment variables.
MAX_PARALLEL_FILES = max(1, int(os.environ.get("TIR_BRIDGE_MAX_WORKERS", "16")))
GPU_MAX_PARALLEL_FILES = max(1, int(os.environ.get("TIR_BRIDGE_GPU_WORKERS", "8")))
CPU_CORES_PER_SCENE_TARGET = max(1, int(os.environ.get("TIR_BRIDGE_CORES_PER_SCENE", "2")))
CPU_FFT_MAX_WORKERS = max(1, int(os.environ.get("TIR_BRIDGE_FFT_WORKERS", str(os.cpu_count() or 1))))
GDAL_MAX_WARP_THREADS = max(1, int(os.environ.get("TIR_BRIDGE_GDAL_THREADS", "8")))
GDAL_WARP_MEMORY_MB = max(64, int(os.environ.get("TIR_BRIDGE_GDAL_WARP_MB", "512")))
TIFF_MAX_COMPRESSION_THREADS = max(1, int(os.environ.get("TIR_BRIDGE_TIFF_THREADS", "8")))
TIFF_DEFLATE_LEVEL = max(1, min(9, int(os.environ.get("TIR_BRIDGE_DEFLATE_LEVEL", "1"))))
# 0 selects an automatic stream count from VRAM; explicit values are capped at 8.
GPU_JOB_WORKERS_REQUESTED = max(0, int(os.environ.get("TIR_BRIDGE_GPU_JOB_WORKERS", "0")))
LOCAL_WINDOW_WORKERS_REQUESTED = max(
    0,
    int(os.environ.get("TIR_BRIDGE_LOCAL_WORKERS", "0")),
)
RUNTIME_MEMORY_FRACTION = 0.90
CPU_ESTIMATED_BYTES_PER_SCENE_PIXEL = 180
GPU_ESTIMATED_BYTES_PER_SCENE_PIXEL = 104
REFERENCE_GRID_CACHE_MAX_ITEMS = 4
REFERENCE_GRID_CACHE_MAX_BYTES = int(1.5 * 1024 ** 3)
REFERENCE_PYRAMID_CACHE_MAX_FACTOR = 16
QA_REFERENCE_FFT_CACHE_MAX_BYTES = max(
    0, int(os.environ.get("TIR_BRIDGE_QA_REF_FFT_MB", "384"))
) * 1024 ** 2
GPU_NCC_NATIVE_FLOAT32 = os.environ.get(
    "TIR_BRIDGE_GPU_NCC_NATIVE32", "1"
).strip().casefold() not in {"0", "off", "false", "no"}
GPU_GROUPED_DOWNLOADS = os.environ.get(
    "TIR_BRIDGE_GPU_GROUPED_DOWNLOADS", "1"
).strip().casefold() not in {"0", "off", "false", "no"}
GPU_GROUPED_DOWNLOAD_MAX_BYTES = 16 * 1024 ** 2
QA_REUSE_IMAGE_MAX_BYTES = max(
    0, int(os.environ.get("TIR_BRIDGE_QA_REUSE_IMAGE_MB", "64"))
) * 1024 ** 2
GPU_MEMORY_SAFETY_FRACTION = 0.90
GPU_MIN_PHASE_INPUT_PIXELS = 256 * 256
GPU_MIN_ZNCC_INPUT_PIXELS = 256 * 256
GPU_MIN_WARP_PIXELS = 1024 * 1024
GPU_MIN_GRADIENT_PIXELS = 1024 * 1024
GPU_MIN_ROBUST_GRADIENT_PIXELS = 768 * 768
GPU_NCC_WORK_THRESHOLD = 60_000_000
GPU_QA_BATCH_MIN_WINDOWS = 4
GPU_WARP_BLOCK_ROWS = max(256, int(os.environ.get("TIR_BRIDGE_GPU_WARP_BLOCK_ROWS", "1024")))
GPU_FFT_PLAN_CACHE_SIZE = max(
    1,
    int(os.environ.get("TIR_BRIDGE_GPU_PLAN_CACHE_SIZE", "8")),
)
GPU_FFT_PLAN_CACHE_BYTES = max(
    64 * 1024 ** 2,
    int(os.environ.get("TIR_BRIDGE_GPU_PLAN_CACHE_MB", "512")) * 1024 ** 2,
)

GLOBAL_MAX_ECO_SHIFT_PIXELS = 300
COARSE_GLOBAL_DOWNSAMPLE = 16
NATIVE_REFINE_RADIUS_PIXELS = 50
ENABLE_ADAPTIVE_PHASE_REFINEMENT = True
ADAPTIVE_PHASE_DOWNSAMPLE = 4
ADAPTIVE_NCC_DOWNSAMPLE = 2
ADAPTIVE_PYRAMID_REFINE_RADIUS = 2
ADAPTIVE_POST_WARP_MAX_SHIFT_PIXELS = 16
ADAPTIVE_PHASE_MIN_RELATIVE_MARGIN = 0.01
ADAPTIVE_PHASE_MIN_PEAK_ZSCORE = 3.0
ADAPTIVE_MIN_OVERLAP_FRACTION = 0.05
ADAPTIVE_MIN_NATIVE_NCC_SCORE = 0.10
LOW_CONFIDENCE_RETRY_THRESHOLD = max(
    1,
    min(99, int(os.environ.get("TIR_BRIDGE_RETRY_CONFIDENCE", "30"))),
)
RETRY_COARSE_GLOBAL_DOWNSAMPLE = 1
RETRY_NATIVE_SEARCH_RADIUS_PIXELS = GLOBAL_MAX_ECO_SHIFT_PIXELS
RETRY_FORCE_ROBUST_MATCHER = True
COARSE_SEARCH_RADIUS = 4
REFINE_SEARCH_RADIUS = 3
ROBUST_SMOOTH_RADIUS_PIXELS = 12
ROBUST_GLOBAL_DOWNSAMPLE = 4
ROBUST_PYRAMID_REFINE_RADIUS = 4
ROBUST_AGREEMENT_TOLERANCE_PIXELS = 2.0
ROBUST_SCORE_ADVANTAGE = 0.02
ROBUST_SUPPORT_LOSS_TRIGGER = 0.15
ROBUST_MIN_OVERLAP_RATIO = 0.05
LOCAL_WARP_GRID = 5
LOCAL_WARP_RESIDUAL_RADIUS = 12
LOCAL_WARP_MIN_CONTROLS = 6
LOCAL_CLOUD_MIN_VALID_PIXELS = 256
LOCAL_CLOUD_MIN_VALID_FRACTION = 0.05
LOCAL_CLOUD_MIN_OVERLAP_FRACTION = 0.20
LOCAL_CLOUD_MIN_SCORE = 0.05
LOCAL_CLOUD_MAX_RMSE_PIXELS = 3.0
LOCAL_CLOUD_MIN_SPAN_FRACTION = 0.30
LOCAL_CLOUD_MAX_CORNER_DEVIATION_PIXELS = 25.0

# Native translation alternative; existing matcher thresholds stay intact.
FINE_TEXTURE_ENABLED = os.environ.get("TIR_BRIDGE_FINE_TEXTURE", "1").strip().casefold() not in {"0", "off", "false", "no"}
LOCAL_TEXTURE_ENABLED = os.environ.get("TIR_BRIDGE_LOCAL_WARP", "1").strip().casefold() not in {"0", "off", "false", "no"}
PRECISION_ENABLED = os.environ.get("TIR_BRIDGE_PRECISION", "1").strip().casefold() not in {"0", "off", "false", "no"}
FINE_TEXTURE_SEARCH_RADIUS = 12
FINE_TEXTURE_GRID = 4
FINE_TEXTURE_WINDOW_MAX = 192
FINE_TEXTURE_MIN_CORE = 64
FINE_TEXTURE_HALO = 32
FINE_TEXTURE_MIN_FIT_WINDOWS = 4
FINE_TEXTURE_MIN_VALIDATION_WINDOWS = 4
FINE_TEXTURE_MIN_SCORE = 0.18
FINE_TEXTURE_MIN_MARGIN = 0.012
FINE_TEXTURE_MIN_ZSCORE = 2.5
FINE_TEXTURE_CONSENSUS_RADIUS = 0.85
FINE_TEXTURE_MIN_CONSENSUS_FRACTION = 0.70
FINE_TEXTURE_MIN_GAIN = 0.010
FINE_TEXTURE_MAX_LEGACY_LOSS = 0.020

# Iterative refinement: translation -> guarded warp -> residual translation -> guarded warp.
ENABLE_POST_WARP_REFINEMENT = True
POST_WARP_TRANSLATION_MAX_PIXELS = 16
POST_WARP_TRANSLATION_DOWNSAMPLE = 2
SECOND_WARP_MAX_RESIDUAL_PIXELS = 8
SECOND_WARP_CONFIDENCE_MULTIPLIER = 1.5

# Warp confidence increases with nonrigid displacement magnitude.
WARP_VALIDATION_DOWNSAMPLE = 4
WARP_MIN_VALIDATION_WINDOWS = 6
WARP_MAX_NONRIGID_DEVIATION_PIXELS = 25.0
WARP_BASE_MIN_GLOBAL_SCORE_GAIN = 0.0010
WARP_MIN_GLOBAL_SCORE_GAIN_PER_PIXEL = 0.0005
WARP_BASE_MAX_WORSENED_FRACTION = 0.30
WARP_WORSENED_FRACTION_TIGHTEN_PER_PIXEL = 0.010
WARP_MIN_MAX_WORSENED_FRACTION = 0.10
WARP_BASE_MAX_LOCAL_SCORE_DROP = 0.050
WARP_LOCAL_SCORE_DROP_TIGHTEN_PER_PIXEL = 0.002
WARP_MIN_MAX_LOCAL_SCORE_DROP = 0.015
WARP_BASE_MAX_RMSE_PIXELS = 3.0
WARP_RMSE_TIGHTEN_PER_PIXEL = 0.08
WARP_MIN_MAX_RMSE_PIXELS = 1.25
WARP_BASE_MIN_MEAN_CONTROL_SCORE = 0.05
WARP_MIN_MEAN_CONTROL_SCORE_PER_PIXEL = 0.003
WARP_BASE_MIN_Q25_CONTROL_SCORE = 0.00
WARP_MIN_Q25_CONTROL_SCORE_PER_PIXEL = 0.002
WARP_CONTROL_POINTS_PER_5_PIXELS = 1

# Residual translations are also retained only when they improve the aligned image.
RESIDUAL_TRANSLATION_MIN_MAGNITUDE_PIXELS = 0.25
RESIDUAL_TRANSLATION_BASE_MIN_SCORE_GAIN = 0.0010
RESIDUAL_TRANSLATION_MIN_SCORE_GAIN_PER_PIXEL = 0.0005
RESIDUAL_TRANSLATION_BASE_MAX_WORSENED_FRACTION = 0.25
RESIDUAL_TRANSLATION_WORSENED_TIGHTEN_PER_PIXEL = 0.008
RESIDUAL_TRANSLATION_MIN_MAX_WORSENED_FRACTION = 0.10
RESIDUAL_TRANSLATION_BASE_MAX_LOCAL_SCORE_DROP = 0.050
RESIDUAL_TRANSLATION_LOCAL_DROP_TIGHTEN_PER_PIXEL = 0.002
RESIDUAL_TRANSLATION_MIN_MAX_LOCAL_SCORE_DROP = 0.015

ALIGNMENT_METHOD = (
    f"hybrid adaptive {COARSE_GLOBAL_DOWNSAMPLE}x/{ADAPTIVE_PHASE_DOWNSAMPLE}x "
    "masked phase plus factor-2/native NCC with guarded full-range native-FFT fallback / cloud-tolerant "
    f"radius-{ROBUST_SMOOTH_RADIUS_PIXELS} smoothed gradient + "
    f"{ROBUST_GLOBAL_DOWNSAMPLE}x exact masked ZNCC + iterative guarded "
    "translation-warp-translation-warp refinement; low-confidence retry skips "
    "the 16x catch and performs full-range native plus forced robust matching"
)

GPU_STATE_LOCK = threading.Lock()
REFERENCE_GRID_CACHE_LOCK = threading.Lock()

ACTIVE_SCENE_WORKERS = 1
ACTIVE_CPU_FFT_WORKERS = 1
ACTIVE_GDAL_WARP_THREADS = 1
ACTIVE_TIFF_THREADS = 1
ACTIVE_LOCAL_WINDOW_WORKERS = 1
ACTIVE_LOCAL_FFT_WORKERS = 1
AVAILABLE_MEMORY_BYTES = 0
GPU_AVAILABLE = False
GPU_RUNTIME_DISABLED = False
GPU_DEVICE_NAME = ""
GPU_TOTAL_MEMORY_BYTES = 0
GPU_EXECUTOR = None
GPU_JOB_WORKERS_ACTIVE = 1
GPU_THREAD_STATE = threading.local()
CPU_THREAD_STATE = threading.local()
GPU_FALLBACK_NOTICE_EMITTED = False
GPU_FALLBACK_GENERATION = 0
REFERENCE_GRID_CACHE_BYTES = 0
REFERENCE_GRID_CACHE_HITS = 0
REFERENCE_GRID_CACHE_MISSES = 0
REFERENCE_GRID_CACHE = OrderedDict()
REFERENCE_GRID_INFLIGHT = {}
REFERENCE_GRID_ERRORS = {}
BATCH_OUTPUTS_PRE_CLEANED = False


class GPUInsufficientMemory(RuntimeError):
    pass


def automatic_gpu_job_workers(total_memory_bytes: int) -> int:
    """Choose conservative CUDA stream concurrency from VRAM."""
    if GPU_JOB_WORKERS_REQUESTED > 0:
        return max(1, min(8, int(GPU_JOB_WORKERS_REQUESTED)))
    gib = float(total_memory_bytes) / float(1024 ** 3)
    if gib >= 48.0:
        return 4
    if gib >= 20.0:
        return 3
    if gib >= 10.0:
        return 2
    return 1


def _configure_gpu_worker_job() -> None:
    """Initialize one CUDA thread, stream, and per-thread cuFFT plan cache."""
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        stream = cp.cuda.Stream(non_blocking=True)
        GPU_THREAD_STATE.stream = stream
        cache = cp.fft.config.get_plan_cache()
        try:
            cache.set_size(GPU_FFT_PLAN_CACHE_SIZE)
        except Exception:
            pass
        try:
            per_thread_cache = max(
                64 * 1024 ** 2,
                GPU_FFT_PLAN_CACHE_BYTES // max(1, GPU_JOB_WORKERS_ACTIVE),
            )
            cache.set_memsize(per_thread_cache)
        except Exception:
            pass
        with stream:
            cp.zeros(1, dtype=cp.float32)
        stream.synchronize()


def _gpu_prewarm_barrier_job(barrier: threading.Barrier) -> None:
    barrier.wait(timeout=60.0)


def _execute_gpu_job(function: Callable, args: tuple, submitted=0.0, metrics=None):
    """Run one job on the CUDA stream assigned to the executor thread."""
    assert cp is not None
    started = time.perf_counter()
    if metrics is not None:
        metrics["gpu_queue_seconds"] = metrics.get("gpu_queue_seconds", 0.0) + max(0.0, started - submitted)
    stream = getattr(GPU_THREAD_STATE, "stream", None)
    if stream is None:
        _configure_gpu_worker_job()
        stream = GPU_THREAD_STATE.stream
    with cp.cuda.Device(GPU_DEVICE_ID):
        with stream:
            result = function(*args)
        # Every current GPU job returns host data. Synchronize only this stream,
        # allowing independent jobs on other streams to continue concurrently.
        stream.synchronize()
        if metrics is not None:
            metrics["gpu_job_seconds"] = metrics.get("gpu_job_seconds", 0.0) + time.perf_counter() - started
            metrics["gpu_jobs"] = metrics.get("gpu_jobs", 0) + 1
        return result


def initialize_gpu_runtime() -> None:
    global GPU_AVAILABLE, GPU_DEVICE_NAME, GPU_TOTAL_MEMORY_BYTES, GPU_EXECUTOR
    global GPU_JOB_WORKERS_ACTIVE, CUPY_IMPORT_ERROR
    if cp is None or cp_ndimage is None:
        return
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if GPU_DEVICE_ID < 0 or GPU_DEVICE_ID >= device_count:
            raise RuntimeError(
                f"requested CUDA device {GPU_DEVICE_ID}, but {device_count} device(s) are available"
            )
        with cp.cuda.Device(GPU_DEVICE_ID):
            properties = cp.cuda.runtime.getDeviceProperties(GPU_DEVICE_ID)
            name = properties.get("name", "CUDA GPU")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            _free, total = cp.cuda.runtime.memGetInfo()
            GPU_DEVICE_NAME = str(name)
            GPU_TOTAL_MEMORY_BYTES = int(total)
        GPU_JOB_WORKERS_ACTIVE = automatic_gpu_job_workers(GPU_TOTAL_MEMORY_BYTES)
        GPU_EXECUTOR = ThreadPoolExecutor(
            max_workers=GPU_JOB_WORKERS_ACTIVE,
            thread_name_prefix="tir_bridge_gpu",
            initializer=_configure_gpu_worker_job,
        )
        GPU_AVAILABLE = True
        # Force creation of every worker so each stream and FFT cache is warm
        # before scene processing begins.
        barrier = threading.Barrier(GPU_JOB_WORKERS_ACTIVE)
        futures = [
            GPU_EXECUTOR.submit(_gpu_prewarm_barrier_job, barrier)
            for _ in range(GPU_JOB_WORKERS_ACTIVE)
        ]
        for future in futures:
            future.result()
    except Exception as exc:
        GPU_AVAILABLE = False
        GPU_DEVICE_NAME = ""
        GPU_TOTAL_MEMORY_BYTES = 0
        GPU_JOB_WORKERS_ACTIVE = 1
        CUPY_IMPORT_ERROR = str(exc)
        if GPU_EXECUTOR is not None:
            GPU_EXECUTOR.shutdown(wait=True, cancel_futures=False)
            GPU_EXECUTOR = None


def gpu_is_active() -> bool:
    return bool(GPU_AVAILABLE and not GPU_RUNTIME_DISABLED and GPU_EXECUTOR is not None)


def record_gpu_fallback(exc: Exception) -> None:
    global GPU_RUNTIME_DISABLED, GPU_FALLBACK_NOTICE_EMITTED, GPU_FALLBACK_GENERATION
    with GPU_STATE_LOCK:
        GPU_FALLBACK_GENERATION += 1
    out_of_memory_type = ()
    if cp is not None:
        out_of_memory_type = getattr(cp.cuda.memory, "OutOfMemoryError", ())
    if out_of_memory_type and isinstance(exc, out_of_memory_type):
        try:
            with cp.cuda.Device(GPU_DEVICE_ID):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.fft.config.get_plan_cache().clear()
        except Exception:
            pass
        return
    if isinstance(exc, GPUInsufficientMemory):
        return
    with GPU_STATE_LOCK:
        GPU_RUNTIME_DISABLED = True
        if not GPU_FALLBACK_NOTICE_EMITTED:
            GPU_FALLBACK_NOTICE_EMITTED = True
            print(
                f"GPU acceleration disabled after a CUDA error; continuing on CPU: {exc}",
                file=sys.stderr,
                flush=True,
            )


def submit_gpu_job(function: Callable, *args):
    if not gpu_is_active():
        raise RuntimeError("GPU acceleration is unavailable.")
    metrics = getattr(CPU_THREAD_STATE, "performance", None)
    return GPU_EXECUTOR.submit(_execute_gpu_job, function, args, time.perf_counter(), metrics).result()


def shutdown_gpu_runtime() -> None:
    global GPU_EXECUTOR
    executor = GPU_EXECUTOR
    GPU_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)


if os.environ.get("TIR_BRIDGE_PROCESS_LANE", "0") != "1":
    initialize_gpu_runtime()


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

    def progress(self, current: int, total: int, message: str = ""):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        fraction = current / total
        bar_width = max(20, min(40, self.width - 48))
        filled = int(round(bar_width * fraction))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[{self.elapsed()}] [{bar}] {current:>3}/{total:<3} "
            f"{100.0 * fraction:6.2f}% {message}",
            flush=True,
        )


def format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def available_system_memory_bytes() -> int:
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except Exception:
        return 0


def largest_input_pixel_count(paths: Sequence[Path]) -> int:
    largest = 0
    for path in paths:
        try:
            with rasterio.open(path) as src:
                largest = max(largest, int(src.width) * int(src.height))
        except Exception:
            continue
    return max(1, largest)


def configure_runtime(paths: Sequence[Path], largest_pixels: Optional[int] = None) -> int:
    global ACTIVE_SCENE_WORKERS, ACTIVE_CPU_FFT_WORKERS
    global ACTIVE_GDAL_WARP_THREADS, ACTIVE_TIFF_THREADS
    global ACTIVE_LOCAL_WINDOW_WORKERS, ACTIVE_LOCAL_FFT_WORKERS
    global AVAILABLE_MEMORY_BYTES

    logical_cores = max(1, int(os.cpu_count() or 1))
    AVAILABLE_MEMORY_BYTES = available_system_memory_bytes()
    if largest_pixels is None:
        largest_pixels = largest_input_pixel_count(paths)
    bytes_per_pixel = (
        GPU_ESTIMATED_BYTES_PER_SCENE_PIXEL
        if gpu_is_active()
        else CPU_ESTIMATED_BYTES_PER_SCENE_PIXEL
    )
    estimated_scene_bytes = max(512 * 1024 ** 2, largest_pixels * bytes_per_pixel)
    if AVAILABLE_MEMORY_BYTES > 0:
        memory_budget = int(AVAILABLE_MEMORY_BYTES * RUNTIME_MEMORY_FRACTION)
        memory_limit = max(1, memory_budget // max(1, estimated_scene_bytes))
    else:
        memory_limit = MAX_PARALLEL_FILES

    if gpu_is_active():
        core_limit = min(GPU_MAX_PARALLEL_FILES, logical_cores)
    else:
        core_limit = max(1, logical_cores // CPU_CORES_PER_SCENE_TARGET)

    workers = max(1, min(
        len(paths),
        MAX_PARALLEL_FILES,
        core_limit,
        memory_limit,
    ))
    ACTIVE_SCENE_WORKERS = workers
    threads_per_scene = max(1, logical_cores // workers)
    ACTIVE_CPU_FFT_WORKERS = max(1, min(CPU_FFT_MAX_WORKERS, threads_per_scene))
    ACTIVE_GDAL_WARP_THREADS = max(1, min(GDAL_MAX_WARP_THREADS, threads_per_scene))
    ACTIVE_TIFF_THREADS = max(1, min(TIFF_MAX_COMPRESSION_THREADS, threads_per_scene))
    automatic_local_workers = min(8, max(1, threads_per_scene))
    if gpu_is_active() and workers >= 4:
        # Multiple scene workers already supply CPU parallelism. Avoid another
        # layer of Python window workers competing with CUDA kernel launches;
        # each scene keeps its configured FFT threads. Explicit overrides win.
        automatic_local_workers = 1
    requested_local_workers = (
        LOCAL_WINDOW_WORKERS_REQUESTED
        if LOCAL_WINDOW_WORKERS_REQUESTED > 0
        else automatic_local_workers
    )
    ACTIVE_LOCAL_WINDOW_WORKERS = max(
        1,
        min(
            8,
            requested_local_workers,
            max(1, threads_per_scene),
            logical_cores,
        ),
    )
    ACTIVE_LOCAL_FFT_WORKERS = max(
        1,
        min(
            CPU_FFT_MAX_WORKERS,
            max(1, threads_per_scene // ACTIVE_LOCAL_WINDOW_WORKERS),
        ),
    )
    return workers


@lru_cache(maxsize=256)
def fast_fft_length(value: int) -> int:
    target = max(1, int(value))
    if scipy_fft is not None:
        try:
            return int(scipy_fft.next_fast_len(target, real=True))
        except TypeError:
            return int(scipy_fft.next_fast_len(target))
    return 1 << (target - 1).bit_length()


@lru_cache(maxsize=128)
def linear_fft_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    height, width = shape
    return (
        fast_fft_length(2 * int(height) - 1),
        fast_fft_length(2 * int(width) - 1),
    )


def active_fft_workers() -> int:
    return max(
        1,
        int(
            getattr(
                CPU_THREAD_STATE,
                "fft_workers",
                ACTIVE_CPU_FFT_WORKERS,
            )
        ),
    )


def configure_local_cpu_worker() -> None:
    CPU_THREAD_STATE.fft_workers = ACTIVE_LOCAL_FFT_WORKERS
    if globals().get("_PROCESS_LANE", False):
        _configure_lane_scene_thread()


def cpu_rfft2(
    data: np.ndarray,
    shape: Tuple[int, int],
    axes=(-2, -1),
    overwrite: bool = False,
) -> np.ndarray:
    if scipy_fft is not None:
        return scipy_fft.rfft2(
            data,
            s=shape,
            axes=axes,
            workers=active_fft_workers(),
            overwrite_x=bool(overwrite),
        )
    return np.fft.rfft2(data, s=shape, axes=axes)


def cpu_irfft2(
    data: np.ndarray,
    shape: Tuple[int, int],
    axes=(-2, -1),
    overwrite: bool = False,
) -> np.ndarray:
    if scipy_fft is not None:
        return scipy_fft.irfft2(
            data,
            s=shape,
            axes=axes,
            workers=active_fft_workers(),
            overwrite_x=bool(overwrite),
        )
    return np.fft.irfft2(data, s=shape, axes=axes)


def gpu_effective_free_bytes() -> int:
    if cp is None:
        return 0
    free, _total = cp.cuda.runtime.memGetInfo()
    reusable = 0
    try:
        reusable = int(cp.get_default_memory_pool().free_bytes())
    except Exception:
        pass
    return int(free) + reusable


def require_gpu_memory(estimated_bytes: int) -> None:
    available = gpu_effective_free_bytes()
    safe_available = int(available * GPU_MEMORY_SAFETY_FRACTION)
    if int(estimated_bytes) > safe_available:
        raise GPUInsufficientMemory(
            f"CUDA operation requires approximately {format_bytes(estimated_bytes)}; "
            f"safe available memory is {format_bytes(safe_available)}."
        )


def phase_gpu_memory_estimate(fft_shape: Tuple[int, int], input_pixels: int) -> int:
    fft_pixels = int(fft_shape[0]) * int(fft_shape[1])
    return int(26 * fft_pixels + 12 * int(input_pixels))


def zncc_gpu_memory_estimate(fft_shape: Tuple[int, int], input_pixels: int) -> int:
    fft_pixels = int(fft_shape[0]) * int(fft_shape[1])
    return int(62 * fft_pixels + 40 * int(input_pixels))


def qa_batch_gpu_memory_estimate(
    batch_size: int,
    input_shape: Tuple[int, int],
    fft_shape: Tuple[int, int],
) -> int:
    input_pixels = int(batch_size) * int(input_shape[0]) * int(input_shape[1])
    fft_pixels = int(batch_size) * int(fft_shape[0]) * int(fft_shape[1])
    return int(58 * fft_pixels + 40 * input_pixels)


def warp_gpu_memory_estimate(shape: Tuple[int, int], block_rows: int) -> int:
    height, width = shape
    pixels = int(height) * int(width)
    block_pixels = min(int(height), int(block_rows)) * int(width)
    return int(10 * pixels + 42 * block_pixels)


def warp_pair_gpu_memory_estimate(shape: Tuple[int, int], block_rows: int) -> int:
    height, width = shape
    pixels = int(height) * int(width)
    block_pixels = min(int(height), int(block_rows)) * int(width)
    return int(20 * pixels + 42 * block_pixels)


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
    method: str = ""


@dataclass
class PeakDiagnostics:
    result: ShiftResult
    peak_margin: float
    peak_zscore: float
    overlap_fraction: float
    search_edge_hit: bool
    ncc_scores: Dict[Tuple[int, int], Tuple[float, int]] = field(
        default_factory=dict
    )


@dataclass
class TranslationChoice:
    result: ShiftResult
    cloud_tolerant: bool
    method: str
    legacy_feature_score: float
    robust_feature_score: float


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
    score_q25: float = math.nan
    min_score: float = math.nan

    @classmethod
    def from_translation(cls, shift: ShiftResult, method: str = "translation fallback"):
        return cls(
            float(shift.row_shift), 0.0, 0.0,
            float(shift.col_shift), 0.0, 0.0,
            0, 0.0, float(shift.score), method,
        )

    @classmethod
    def from_pixel_affine(
        cls,
        pixel_affine: Affine,
        method: str,
        control_points: int = 0,
        rmse_px: float = 0.0,
        mean_score: float = math.nan,
        score_q25: float = math.nan,
        min_score: float = math.nan,
    ):
        return cls(
            float(pixel_affine.f),
            float(pixel_affine.d),
            float(pixel_affine.e - 1.0),
            float(pixel_affine.c),
            float(pixel_affine.a - 1.0),
            float(pixel_affine.b),
            int(control_points),
            float(rmse_px),
            float(mean_score),
            method,
            float(score_q25),
            float(min_score),
        )

    def is_translation_only(self, tolerance: float = 1e-10) -> bool:
        return all(
            abs(value) <= tolerance
            for value in (
                self.row_col_coef,
                self.row_row_coef,
                self.col_col_coef,
                self.col_row_coef,
            )
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


@dataclass
class CachedRaster:
    data: np.ndarray
    transform: Affine
    crs: object
    cache_token: object = field(default_factory=object, repr=False, compare=False)


@dataclass
class ReferenceGridProducts:
    data: np.ndarray
    gradient: np.ndarray
    robust_gradient: np.ndarray
    cache_bytes: int
    reference_pyramids: Dict[str, Dict[int, np.ndarray]] = field(
        default_factory=dict, repr=False, compare=False
    )
    reference_pyramid_lock: object = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


@dataclass
class NaNFootprintStats:
    footprint_mask: np.ndarray
    internal_nan_mask: np.ndarray
    footprint_pixels: int
    valid_pixels: int
    internal_nan_pixels: int
    exterior_nan_pixels: int
    raw_nan_percent: float
    internal_nan_percent: float


@dataclass
class RegistrationPreprocessing:
    reference: np.ndarray
    moving: np.ndarray
    geometry: RasterGeometry
    nan_stats: NaNFootprintStats
    eco_nan_dilation_pixels: int
    legacy_reference: np.ndarray
    legacy_moving: np.ndarray
    robust_reference: np.ndarray
    robust_moving: np.ndarray
    support_loss: float
    overlap_pixels: int
    gradient_overlap_pixels: int
    robust_gradient_overlap_pixels: int
    fine_texture_masked_moving: Optional[np.ndarray] = None

    @property
    def fine_texture_moving(self):
        return self.moving if self.fine_texture_masked_moving is None else self.fine_texture_masked_moving


@dataclass
class RegistrationPreparationCache:
    prepared: Optional[RegistrationPreprocessing] = None
    robust_global_result: Optional[ShiftResult] = None
    robust_global_max_shift_pixels: int = -1
    robust_global_feature_score: float = math.nan
    robust_global_feature_count: int = 0
    downstream_key: Optional[Tuple[float, float, float, int, bool]] = None
    downstream_warp_model: Optional[WarpModel] = None
    downstream_stage_tail: Tuple[str, ...] = ()
    assessment_affine_key: Optional[Tuple[float, float, float, float, float, float]] = None
    assessment_result: Optional["AssessmentResult"] = None
    native_phase_results: Dict[Tuple[object, ...], ShiftResult] = field(default_factory=dict)


@dataclass
class BatchResult:
    input_name: str
    output_name: str
    status: str
    nan_percent: float
    distance_m: float
    azimuth_deg: float
    elapsed_seconds: float
    error: str = ""


@dataclass(frozen=True)
class DatasetBatch:
    folder: Path
    eco_paths: Tuple[Path, ...]

    @property
    def output_folder(self) -> Path:
        return self.folder / OUTPUT_FOLDER_NAME


@dataclass
class AlignmentValidation:
    baseline_score: float
    candidate_score: float
    score_gain: float
    valid_windows: int
    worsened_fraction: float
    median_local_gain: float
    worst_local_gain: float


def find_case_insensitive(folder: Path, filename: str) -> Path:
    target = filename.casefold()
    for path in folder.iterdir():
        if path.is_file() and path.name.casefold() == target:
            return path
    raise RuntimeError(f"Required input was not found: {filename}")


def processed_scene_stem(path: Path) -> str:
    """Remove _processed and normalize legacy _NN_processed scene names."""
    stem = path.stem
    suffix_length = len(ECO_INPUT_SUFFIX)
    if not stem.casefold().endswith(ECO_INPUT_SUFFIX):
        return stem
    core = stem[:-suffix_length]
    if len(core) >= 3 and core[-3] == "_" and core[-2:].isdigit():
        core = core[:-3]
    return core.rstrip("_")


def matching_error_raster(input_path: Path) -> Optional[Path]:
    """Match the raw input stem; preserve legacy scene numbers."""
    if not input_path.stem.casefold().endswith(ECO_INPUT_SUFFIX):
        raise ValueError(f"Expected an {ECO_INPUT_SUFFIX} TIFF: {input_path.name}")
    error_stem = input_path.stem[:-len(ECO_INPUT_SUFFIX)] + ECO_ERROR_SUFFIX
    exact_path = input_path.with_name(error_stem + input_path.suffix)
    if exact_path.is_file():
        return exact_path
    matches = sorted(
        path for path in input_path.parent.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".tif", ".tiff"}
        and path.stem.casefold() == error_stem.casefold()
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous error companion for {input_path.name}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def confidence_filename_code(value: float) -> int:
    """Cap the existing 0-100 confidence at two filename digits."""
    if not math.isfinite(value):
        return 0
    return max(0, min(99, int(round(float(value)))))


def matching_ecostress_inputs(folder: Path) -> List[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".tif", ".tiff"}
        and path.name.casefold().startswith("eco")
        and path.stem.casefold().endswith(ECO_INPUT_SUFFIX)
    )


def find_ecostress_inputs(folder: Path) -> List[Path]:
    paths = matching_ecostress_inputs(folder)
    if not paths:
        raise RuntimeError(
            f"No ECOSTRESS *_processed.tif files were found in dataset folder: {folder}"
        )

    by_scene: Dict[str, Path] = {}
    for path in paths:
        key = processed_scene_stem(path).casefold()
        prior = by_scene.get(key)
        if prior is not None:
            raise RuntimeError(
                "Multiple *_processed TIFFs resolve to the same scene stem: "
                f"{prior.name} and {path.name}. Keep only one before running."
            )
        by_scene[key] = path
    return paths


def discover_dataset_batches(root_folder: Path) -> List[DatasetBatch]:
    child_folders = sorted(
        (
            path
            for path in root_folder.iterdir()
            if path.is_dir()
            and path.name.casefold() != OUTPUT_FOLDER_NAME.casefold()
            and matching_ecostress_inputs(path)
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    dataset_folders = child_folders if child_folders else [root_folder]
    return [
        DatasetBatch(folder=folder, eco_paths=tuple(find_ecostress_inputs(folder)))
        for folder in dataset_folders
    ]


def nodata_to_nan(data: np.ndarray, nodata_value, copy: bool = True) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    out = values.copy() if copy else values
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


def raster_band_to_nan(src) -> np.ndarray:
    """Read band 1 while honoring nodata, alpha, and internal GDAL masks."""
    masked = src.read(1, masked=True, out_dtype="float32")
    out = np.asarray(masked.data, dtype=np.float32)
    if not out.flags.writeable:
        out = out.copy()
    invalid = np.ma.getmaskarray(masked) | ~np.isfinite(out) | (out == 0)
    if src.nodata is not None:
        try:
            nodata_value = float(src.nodata)
            if math.isnan(nodata_value):
                invalid |= np.isnan(out)
            else:
                invalid |= out == nodata_value
        except Exception:
            pass
    out[invalid] = np.nan
    return out


def read_single_band_as_nan(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 1:
            raise RuntimeError(f"{path.name} has no raster bands.")
        return raster_band_to_nan(src)


def load_cached_raster(path: Path) -> CachedRaster:
    """Read a reference raster into RAM once for reuse across the full batch."""
    with rasterio.open(path) as src:
        crs = require_crs(src, path)
        return CachedRaster(
            data=raster_band_to_nan(src),
            transform=src.transform,
            crs=crs,
        )


def nan_percentage(data: np.ndarray) -> float:
    if data.size == 0:
        return 100.0
    return 100.0 * float(np.count_nonzero(~np.isfinite(data))) / float(data.size)


def convex_hull_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return a monotonic-chain convex hull in pixel coordinates."""
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 1:
        return unique

    def cross(origin, point_a, point_b):
        return (
            (point_a[0] - origin[0]) * (point_b[1] - origin[1])
            - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def estimate_scene_footprint(
    data: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the acquired scene footprint while excluding exterior swath triangles."""
    valid = (
        np.isfinite(data)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != data.shape:
        raise RuntimeError("The supplied valid mask does not match the raster shape.")
    if valid.size and (
        valid[0, 0] and valid[0, -1] and valid[-1, 0] and valid[-1, -1]
    ):
        return np.ones(data.shape, dtype=bool)
    if not bool(valid.any()):
        return np.zeros(data.shape, dtype=bool)

    valid_rows = valid.any(axis=1)
    row_indices = np.flatnonzero(valid_rows)
    first_columns = np.argmax(valid, axis=1)[row_indices]
    last_columns = data.shape[1] - np.argmax(valid[:, ::-1], axis=1)[row_indices]

    rows = row_indices.astype(np.float64, copy=False)
    left = first_columns.astype(np.float64, copy=False)
    right = last_columns.astype(np.float64, copy=False)
    points_array = np.empty((row_indices.size * 4, 2), dtype=np.float64)
    points_array[0::4, 0] = left
    points_array[0::4, 1] = rows
    points_array[1::4, 0] = left
    points_array[1::4, 1] = rows + 1.0
    points_array[2::4, 0] = right
    points_array[2::4, 1] = rows
    points_array[3::4, 0] = right
    points_array[3::4, 1] = rows + 1.0
    points = list(zip(points_array[:, 0].tolist(), points_array[:, 1].tolist()))

    hull = convex_hull_2d(points)
    if len(hull) < 3:
        return valid.copy()

    polygon = {
        "type": "Polygon",
        "coordinates": [hull + [hull[0]]],
    }
    footprint = rasterize(
        [(polygon, 1)],
        out_shape=data.shape,
        transform=Affine.identity(),
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    footprint |= valid
    return footprint


def nan_footprint_statistics(data: np.ndarray) -> NaNFootprintStats:
    """Separate missing pixels inside the scene from NaNs outside its footprint."""
    valid = np.isfinite(data)
    valid_pixels = int(np.count_nonzero(valid))
    footprint = estimate_scene_footprint(data, valid_mask=valid)
    internal_nan = footprint & ~valid
    footprint_pixels = int(np.count_nonzero(footprint))
    internal_nan_pixels = footprint_pixels - valid_pixels
    exterior_nan_pixels = int(data.size) - footprint_pixels
    raw_nan_percent = (
        100.0 * float(int(data.size) - valid_pixels) / float(data.size)
        if data.size > 0
        else 100.0
    )
    internal_nan_percent = (
        100.0 * internal_nan_pixels / footprint_pixels
        if footprint_pixels > 0
        else 100.0
    )
    return NaNFootprintStats(
        footprint_mask=footprint,
        internal_nan_mask=internal_nan,
        footprint_pixels=footprint_pixels,
        valid_pixels=valid_pixels,
        internal_nan_pixels=internal_nan_pixels,
        exterior_nan_pixels=exterior_nan_pixels,
        raw_nan_percent=raw_nan_percent,
        internal_nan_percent=internal_nan_percent,
    )


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
    valid_count = int(np.count_nonzero(valid))
    total = int(data.size)
    if valid_count <= 0:
        return f"valid=0 / {total} (0.00%)"
    pct = 100.0 * valid_count / max(1, total)
    return f"valid={valid_count:,} / {total:,} ({pct:.1f}%)"


def automatic_eco_nan_dilation_pixels(nan_percent: float) -> int:
    """Convert internal-scene NaN percentage to a dilation radius."""
    return max(
        0,
        int(
            math.ceil(
                ECO_NAN_DILATION_PIXELS_PER_PERCENT * float(nan_percent)
            )
        ),
    )


def _gradient_magnitude_cpu(data: np.ndarray) -> np.ndarray:
    if (EXACT_CPU_KERNELS_ACTIVE and data.dtype == np.float32
            and data.flags.writeable and data.size >= 1024):
        return _exact_gradient(data, True)
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
    np.multiply(gx, gx, out=gx)
    np.multiply(gy, gy, out=gy)
    np.add(gx, gy, out=gx)
    np.sqrt(gx, out=gx)
    mag = gx.astype(np.float32, copy=False)
    inner = grad[1:-1, 1:-1]
    np.copyto(inner, mag, where=valid)
    grad[~np.isfinite(grad) | (grad == 0)] = np.nan
    return grad


def _gpu_gradient_from_values(values):
    out = cp.full(values.shape, cp.nan, dtype=cp.float32)
    if min(values.shape) >= 3:
        center = values[1:-1, 1:-1]
        north = values[:-2, 1:-1]
        south = values[2:, 1:-1]
        west = values[1:-1, :-2]
        east = values[1:-1, 2:]
        valid = (
            cp.isfinite(center)
            & cp.isfinite(north)
            & cp.isfinite(south)
            & cp.isfinite(west)
            & cp.isfinite(east)
        )
        magnitude = cp.sqrt(
            ((east - west) * 0.5) ** 2 + ((south - north) * 0.5) ** 2
        ).astype(cp.float32)
        out[1:-1, 1:-1] = cp.where(valid, magnitude, cp.nan)
    return cp.where(cp.isfinite(out) & (out != 0), out, cp.nan)


def _gpu_robust_gradient_from_values(values, radius: int):
    valid = cp.isfinite(values)
    window_size = (2 * int(radius)) + 1
    window_area = float(window_size * window_size)
    totals = cp_ndimage.uniform_filter(
        cp.where(valid, values, 0.0).astype(cp.float64),
        size=window_size,
        mode="constant",
        cval=0.0,
    ) * window_area
    counts = cp_ndimage.uniform_filter(
        valid.astype(cp.float64),
        size=window_size,
        mode="constant",
        cval=0.0,
    ) * window_area
    possible = cp_ndimage.uniform_filter(
        cp.ones(values.shape, dtype=cp.float64),
        size=window_size,
        mode="constant",
        cval=0.0,
    ) * window_area
    usable = valid & (counts >= cp.maximum(9.0, 0.20 * possible))
    smooth = cp.where(usable, totals / cp.maximum(counts, 1.0), cp.nan).astype(cp.float32)
    out = cp.full(values.shape, cp.nan, dtype=cp.float32)
    if min(values.shape) >= 3:
        center = smooth[1:-1, 1:-1]
        north = smooth[:-2, 1:-1]
        south = smooth[2:, 1:-1]
        west = smooth[1:-1, :-2]
        east = smooth[1:-1, 2:]
        gradient_valid = (
            cp.isfinite(center)
            & cp.isfinite(north)
            & cp.isfinite(south)
            & cp.isfinite(west)
            & cp.isfinite(east)
        )
        magnitude = cp.sqrt(
            ((east - west) * 0.5) ** 2 + ((south - north) * 0.5) ** 2
        ).astype(cp.float32)
        out[1:-1, 1:-1] = cp.where(gradient_valid, magnitude, cp.nan)
    return cp.where(cp.isfinite(out) & (out != 0), out, cp.nan)


def gpu_download_arrays(*arrays) -> Tuple[np.ndarray, ...]:
    """Download independent outputs with one stream synchronization."""
    assert cp is not None
    total_bytes = sum(int(array.nbytes) for array in arrays)
    if (
        not GPU_GROUPED_DOWNLOADS or len(arrays) < 2
        or total_bytes <= 0 or total_bytes > GPU_GROUPED_DOWNLOAD_MAX_BYTES
    ):
        return tuple(cp.asnumpy(array) for array in arrays)
    outputs = []
    stream = cp.cuda.get_current_stream()
    try:
        # Allocate first; no in-flight copies if pinned allocation fails.
        for array in arrays:
            memory = cp.cuda.alloc_pinned_memory(max(1, int(array.nbytes)))
            outputs.append(np.frombuffer(
                memory, dtype=array.dtype, count=int(array.size)
            ).reshape(array.shape))
    except Exception:
        return tuple(cp.asnumpy(array) for array in arrays)
    try:
        for array, host in zip(arrays, outputs):
            cp.asnumpy(array, stream=stream, out=host, blocking=False)
    except TypeError:
        # Older CuPy: finish any queued copies before synchronous fallback.
        stream.synchronize()
        return tuple(cp.asnumpy(array) for array in arrays)
    finally:
        # Retain sources and pinned destinations until transfers complete.
        stream.synchronize()
    return tuple(outputs)


def _gpu_gradient_job(data: np.ndarray) -> np.ndarray:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(int(data.size) * 36)
        values = cp.asarray(data, dtype=cp.float32)
        return cp.asnumpy(_gpu_gradient_from_values(values))


def _gpu_gradient_pair_job(data: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    assert cp is not None and cp_ndimage is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(int(data.size) * 108)
        values = cp.asarray(data, dtype=cp.float32)
        gradient = _gpu_gradient_from_values(values)
        robust_gradient = _gpu_robust_gradient_from_values(values, radius)
        return gpu_download_arrays(gradient, robust_gradient)


def gradient_magnitude_for_alignment(data: np.ndarray) -> np.ndarray:
    if gpu_is_active() and data.size >= GPU_MIN_GRADIENT_PIXELS:
        try:
            return submit_gpu_job(_gpu_gradient_job, data)
        except Exception as exc:
            record_gpu_fallback(exc)
    return _gradient_magnitude_cpu(data)


def box_sum_same(
    data: np.ndarray,
    radius: int,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    radius = max(0, int(radius))
    values = np.asarray(data, dtype=np.float64) if valid is None else np.asarray(data)
    if radius == 0:
        if valid is None:
            return values.copy()
        out = np.zeros(values.shape, dtype=np.float64)
        np.copyto(out, values, where=valid, casting="unsafe")
        return out
    if (
        NATIVE_KERNELS_ACTIVE and valid is not None
        and values.dtype == np.dtype(np.float32) and values.flags.writeable
        and valid.dtype == np.bool_ and values.size >= 1024
    ):
        return _masked_box_sum_native(values, valid, radius)
    height, width = values.shape
    integral = np.zeros(
        (height + (2 * radius) + 1, width + (2 * radius) + 1),
        dtype=np.float64,
    )
    interior = integral[
        1 + radius : 1 + radius + height,
        1 + radius : 1 + radius + width,
    ]
    if valid is None:
        interior[:] = values
    else:
        # Write the original valid samples directly into the double integral
        # image instead of materializing masked float and double copies first.
        np.copyto(interior, values, where=valid, casting="unsafe")
    # Preserve both cumulative-sum orders while reusing their destination.
    np.cumsum(integral, axis=0, out=integral)
    np.cumsum(integral, axis=1, out=integral)
    size = (2 * radius) + 1
    out = integral[size:, size:] - integral[:-size, size:]
    np.subtract(out, integral[size:, :-size], out=out)
    np.add(out, integral[:-size, :-size], out=out)
    return out


def box_count_same(valid: np.ndarray, radius: int) -> np.ndarray:
    """Exact Boolean box counts with a smaller integer integral image."""
    radius = max(0, int(radius))
    if valid.size > np.iinfo(np.int32).max or valid.dtype != np.bool_:
        return box_sum_same(valid, radius)
    if radius == 0:
        return valid.astype(np.int32)
    if NATIVE_KERNELS_ACTIVE and valid.flags.writeable and valid.size >= 1024:
        return _boolean_box_count_native(valid, radius)
    height, width = valid.shape
    integral = np.zeros(
        (height + 2 * radius + 1, width + 2 * radius + 1),
        dtype=np.int32,
    )
    integral[1 + radius:1 + radius + height,
             1 + radius:1 + radius + width] = valid
    np.cumsum(integral, axis=0, dtype=np.int32, out=integral)
    np.cumsum(integral, axis=1, dtype=np.int32, out=integral)
    size = 2 * radius + 1
    out = integral[size:, size:] - integral[:-size, size:]
    np.subtract(out, integral[size:, :-size], out=out)
    np.add(out, integral[:-size, :-size], out=out)
    return out


@lru_cache(maxsize=64)
def box_axis_counts(length: int, radius: int) -> np.ndarray:
    """Return the exact in-bounds sample count for each 1-D box position."""
    length = max(0, int(length))
    radius = max(0, int(radius))
    indices = np.arange(length, dtype=np.int64)
    counts = (
        np.minimum(length - 1, indices + radius)
        - np.maximum(0, indices - radius)
        + 1
    ).astype(np.int32, copy=False)
    counts.setflags(write=False)
    return counts


@lru_cache(maxsize=64)
def box_possible_same(shape: Tuple[int, int], radius: int) -> np.ndarray:
    """Exact number of in-bounds pixels in each same-sized square window."""
    row_counts = box_axis_counts(int(shape[0]), int(radius))
    col_counts = box_axis_counts(int(shape[1]), int(radius))
    possible = np.multiply.outer(row_counts, col_counts)
    possible.setflags(write=False)
    return possible


def _gpu_dilate_nan_job(
    data: np.ndarray,
    invalid: np.ndarray,
    footprint: Optional[np.ndarray],
    radius: int,
) -> Tuple[np.ndarray, int]:
    assert cp is not None and cp_ndimage is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(int(data.size) * 14)
        data_gpu = cp.asarray(data, dtype=cp.float32)
        invalid_gpu = cp.asarray(invalid, dtype=cp.bool_)
        dilated = cp_ndimage.maximum_filter(
            invalid_gpu,
            size=(2 * int(radius)) + 1,
            mode="constant",
            cval=0,
        ).astype(cp.bool_)
        if footprint is not None:
            dilated &= cp.asarray(footprint, dtype=cp.bool_)
        added = int(cp.count_nonzero(dilated & ~invalid_gpu).item())
        out = data_gpu.copy()
        out[dilated] = cp.nan
        return cp.asnumpy(out), added


def _gpu_prepare_moving_features_job(
    data: np.ndarray,
    invalid: np.ndarray,
    footprint: Optional[np.ndarray],
    radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Dilate nodata and derive both feature rasters in one device residency."""
    assert cp is not None and cp_ndimage is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(int(data.size) * 128)
        values = cp.asarray(data, dtype=cp.float32)
        invalid_gpu = cp.asarray(invalid, dtype=cp.bool_)
        dilated = cp_ndimage.maximum_filter(
            invalid_gpu,
            size=(2 * int(radius)) + 1,
            mode="constant",
            cval=0,
        ).astype(cp.bool_)
        if footprint is not None:
            dilated &= cp.asarray(footprint, dtype=cp.bool_)
        # Keep the count on device while the feature kernels are queued; reading
        # it here would stall this CUDA worker before those kernels can launch.
        added = cp.count_nonzero(dilated & ~invalid_gpu)
        moving = values.copy()
        moving[dilated] = cp.nan
        gradient = _gpu_gradient_from_values(moving)
        robust_gradient = _gpu_robust_gradient_from_values(moving, radius=ROBUST_SMOOTH_RADIUS_PIXELS)
        moving_host, gradient_host, robust_host = gpu_download_arrays(
            moving, gradient, robust_gradient
        )
        return moving_host, gradient_host, robust_host, int(added.item())


def dilate_nan_mask(
    data: np.ndarray,
    radius: int,
    invalid_mask: Optional[np.ndarray] = None,
    footprint_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Dilate selected in-scene NaNs without buffering exterior swath nodata."""
    radius = max(0, int(radius))
    if radius == 0:
        return data, 0

    if invalid_mask is None:
        invalid = ~np.isfinite(data)
    else:
        invalid = np.asarray(invalid_mask, dtype=bool)
        if invalid.shape != data.shape:
            raise RuntimeError("The NaN dilation mask does not match the raster shape.")
        invalid = invalid & ~np.isfinite(data)

    if not bool(invalid.any()):
        return data, 0

    footprint = None
    if footprint_mask is not None:
        footprint = np.asarray(footprint_mask, dtype=bool)
        if footprint.shape != data.shape:
            raise RuntimeError("The scene footprint mask does not match the raster shape.")

    if gpu_is_active() and data.size >= GPU_MIN_ROBUST_GRADIENT_PIXELS:
        try:
            return submit_gpu_job(
                _gpu_dilate_nan_job,
                data,
                invalid,
                footprint,
                radius,
            )
        except Exception as exc:
            record_gpu_fallback(exc)

    if scipy_ndimage is not None:
        dilated_invalid = scipy_ndimage.maximum_filter(
            invalid,
            size=(2 * radius) + 1,
            mode="constant",
            cval=False,
        ).astype(bool, copy=False)
    else:
        dilated_invalid = box_sum_same(invalid.astype(np.float32), radius) > 0.0
    if footprint is not None:
        dilated_invalid &= footprint

    added = int(np.count_nonzero(dilated_invalid & ~invalid))
    out = data.copy()
    out[dilated_invalid] = np.nan
    return out, added


def nan_box_smooth(data: np.ndarray, radius: int) -> np.ndarray:
    """Box-smooth valid pixels without filling cloud/nodata holes."""
    if (EXACT_CPU_KERNELS_ACTIVE and data.dtype == np.float32
            and data.flags.writeable and 1024 <= data.size <= np.iinfo(np.int32).max):
        return _exact_smooth(data, max(0, int(radius)), 0.20)
    valid = np.isfinite(data)
    totals = box_sum_same(data, radius, valid=valid)
    counts = box_count_same(valid, radius)
    possible = box_possible_same(data.shape, radius)
    minimum = np.maximum(9.0, 0.20 * possible)
    usable = valid & (counts >= minimum)
    out = np.full(data.shape, np.nan, dtype=np.float32)
    np.divide(totals, counts, out=out, where=usable)
    out[~usable] = np.nan
    return out


def _gpu_robust_gradient_job(data: np.ndarray, radius: int) -> np.ndarray:
    assert cp is not None and cp_ndimage is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(int(data.size) * 72)
        values = cp.asarray(data, dtype=cp.float32)
        return cp.asnumpy(_gpu_robust_gradient_from_values(values, radius))


def robust_gradient_for_alignment(data: np.ndarray, radius: int) -> np.ndarray:
    if gpu_is_active() and data.size >= GPU_MIN_ROBUST_GRADIENT_PIXELS:
        try:
            return submit_gpu_job(_gpu_robust_gradient_job, data, radius)
        except Exception as exc:
            record_gpu_fallback(exc)
    return _gradient_magnitude_cpu(nan_box_smooth(data, radius))


def gradient_pair_for_alignment(
    data: np.ndarray,
    radius: int,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute legacy and robust gradients with one GPU upload when possible."""
    can_pair = (
        gpu_is_active()
        and data.size >= max(GPU_MIN_GRADIENT_PIXELS, GPU_MIN_ROBUST_GRADIENT_PIXELS)
    )
    if can_pair:
        try:
            return submit_gpu_job(_gpu_gradient_pair_job, data, radius)
        except Exception as exc:
            record_gpu_fallback(exc)
    if (
        local_executor is not None
        and ACTIVE_SCENE_WORKERS == 1
        and ACTIVE_LOCAL_WINDOW_WORKERS > 1
        and data.size >= 256 * 256
    ):
        gradient_future = local_executor.submit(
            gradient_magnitude_for_alignment, data
        )
        robust_future = local_executor.submit(
            robust_gradient_for_alignment, data, radius
        )
        return gradient_future.result(), robust_future.result()
    return (
        gradient_magnitude_for_alignment(data),
        robust_gradient_for_alignment(data, radius),
    )


def prepare_moving_alignment_features(
    data: np.ndarray,
    radius: int,
    invalid_mask: Optional[np.ndarray],
    footprint_mask: Optional[np.ndarray],
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Prepare masked moving data and both alignment features."""
    radius = max(0, int(radius))
    if invalid_mask is None:
        invalid = ~np.isfinite(data)
    else:
        invalid = np.asarray(invalid_mask, dtype=bool)
        if invalid.shape != data.shape:
            raise RuntimeError("The NaN dilation mask does not match the raster shape.")
        invalid = invalid & ~np.isfinite(data)

    footprint = None
    if footprint_mask is not None:
        footprint = np.asarray(footprint_mask, dtype=bool)
        if footprint.shape != data.shape:
            raise RuntimeError("The scene footprint mask does not match the raster shape.")
    can_combine = (
        radius > 0
        and bool(invalid.any())
        and gpu_is_active()
        and data.size >= max(GPU_MIN_GRADIENT_PIXELS, GPU_MIN_ROBUST_GRADIENT_PIXELS)
    )
    if can_combine:
        try:
            return submit_gpu_job(
                _gpu_prepare_moving_features_job,
                data,
                invalid,
                footprint,
                radius,
            )
        except Exception as exc:
            record_gpu_fallback(exc)

    moving, added = dilate_nan_mask(
        data,
        radius,
        invalid_mask=invalid,
        footprint_mask=footprint,
    )
    gradient, robust_gradient = gradient_pair_for_alignment(
        moving,
        ROBUST_SMOOTH_RADIUS_PIXELS,
        local_executor=local_executor,
    )
    return moving, gradient, robust_gradient, added


def support_loss_fraction(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_valid_mask: Optional[np.ndarray] = None,
    moving_valid_mask: Optional[np.ndarray] = None,
    overlap_mask: Optional[np.ndarray] = None,
) -> float:
    """Estimate missing data inside per-row overlap spans, excluding swath exterior."""
    reference_valid = (
        np.isfinite(reference)
        if reference_valid_mask is None
        else np.asarray(reference_valid_mask, dtype=bool)
    )
    moving_valid = (
        np.isfinite(moving)
        if moving_valid_mask is None
        else np.asarray(moving_valid_mask, dtype=bool)
    )
    if reference_valid.shape != reference.shape or moving_valid.shape != moving.shape:
        raise RuntimeError("A support-loss validity mask has an incompatible shape.")
    overlap = (
        reference_valid & moving_valid
        if overlap_mask is None
        else np.asarray(overlap_mask, dtype=bool)
    )
    if overlap.shape != reference.shape:
        raise RuntimeError("The support-loss overlap mask has an incompatible shape.")
    if (EXACT_CPU_KERNELS_ACTIVE and overlap.size >= 1024
            and reference_valid.flags.writeable and overlap.flags.writeable):
        span_pixels, overlap_pixels = _exact_support_counts(reference_valid, overlap)
        if span_pixels <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (overlap_pixels / span_pixels)))
    row_has_overlap = overlap.any(axis=1)
    if not bool(row_has_overlap.any()):
        return 1.0

    first_columns = np.argmax(overlap, axis=1)
    stop_columns = overlap.shape[1] - np.argmax(overlap[:, ::-1], axis=1)
    column_indices = np.arange(overlap.shape[1], dtype=np.int64)[None, :]
    span_pixels = 0
    block_rows = 256
    for row0 in range(0, overlap.shape[0], block_rows):
        row1 = min(overlap.shape[0], row0 + block_rows)
        active = row_has_overlap[row0:row1]
        if not bool(active.any()):
            continue
        in_span = (
            (column_indices >= first_columns[row0:row1, None])
            & (column_indices < stop_columns[row0:row1, None])
            & active[:, None]
        )
        span_pixels += int(np.count_nonzero(reference_valid[row0:row1] & in_span))

    overlap_pixels = int(np.count_nonzero(overlap))
    if span_pixels <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (overlap_pixels / span_pixels)))


def read_eco_scene(eco_path: Path) -> Tuple[np.ndarray, RasterGeometry]:
    with rasterio.open(eco_path) as eco_src:
        eco_crs = require_crs(eco_src, eco_path)
        eco_data = raster_band_to_nan(eco_src)
        geometry = RasterGeometry(
            eco_src.transform,
            eco_crs,
            eco_src.width,
            eco_src.height,
        )
    return eco_data, geometry


def reference_grid_key(
    landsat: CachedRaster,
    geometry: RasterGeometry,
) -> Tuple[object, ...]:
    transform = geometry.transform
    return (
        landsat.cache_token,
        geometry.crs.to_wkt() if hasattr(geometry.crs, "to_wkt") else str(geometry.crs),
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
        int(geometry.width),
        int(geometry.height),
    )


def reserve_reference_pyramid_cache(
    products: ReferenceGridProducts,
) -> ReferenceGridProducts:
    """Include all possible lazy pyramid levels in the existing cache budget."""
    # Reserve once, before cache admission. Later concurrent construction and
    # eviction then need no adjustments to the shared byte counter.
    for source in (products.gradient, products.robust_gradient):
        height, width = source.shape
        factor = 1
        while factor < REFERENCE_PYRAMID_CACHE_MAX_FACTOR:
            height = (height + 1) // 2
            width = (width + 1) // 2
            products.cache_bytes += height * width * np.dtype(np.float32).itemsize
            factor *= 2
    return products


def build_reference_grid_products(
    landsat: CachedRaster,
    geometry: RasterGeometry,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> ReferenceGridProducts:
    destination = np.full(
        (geometry.height, geometry.width),
        np.nan,
        dtype=np.float32,
    )
    reproject(
        source=landsat.data,
        destination=destination,
        src_transform=landsat.transform,
        src_crs=landsat.crs,
        src_nodata=np.nan,
        dst_transform=geometry.transform,
        dst_crs=geometry.crs,
        dst_nodata=np.nan,
        init_dest_nodata=True,
        resampling=Resampling.average,
        num_threads=ACTIVE_GDAL_WARP_THREADS,
        warp_mem_limit=GDAL_WARP_MEMORY_MB,
    )
    destination[~np.isfinite(destination) | (destination == 0)] = np.nan
    gradient, robust_gradient = gradient_pair_for_alignment(
        destination,
        ROBUST_SMOOTH_RADIUS_PIXELS,
        local_executor=local_executor,
    )
    cache_bytes = int(destination.nbytes + gradient.nbytes + robust_gradient.nbytes)
    return reserve_reference_pyramid_cache(ReferenceGridProducts(
        data=destination,
        gradient=gradient,
        robust_gradient=robust_gradient,
        cache_bytes=cache_bytes,
    ))


def get_reference_grid_products(
    landsat: CachedRaster,
    geometry: RasterGeometry,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> ReferenceGridProducts:
    global REFERENCE_GRID_CACHE_BYTES
    global REFERENCE_GRID_CACHE_HITS, REFERENCE_GRID_CACHE_MISSES

    key = reference_grid_key(landsat, geometry)
    with REFERENCE_GRID_CACHE_LOCK:
        cached = REFERENCE_GRID_CACHE.get(key)
        if cached is not None:
            REFERENCE_GRID_CACHE.move_to_end(key)
            REFERENCE_GRID_CACHE_HITS += 1
            return cached
        event = REFERENCE_GRID_INFLIGHT.get(key)
        if event is None:
            event = threading.Event()
            REFERENCE_GRID_INFLIGHT[key] = event
            owner = True
        else:
            owner = False

    if not owner:
        event.wait()
        with REFERENCE_GRID_CACHE_LOCK:
            cached = REFERENCE_GRID_CACHE.get(key)
            if cached is not None:
                REFERENCE_GRID_CACHE.move_to_end(key)
                REFERENCE_GRID_CACHE_HITS += 1
                return cached
            error = REFERENCE_GRID_ERRORS.get(key, "Reference-grid preprocessing failed.")
        raise RuntimeError(error)

    try:
        products = build_reference_grid_products(
            landsat,
            geometry,
            local_executor=local_executor,
        )
    except Exception as exc:
        with REFERENCE_GRID_CACHE_LOCK:
            REFERENCE_GRID_ERRORS[key] = str(exc)
            REFERENCE_GRID_INFLIGHT.pop(key, None)
            event.set()
        raise

    with REFERENCE_GRID_CACHE_LOCK:
        REFERENCE_GRID_CACHE_MISSES += 1
        REFERENCE_GRID_ERRORS.pop(key, None)
        REFERENCE_GRID_CACHE[key] = products
        REFERENCE_GRID_CACHE.move_to_end(key)
        REFERENCE_GRID_CACHE_BYTES += products.cache_bytes
        while (
            len(REFERENCE_GRID_CACHE) > REFERENCE_GRID_CACHE_MAX_ITEMS
            or REFERENCE_GRID_CACHE_BYTES > REFERENCE_GRID_CACHE_MAX_BYTES
        ) and len(REFERENCE_GRID_CACHE) > 1:
            _old_key, old_products = REFERENCE_GRID_CACHE.popitem(last=False)
            REFERENCE_GRID_CACHE_BYTES -= old_products.cache_bytes
        REFERENCE_GRID_INFLIGHT.pop(key, None)
        event.set()
    return products


def clear_reference_grid_cache() -> None:
    """Release products from a completed dataset before changing basemaps."""
    global REFERENCE_GRID_CACHE_BYTES
    with REFERENCE_GRID_CACHE_LOCK:
        if REFERENCE_GRID_INFLIGHT:
            raise RuntimeError(
                "Reference-grid cache cannot be cleared while scene work is active."
            )
        REFERENCE_GRID_CACHE.clear()
        REFERENCE_GRID_ERRORS.clear()
        REFERENCE_GRID_CACHE_BYTES = 0
    clear_shared_reference_features()
    clear_reference_phase_cache()


def nanmean_downsample_2x(data: np.ndarray) -> np.ndarray:
    height, width = data.shape
    if (EXACT_CPU_KERNELS_ACTIVE and data.dtype == np.float32
            and data.flags.c_contiguous and data.flags.writeable
            and width > 2 and data.size >= 1024):
        return _exact_downsample(data)
    if height % 2 == 0 and width % 2 == 0:
        padded = np.asarray(data, dtype=np.float32)
        padded_height = height
        padded_width = width
    else:
        padded_height = height + (height % 2)
        padded_width = width + (width % 2)
        padded = np.full((padded_height, padded_width), np.nan, dtype=np.float32)
        padded[:height, :width] = data
    # NumPy collapses singleton/strided reduction axes differently. Retain the
    # general reducer there; ordinary C-order rasters use paired row sums.
    if padded_width <= 2 or not padded.flags.c_contiguous:
        blocks = padded.reshape(padded_height // 2, 2, padded_width // 2, 2)
        valid = np.isfinite(blocks)
        totals = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
        counts = valid.sum(axis=(1, 3))
        out = np.full(totals.shape, np.nan, dtype=np.float32)
        np.divide(totals, counts, out=out, where=counts > 0)
        return out
    valid = np.isfinite(padded)
    values = np.where(valid, padded, 0.0)
    # Preserve the tuple-axis reducer's (a+b)+(c+d) order and its positive
    # zero identity, while reducing four stride-2 planes with vector ufuncs.
    totals = np.add(values[::2, ::2], values[::2, 1::2], dtype=np.float64)
    lower = np.add(values[1::2, ::2], values[1::2, 1::2], dtype=np.float64)
    np.add(totals, lower, out=totals)
    np.add(totals, 0.0, out=totals)
    counts = np.add(valid[::2, ::2], valid[::2, 1::2], dtype=np.uint8)
    np.add(counts, valid[1::2, ::2], out=counts)
    np.add(counts, valid[1::2, 1::2], out=counts)
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


def cached_reference_pyramid_levels(
    reference: np.ndarray,
    max_factor: int,
) -> Optional[Dict[int, np.ndarray]]:
    """Reuse only immutable feature arrays owned by a resident reference grid."""
    if max_factor <= 1 or max_factor > REFERENCE_PYRAMID_CACHE_MAX_FACTOR:
        return None
    owner = None
    source_kind = ""
    with REFERENCE_GRID_CACHE_LOCK:
        for products in REFERENCE_GRID_CACHE.values():
            for kind in ("gradient", "robust_gradient"):
                if reference is getattr(products, kind):
                    owner = products
                    source_kind = kind
                    break
            if owner is not None:
                break
    if owner is None:
        return None
    # Separate from the LRU lock: different grids can build concurrently, and
    # lookups/evictions do not wait for downsampling. The owner stays alive for
    # this call even if another scene evicts it from the LRU.
    with owner.reference_pyramid_lock:
        levels = owner.reference_pyramids.setdefault(source_kind, {1: reference})
        factor = 1
        while factor < max_factor:
            if factor * 2 not in levels:
                levels[factor * 2] = nanmean_downsample_2x(levels[factor])
            factor *= 2
        return dict(levels)


def build_requested_pyramid_levels(
    reference: np.ndarray,
    moving: np.ndarray,
    max_factor: int,
) -> Dict[int, PyramidLevel]:
    max_factor = max(1, int(max_factor))
    if max_factor & (max_factor - 1):
        raise ValueError("Pyramid downsampling factors must be powers of two.")
    reference_levels = cached_reference_pyramid_levels(reference, max_factor)
    levels = {1: PyramidLevel(1, reference, moving)}
    factor = 1
    ref_level = reference
    mov_level = moving
    while factor < max_factor:
        ref_level = (
            reference_levels[factor * 2]
            if reference_levels is not None
            else nanmean_downsample_2x(ref_level)
        )
        mov_level = nanmean_downsample_2x(mov_level)
        factor *= 2
        levels[factor] = PyramidLevel(factor, ref_level, mov_level)
    return levels


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
    finite_limit = min(int(np.count_nonzero(np.isfinite(reference))), int(np.count_nonzero(np.isfinite(moving))))
    return min(1000, max(30, int(finite_limit * 0.002)))


# Optional native preparation; NumPy retains all NCC reductions.
NATIVE_KERNELS_REQUESTED = os.environ.get("TIR_BRIDGE_NATIVE_KERNELS", "1").casefold() not in {"0", "off", "false", "no"}
NATIVE_KERNELS_ACTIVE = False
NATIVE_KERNELS_ERROR = ""
NATIVE_DISK_CACHE_REQUESTED = os.environ.get(
    "TIR_BRIDGE_NATIVE_DISK_CACHE", "1"
).strip().casefold() not in {"0", "off", "false", "no"}
NATIVE_DISK_CACHE_ENABLED = False
NATIVE_CACHE_DIRECTORY = ""
NATIVE_CACHE_ERROR = ""
_native_njit = None


def configure_native_disk_cache(config) -> None:
    """Keep compiled kernels outside the data workspace."""
    global NATIVE_DISK_CACHE_ENABLED, NATIVE_CACHE_DIRECTORY, NATIVE_CACHE_ERROR
    if not NATIVE_DISK_CACHE_REQUESTED:
        return
    try:
        override = os.environ.get("TIR_BRIDGE_NATIVE_CACHE_DIR") or os.environ.get("NUMBA_CACHE_DIR")
        if override:
            directory = Path(override).expanduser()
        elif os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
            directory = base / "TIR_ReCAST" / "numba"
        elif sys.platform == "darwin":
            directory = Path.home() / "Library" / "Caches" / "TIR_ReCAST" / "numba"
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
            directory = base / "TIR_ReCAST" / "numba"
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
        os.environ["NUMBA_CACHE_DIR"] = str(directory)
        config.CACHE_DIR = str(directory)
        # Newer Numba: prohibit fallback to an in-tree cache.
        if hasattr(config, "CACHE_LOCATOR_CLASSES"):
            config.CACHE_LOCATOR_CLASSES = "UserProvidedCacheLocator"
        NATIVE_CACHE_DIRECTORY = str(directory)
        NATIVE_DISK_CACHE_ENABLED = True
    except Exception as exc:
        NATIVE_CACHE_ERROR = str(exc)
        NATIVE_DISK_CACHE_ENABLED = False

if NATIVE_KERNELS_REQUESTED:
    try:
        from numba import njit as _native_njit, config as _native_config
        configure_native_disk_cache(_native_config)
    except Exception as _native_exc:
        NATIVE_KERNELS_ERROR = str(_native_exc)


def _gather_ncc_pairs_impl(reference, moving, ref_out, mov_out):
    count = 0
    for row in range(reference.shape[0]):
        for col in range(reference.shape[1]):
            ref = reference[row, col]
            mov = moving[row, col]
            if np.isfinite(ref) and np.isfinite(mov):
                ref_out[count] = ref
                mov_out[count] = mov
                count += 1
    return count


def _masked_box_sum_impl(data, valid, radius):
    height, width = data.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            value = np.float64(data[row, col]) if valid[row, col] else 0.0
            integral[row + 1, col + 1] = integral[row, col + 1] + value
    for row in range(1, height + 1):
        for col in range(1, width + 1):
            integral[row, col] = integral[row, col - 1] + integral[row, col]
    out = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        top = max(0, row - radius)
        bottom = min(height, row + radius + 1)
        for col in range(width):
            left = max(0, col - radius)
            right = min(width, col + radius + 1)
            value = integral[bottom, right] - integral[top, right]
            value = value - integral[bottom, left]
            out[row, col] = value + integral[top, left]
    return out


def _boolean_box_count_impl(valid, radius):
    height, width = valid.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    for row in range(height):
        for col in range(width):
            integral[row + 1, col + 1] = integral[row, col + 1] + int(valid[row, col])
    for row in range(1, height + 1):
        for col in range(1, width + 1):
            integral[row, col] = integral[row, col - 1] + integral[row, col]
    out = np.empty((height, width), dtype=np.int32)
    for row in range(height):
        top = max(0, row - radius)
        bottom = min(height, row + radius + 1)
        for col in range(width):
            left = max(0, col - radius)
            right = min(width, col + radius + 1)
            value = integral[bottom, right] - integral[top, right]
            value = value - integral[bottom, left]
            out[row, col] = value + integral[top, left]
    return out


def _native_warp_impl(source, a, b, c, d, e, f):
    height, width = source.shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(height):
        for col in range(width):
            sx = np.rint((a * np.float64(col) + b * np.float64(row)) + c)
            sy = np.rint((d * np.float64(col) + e * np.float64(row)) + f)
            if 0.0 <= sx < width and 0.0 <= sy < height:
                out[row, col] = source[int(sy), int(sx)]
    return out


def create_native_dispatchers(cache: bool) -> None:
    global _gather_ncc_pairs_native, _masked_box_sum_native
    global _boolean_box_count_native, _native_warp
    _gather_ncc_pairs_native = _native_njit(cache=cache, nogil=True, fastmath=False)(_gather_ncc_pairs_impl)
    _masked_box_sum_native = _native_njit(cache=cache, nogil=True, fastmath=False)(_masked_box_sum_impl)
    _boolean_box_count_native = _native_njit(cache=cache, nogil=True, fastmath=False)(_boolean_box_count_impl)
    _native_warp = _native_njit(cache=cache, nogil=True, fastmath=False)(_native_warp_impl)


if _native_njit is not None:
    try:
        try:
            create_native_dispatchers(NATIVE_DISK_CACHE_ENABLED)
        except Exception as _cache_exc:
            if not NATIVE_DISK_CACHE_ENABLED:
                raise
            NATIVE_CACHE_ERROR = str(_cache_exc)
            NATIVE_DISK_CACHE_ENABLED = False
            create_native_dispatchers(False)
        NATIVE_KERNELS_ACTIVE = True
    except Exception as _native_exc:
        NATIVE_KERNELS_ERROR = str(_native_exc)


def initialize_native_kernels() -> None:
    global NATIVE_KERNELS_ACTIVE, NATIVE_KERNELS_ERROR
    if not NATIVE_KERNELS_ACTIVE:
        return
    try:
        sample = np.arange(80, dtype=np.float32).reshape(8, 10)
        sample[2, 3] = np.nan
        for source in (sample, sample[:, ::-1]):
            valid = np.isfinite(source)
            outputs = (np.empty(source.size, dtype=np.float64), np.empty(source.size, dtype=np.float64))
            count = _gather_ncc_pairs_native(source, source, *outputs)
            expected = source[valid].astype(np.float64)
            if count != expected.size or not np.array_equal(outputs[0][:count].view(np.uint64), expected.view(np.uint64)):
                raise RuntimeError("NCC sample-order self-check failed")
            sums = _masked_box_sum_native(source, valid, 3)
            counts = _boolean_box_count_native(valid, 3)
            _native_warp(source, 1.0, 0.001, 0.5, 0.002, 1.0, -0.5)
            # Compare with the unchanged NumPy implementation.
            NATIVE_KERNELS_ACTIVE = False
            try:
                expected_sums = box_sum_same(source, 3, valid=valid)
                expected_counts = box_count_same(valid, 3)
            finally:
                NATIVE_KERNELS_ACTIVE = True
            if not np.array_equal(sums.view(np.uint64), expected_sums.view(np.uint64)) or not np.array_equal(counts, expected_counts):
                raise RuntimeError("box-sum self-check failed")
        # Mixed contiguous/strided NCC layouts, before scene threads start.
        strided = sample[:, ::-1]
        for reference, moving in ((sample, strided), (strided, sample)):
            valid = np.isfinite(reference) & np.isfinite(moving)
            count = _gather_ncc_pairs_native(reference, moving, *outputs)
            expected_ref = reference[valid].astype(np.float64)
            expected_mov = moving[valid].astype(np.float64)
            if (count != expected_ref.size
                    or not np.array_equal(outputs[0][:count].view(np.uint64), expected_ref.view(np.uint64))
                    or not np.array_equal(outputs[1][:count].view(np.uint64), expected_mov.view(np.uint64))):
                raise RuntimeError("Mixed-layout NCC self-check failed")
    except Exception as exc:
        NATIVE_KERNELS_ACTIVE = False
        NATIVE_KERNELS_ERROR = str(exc)
        print(f"Native preparation unavailable; retaining NumPy: {exc}", file=sys.stderr, flush=True)
    initialize_exact_cpu_kernels()


# Exact elementwise preparation; no reassociation or reduced precision.
EXACT_CPU_KERNELS_REQUESTED = os.environ.get(
    "TIR_BRIDGE_EXACT_CPU_KERNELS", "1"
).casefold() not in {"0", "off", "false", "no"}
EXACT_CPU_KERNELS_ACTIVE = False
_EXACT_CPU_KERNELS_READY = False


def _exact_gradient_impl(data, zero_is_nan):
    height, width = data.shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            c = data[row, col]
            n, south = data[row - 1, col], data[row + 1, col]
            w, east = data[row, col - 1], data[row, col + 1]
            if (np.isfinite(c) and np.isfinite(n) and np.isfinite(south)
                    and np.isfinite(w) and np.isfinite(east)):
                gx = np.float32(np.float32(east - w) * np.float32(0.5))
                gy = np.float32(np.float32(south - n) * np.float32(0.5))
                gx = np.float32(gx * gx)
                gy = np.float32(gy * gy)
                value = np.sqrt(np.float32(gx + gy))
                if not zero_is_nan or (np.isfinite(value) and value != 0.0):
                    out[row, col] = value
    return out


def _exact_downsample_impl(data):
    height, width = data.shape
    out = np.full(((height + 1) // 2, (width + 1) // 2), np.nan, dtype=np.float32)
    for row in range(out.shape[0]):
        y = 2 * row
        for col in range(out.shape[1]):
            x = 2 * col
            a, b, c, d = 0.0, 0.0, 0.0, 0.0
            count = 0
            v = data[y, x]
            if np.isfinite(v):
                a = np.float64(v)
                count += 1
            if x + 1 < width:
                v = data[y, x + 1]
                if np.isfinite(v):
                    b = np.float64(v)
                    count += 1
            if y + 1 < height:
                v = data[y + 1, x]
                if np.isfinite(v):
                    c = np.float64(v)
                    count += 1
                if x + 1 < width:
                    v = data[y + 1, x + 1]
                    if np.isfinite(v):
                        d = np.float64(v)
                        count += 1
            if count:
                out[row, col] = ((a + b) + (c + d) + 0.0) / count
    return out


def _exact_smooth_impl(data, radius, minimum_fraction):
    height, width = data.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    counts = np.zeros((height + 1, width + 1), dtype=np.int32)
    for row in range(height):
        for col in range(width):
            value = data[row, col]
            finite = np.isfinite(value)
            integral[row + 1, col + 1] = integral[row, col + 1] + (np.float64(value) if finite else 0.0)
            counts[row + 1, col + 1] = counts[row, col + 1] + int(finite)
    for row in range(1, height + 1):
        for col in range(1, width + 1):
            integral[row, col] = integral[row, col - 1] + integral[row, col]
            counts[row, col] = counts[row, col - 1] + counts[row, col]
    out = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(height):
        top = max(0, row - radius)
        bottom = min(height, row + radius + 1)
        for col in range(width):
            if not np.isfinite(data[row, col]):
                continue
            left = max(0, col - radius)
            right = min(width, col + radius + 1)
            n = (counts[bottom, right] - counts[top, right]) - counts[bottom, left] + counts[top, left]
            possible = (bottom - top) * (right - left)
            if n >= max(9.0, minimum_fraction * possible):
                total = integral[bottom, right] - integral[top, right]
                total = total - integral[bottom, left]
                total = total + integral[top, left]
                out[row, col] = total / n
    return out


def _exact_warp_pair_impl(first, second, a, b, c, d, e, f):
    height, width = first.shape
    left = np.full((height, width), np.nan, dtype=np.float32)
    right = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(height):
        for col in range(width):
            sx = np.rint((a * np.float64(col) + b * np.float64(row)) + c)
            sy = np.rint((d * np.float64(col) + e * np.float64(row)) + f)
            if 0.0 <= sx < width and 0.0 <= sy < height:
                left[row, col] = first[int(sy), int(sx)]
                right[row, col] = second[int(sy), int(sx)]
    return left, right


def _exact_support_counts_impl(reference_valid, overlap):
    span_pixels = 0
    overlap_pixels = 0
    height, width = overlap.shape
    for row in range(height):
        left = 0
        while left < width and not overlap[row, left]:
            left += 1
        if left == width:
            continue
        right = width - 1
        while right > left and not overlap[row, right]:
            right -= 1
        for col in range(left, right + 1):
            span_pixels += int(reference_valid[row, col])
            overlap_pixels += int(overlap[row, col])
    return span_pixels, overlap_pixels


def _create_exact_cpu_dispatchers(cache):
    global _exact_gradient, _exact_downsample, _exact_smooth
    global _exact_warp_pair, _exact_support_counts
    def compile_function(function):
        return _native_njit(cache=cache, nogil=True, fastmath=False)(function)
    _exact_gradient = compile_function(_exact_gradient_impl)
    _exact_downsample = compile_function(_exact_downsample_impl)
    _exact_smooth = compile_function(_exact_smooth_impl)
    _exact_warp_pair = compile_function(_exact_warp_pair_impl)
    _exact_support_counts = compile_function(_exact_support_counts_impl)


def _same_float32_bits(left, right):
    return np.array_equal(left.view(np.uint32), right.view(np.uint32))


def initialize_exact_cpu_kernels():
    global EXACT_CPU_KERNELS_ACTIVE, _EXACT_CPU_KERNELS_READY
    if _EXACT_CPU_KERNELS_READY or not EXACT_CPU_KERNELS_REQUESTED or not NATIVE_KERNELS_ACTIVE:
        return
    _EXACT_CPU_KERNELS_READY = True
    try:
        try:
            _create_exact_cpu_dispatchers(NATIVE_DISK_CACHE_ENABLED)
        except Exception:
            _create_exact_cpu_dispatchers(False)
        rng = np.random.default_rng(8197)
        sample = rng.standard_normal((37, 39)).astype(np.float32)
        sample[::7, ::5] = np.nan
        sample[2, 3] = -0.0
        sample[7, 9] = np.float32(1e-40)
        for data in (sample, sample[:, ::-1]):
            for clean, function in ((True, _gradient_magnitude_cpu), (False, qa_gradient_magnitude)):
                if not _same_float32_bits(_exact_gradient(data, clean), function(data)):
                    raise RuntimeError("gradient rounding self-check failed")
            for fraction, function in ((0.20, nan_box_smooth), (0.35, qa_nan_box_smooth)):
                if not _same_float32_bits(_exact_smooth(data, 3, fraction), function(data, 3)):
                    raise RuntimeError("smoothing self-check failed")
            transform = Affine(1.0, 0.001, 0.5, 0.002, 1.0, -0.5)
            inverse = ~transform
            actual = _exact_warp_pair(data, data, *tuple(inverse)[:6])
            expected = _warp_array_nearest_pair_cpu(data, data, transform, 8)
            if not all(_same_float32_bits(a, b) for a, b in zip(actual, expected)):
                raise RuntimeError("paired-warp rounding self-check failed")
            valid = np.isfinite(data)
            for mask in (valid, valid[:, ::-1]):
                span, count = _exact_support_counts(valid, mask)
                actual_loss = 1.0 if span == 0 else max(0.0, min(1.0, 1.0 - count / span))
                expected_loss = support_loss_fraction(data, data, valid, valid, mask)
                if actual_loss != expected_loss:
                    raise RuntimeError("support-count self-check failed")
        for data in (sample, sample[:-1, :-1].copy()):
            if not _same_float32_bits(_exact_downsample(data), nanmean_downsample_2x(data)):
                raise RuntimeError("downsampling rounding self-check failed")
        EXACT_CPU_KERNELS_ACTIVE = True
    except Exception as exc:
        EXACT_CPU_KERNELS_ACTIVE = False
        print(f"Extra CPU preparation unavailable; retaining previous kernels: {exc}", file=sys.stderr, flush=True)


def _ncc_work_arrays(size: int):
    arrays = getattr(CPU_THREAD_STATE, "ncc_work_arrays", None)
    if arrays is None or arrays[0].size < size:
        arrays = (np.empty(size, dtype=np.float64), np.empty(size, dtype=np.float64))
        # Bound retained scratch to 64 MiB per thread.
        if size * 16 <= 64 * 1024 ** 2:
            CPU_THREAD_STATE.ncc_work_arrays = arrays
    return arrays


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
    native = (
        NATIVE_KERNELS_ACTIVE and ref_view.size >= 1024
        and ref_view.dtype == np.dtype(np.float32)
        and mov_view.dtype == np.dtype(np.float32)
        and ref_view.flags.writeable and mov_view.flags.writeable
    )
    if native:
        ref_buffer, mov_buffer = _ncc_work_arrays(ref_view.size)
        valid_count = int(_gather_ncc_pairs_native(ref_view, mov_view, ref_buffer, mov_buffer))
        if valid_count < min_valid:
            return -math.inf, valid_count
        ref_values, mov_values = ref_buffer[:valid_count], mov_buffer[:valid_count]
    else:
        valid = np.isfinite(ref_view) & np.isfinite(mov_view)
        valid_count = int(np.count_nonzero(valid))
        if valid_count < min_valid:
            return -math.inf, valid_count

    if native:
        pass
    elif valid_count == valid.size:
        # Boolean selection would only flatten every pixel into a temporary.
        # Keep the same row-major sequence and private writable float64 inputs.
        ref_values = np.array(ref_view, dtype=np.float64, order="C", copy=True).reshape(-1)
        mov_values = np.array(mov_view, dtype=np.float64, order="C", copy=True).reshape(-1)
    else:
        ref_values = ref_view[valid].astype(np.float64)
        mov_values = mov_view[valid].astype(np.float64)
    ref_values -= ref_values.mean()
    mov_values -= mov_values.mean()
    denom = math.sqrt(float(np.dot(ref_values, ref_values) * np.dot(mov_values, mov_values)))
    if denom <= 0.0 or not math.isfinite(denom):
        return -math.inf, valid_count
    score = float(np.dot(ref_values, mov_values) / denom)
    return score, valid_count


@lru_cache(maxsize=1)
def _gpu_float32_to_float64_kernel():
    """Convert on device without CuPy's default flush-to-zero behavior."""
    assert cp is not None
    return cp.ElementwiseKernel(
        "float32 value",
        "float64 result",
        'asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(value));',
        "tir_exact_float32_to_float64",
    )


def _gpu_asarray_float64(data: np.ndarray):
    """Upload native float32 rasters at half the float64 transfer size.

    The explicit non-FTZ conversion preserves even float32 subnormals. These
    callers mask NaNs before any matching arithmetic, so NaN payload conversion
    has no effect on their results. Keep every other input dtype on its existing
    CuPy conversion path.
    """
    assert cp is not None
    if isinstance(data, np.ndarray) and data.dtype == np.dtype(np.float32):
        return _gpu_float32_to_float64_kernel()(cp.asarray(data))
    return cp.asarray(data, dtype=cp.float64)


@lru_cache(maxsize=1)
def _gpu_ncc_masked_products_kernel():
    """Build the original NCC reduction inputs with one elementwise launch."""
    assert cp is not None
    return cp.ElementwiseKernel(
        "float64 reference, float64 moving",
        "bool valid, float64 ref_values, float64 mov_values, "
        "float64 products, float64 ref_squared, float64 mov_squared",
        """
        valid = isfinite(reference) && isfinite(moving);
        ref_values = valid ? reference : 0.0;
        mov_values = valid ? moving : 0.0;
        products = __dmul_rn(ref_values, mov_values);
        ref_squared = __dmul_rn(ref_values, ref_values);
        mov_squared = __dmul_rn(mov_values, mov_values);
        """,
        "tir_ncc_masked_products",
        options=("--fmad=false",),
    )


@lru_cache(maxsize=1)
def _gpu_ncc_score_kernel():
    """Preserve separately rounded float64 arithmetic in the scalar NCC score."""
    assert cp is not None
    return cp.ElementwiseKernel(
        "int64 count, float64 sum_ref, float64 sum_mov, float64 sum_products, "
        "float64 sum_ref_squared, float64 sum_mov_squared, int64 min_valid",
        "float64 score",
        """
        const double count_safe = fmax((double)count, 1.0);
        const double covariance = __dsub_rn(sum_products,
            __ddiv_rn(__dmul_rn(sum_ref, sum_mov), count_safe));
        const double variance_ref = __dsub_rn(sum_ref_squared,
            __ddiv_rn(__dmul_rn(sum_ref, sum_ref), count_safe));
        const double variance_mov = __dsub_rn(sum_mov_squared,
            __ddiv_rn(__dmul_rn(sum_mov, sum_mov), count_safe));
        const double positive_ref = isnan(variance_ref)
            ? variance_ref : fmax(variance_ref, 0.0);
        const double positive_mov = isnan(variance_mov)
            ? variance_mov : fmax(variance_mov, 0.0);
        const double denominator = __dsqrt_rn(__dmul_rn(positive_ref, positive_mov));
        score = (count >= min_valid && denominator > 0.0)
            ? __ddiv_rn(covariance, denominator)
            : -__longlong_as_double(0x7ff0000000000000LL);
        """,
        "tir_ncc_score_from_sums",
        options=("--fmad=false",),
    )


@lru_cache(maxsize=1)
def _gpu_ncc_native32_products_kernel():
    """Retain float32 source storage; preserve all float64 NCC arithmetic."""
    assert cp is not None
    return cp.ElementwiseKernel(
        "float32 reference_source, float32 moving_source",
        "bool valid, float64 ref_values, float64 mov_values, "
        "float64 products, float64 ref_squared, float64 mov_squared",
        r"""
        double reference, moving;
        asm("cvt.f64.f32 %0, %1;" : "=d"(reference) : "f"(reference_source));
        asm("cvt.f64.f32 %0, %1;" : "=d"(moving) : "f"(moving_source));
        valid = isfinite(reference) && isfinite(moving);
        ref_values = valid ? reference : 0.0;
        mov_values = valid ? moving : 0.0;
        products = __dmul_rn(ref_values, mov_values);
        ref_squared = __dmul_rn(ref_values, ref_values);
        mov_squared = __dmul_rn(mov_values, mov_values);
        """,
        "tir_ncc_native32_masked_products",
        options=("--fmad=false",),
    )


_GPU_NCC_NATIVE32_VERIFIED = None
_GPU_NCC_NATIVE32_CHECK_LOCK = threading.Lock()


def gpu_ncc_native32_verified() -> bool:
    global _GPU_NCC_NATIVE32_VERIFIED
    if _GPU_NCC_NATIVE32_VERIFIED is not None:
        return bool(_GPU_NCC_NATIVE32_VERIFIED)
    with _GPU_NCC_NATIVE32_CHECK_LOCK:
        if _GPU_NCC_NATIVE32_VERIFIED is None:
            try:
                bits = np.asarray([
                    0x00000000, 0x80000000, 0x00000001, 0x80000001,
                    0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
                    0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
                    0x7f800000, 0xff800000, 0x7fc00000, 0x3eaaaaab,
                ], dtype=np.uint32).view(np.float32).reshape(4, 4)
                device = cp.asarray(bits)
                comparisons = []
                for moving in (device, device[::-1, ::-1]):
                    expected = _gpu_ncc_masked_products_kernel()(
                        _gpu_float32_to_float64_kernel()(device),
                        _gpu_float32_to_float64_kernel()(moving),
                    )
                    actual = _gpu_ncc_native32_products_kernel()(device, moving)
                    for left, right in zip(expected, actual):
                        if left.dtype == cp.float64:
                            left, right = left.view(cp.uint64), right.view(cp.uint64)
                        comparisons.append(cp.all(left == right))
                _GPU_NCC_NATIVE32_VERIFIED = bool(cp.all(cp.stack(comparisons)).item())
            except Exception:
                _GPU_NCC_NATIVE32_VERIFIED = False
            if not _GPU_NCC_NATIVE32_VERIFIED:
                print(
                    "GPU NCC storage self-check: retaining the original float64 input path.",
                    file=sys.stderr, flush=True,
                )
    return bool(_GPU_NCC_NATIVE32_VERIFIED)


def _gpu_ncc_scores_original_job(
    reference: np.ndarray,
    moving: np.ndarray,
    shifts: Sequence[Tuple[int, int]],
    min_valid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        # The fused map and uploaded inputs need at most 57 bytes/pixel;
        # retain the original conservative check, including reduction scratch.
        require_gpu_memory(int(reference.size) * 88)
        native32 = (
            GPU_NCC_NATIVE_FLOAT32
            and isinstance(reference, np.ndarray) and isinstance(moving, np.ndarray)
            and reference.dtype == np.dtype(np.float32)
            and moving.dtype == np.dtype(np.float32)
            and reference.flags.c_contiguous and moving.flags.c_contiguous
            and gpu_ncc_native32_verified()
        )
        if native32:
            reference_gpu = cp.asarray(reference)
            moving_gpu = cp.asarray(moving)
            products_kernel = _gpu_ncc_native32_products_kernel()
        else:
            reference_gpu = _gpu_asarray_float64(reference)
            moving_gpu = _gpu_asarray_float64(moving)
            products_kernel = _gpu_ncc_masked_products_kernel()
        score_scalars = []
        count_scalars = []
        for row_shift, col_shift in shifts:
            slices = shifted_slices(reference.shape, int(row_shift), int(col_shift))
            if slices is None:
                score_scalars.append(cp.asarray(-cp.inf, dtype=cp.float64))
                count_scalars.append(cp.asarray(0, dtype=cp.int64))
                continue
            ref_slice, mov_slice = slices
            ref_view = reference_gpu[ref_slice]
            mov_view = moving_gpu[mov_slice]
            (
                valid, ref_values, mov_values, products,
                ref_squared, mov_squared,
            ) = products_kernel(ref_view, mov_view)
            # Keep the same six CuPy reductions and their numerical order.
            # Release each full-sized input before allocating the next shift.
            count = cp.count_nonzero(valid).astype(cp.int64, copy=False)
            del valid
            sum_ref = cp.sum(ref_values, dtype=cp.float64)
            del ref_values
            sum_mov = cp.sum(mov_values, dtype=cp.float64)
            del mov_values
            sum_products = cp.sum(products, dtype=cp.float64)
            del products
            sum_ref_squared = cp.sum(ref_squared, dtype=cp.float64)
            del ref_squared
            sum_mov_squared = cp.sum(mov_squared, dtype=cp.float64)
            del mov_squared
            score = _gpu_ncc_score_kernel()(
                count, sum_ref, sum_mov, sum_products,
                sum_ref_squared, sum_mov_squared, np.int64(min_valid),
            )
            score_scalars.append(score)
            count_scalars.append(count)
        scores_host, counts_host = gpu_download_arrays(
            cp.stack(score_scalars), cp.stack(count_scalars)
        )
        return scores_host.astype(np.float64, copy=False), counts_host.astype(np.int64, copy=False)


def _gpu_ncc_scores_preallocated_job(reference, moving, shifts, min_valid):
    """Reuse map buffers; retain each original floating-point reduction."""
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(int(reference.size) * 88)
        native32 = (
            GPU_NCC_NATIVE_FLOAT32
            and isinstance(reference, np.ndarray) and isinstance(moving, np.ndarray)
            and reference.dtype == np.dtype(np.float32) and moving.dtype == np.dtype(np.float32)
            and reference.flags.c_contiguous and moving.flags.c_contiguous
            and gpu_ncc_native32_verified()
        )
        if native32:
            reference_gpu, moving_gpu = cp.asarray(reference), cp.asarray(moving)
            products_kernel = _gpu_ncc_native32_products_kernel()
        else:
            reference_gpu, moving_gpu = _gpu_asarray_float64(reference), _gpu_asarray_float64(moving)
            products_kernel = _gpu_ncc_masked_products_kernel()
        size = int(reference.size)
        valid_storage = cp.empty(size, dtype=cp.bool_)
        values_storage = tuple(cp.empty(size, dtype=cp.float64) for _ in range(5))
        sums = cp.zeros((5, len(shifts)), dtype=cp.float64)
        counts = cp.zeros(len(shifts), dtype=cp.int64)
        for index, (row_shift, col_shift) in enumerate(shifts):
            selection = shifted_slices(reference.shape, int(row_shift), int(col_shift))
            if selection is None:
                continue
            ref_view, mov_view = reference_gpu[selection[0]], moving_gpu[selection[1]]
            shape, count = ref_view.shape, int(ref_view.size)
            valid = valid_storage[:count].reshape(shape)
            values = tuple(storage[:count].reshape(shape) for storage in values_storage)
            products_kernel(ref_view, mov_view, valid, *values)
            # Boolean integer sums are exact; float sums keep the same shape/order.
            cp.sum(valid, dtype=cp.int64, out=counts[index:index + 1].reshape(()))
            for statistic, value in enumerate(values):
                cp.sum(value, dtype=cp.float64, out=sums[statistic, index:index + 1].reshape(()))
        # Independent scalar formulas become a single elementwise launch.
        scores = _gpu_ncc_score_kernel()(counts, *sums, np.int64(min_valid))
        scores_host, counts_host = gpu_download_arrays(scores, counts)
        return scores_host.astype(np.float64, copy=False), counts_host.astype(np.int64, copy=False)


_GPU_NCC_PREALLOCATED_CHECKED = set()
_GPU_NCC_PREALLOCATED_LOCK = threading.Lock()
_GPU_NCC_PREALLOCATED_DISABLED = os.environ.get("TIR_BRIDGE_GPU_NCC_SCRATCH", "1").casefold() in {"0", "off", "false", "no"}


def _gpu_ncc_scores_job(reference, moving, shifts, min_valid):
    global _GPU_NCC_PREALLOCATED_DISABLED
    if _GPU_NCC_PREALLOCATED_DISABLED or len(shifts) == 0:
        return _gpu_ncc_scores_original_job(reference, moving, shifts, min_valid)
    try:
        # Verify the actual device path before using it for each source layout.
        signature = (reference.dtype.str, moving.dtype.str,
                     bool(reference.flags.c_contiguous), bool(moving.flags.c_contiguous))
        if signature not in _GPU_NCC_PREALLOCATED_CHECKED:
            with _GPU_NCC_PREALLOCATED_LOCK:
                if signature not in _GPU_NCC_PREALLOCATED_CHECKED:
                    actual = _gpu_ncc_scores_preallocated_job(reference, moving, shifts, min_valid)
                    expected = _gpu_ncc_scores_original_job(reference, moving, shifts, min_valid)
                    same = (np.array_equal(actual[0].view(np.uint64), expected[0].view(np.uint64))
                            and np.array_equal(actual[1], expected[1]))
                    if not same:
                        _GPU_NCC_PREALLOCATED_DISABLED = True
                        print("GPU NCC scratch self-check: retaining original reductions.", file=sys.stderr, flush=True)
                    else:
                        _GPU_NCC_PREALLOCATED_CHECKED.add(signature)
                    return expected
        return _gpu_ncc_scores_preallocated_job(reference, moving, shifts, min_valid)
    except GPUInsufficientMemory:
        raise
    except Exception:
        _GPU_NCC_PREALLOCATED_DISABLED = True
        return _gpu_ncc_scores_original_job(reference, moving, shifts, min_valid)


def cpu_ncc_support_crop(
    reference: np.ndarray,
    moving: np.ndarray,
    shifts: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Discard outer pixels that cannot occur in any requested valid pair."""
    if (
        len(shifts) < 5
        or reference.size < 256 * 256
        or reference.shape != moving.shape
    ):
        return reference, moving
    valid = np.isfinite(moving)
    rows = np.flatnonzero(valid.any(axis=1))
    if rows.size == 0:
        return reference, moving
    cols = np.flatnonzero(valid.any(axis=0))
    min_row = min(0, min(int(row) for row, _col in shifts))
    max_row = max(0, max(int(row) for row, _col in shifts))
    min_col = min(0, min(int(col) for _row, col in shifts))
    max_col = max(0, max(int(col) for _row, col in shifts))
    # Retain every finite moving coordinate AND every reference coordinate it
    # can meet under these shifts. Removing shared origins preserves each lag.
    y0 = max(0, int(rows[0]) + min_row)
    y1 = min(reference.shape[0], int(rows[-1]) + 1 + max_row)
    x0 = max(0, int(cols[0]) + min_col)
    x1 = min(reference.shape[1], int(cols[-1]) + 1 + max_col)
    if (y1 - y0) * (x1 - x0) > 0.85 * reference.size:
        return reference, moving
    selection = (slice(y0, y1), slice(x0, x1))
    return reference[selection], moving[selection]


def normalized_correlations_at_shifts(
    reference: np.ndarray,
    moving: np.ndarray,
    shifts: Sequence[Tuple[int, int]],
    min_valid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    use_gpu = (
        gpu_is_active()
        and len(shifts) >= 5
        and int(reference.size) * len(shifts) >= GPU_NCC_WORK_THRESHOLD
    )
    if use_gpu:
        try:
            return submit_gpu_job(
                _gpu_ncc_scores_job,
                reference,
                moving,
                tuple(shifts),
                min_valid,
            )
        except Exception as exc:
            record_gpu_fallback(exc)
    # Make the backend decision on the original inputs. On CPU, these crops
    # preserve the exact row-major valid-pair sequence and caller's threshold.
    reference, moving = cpu_ncc_support_crop(reference, moving, shifts)
    scores = np.empty(len(shifts), dtype=np.float64)
    counts = np.empty(len(shifts), dtype=np.int64)
    for index, (row_shift, col_shift) in enumerate(shifts):
        score, count = normalized_correlation(
            reference,
            moving,
            int(row_shift),
            int(col_shift),
            min_valid,
        )
        scores[index] = score
        counts[index] = count
    return scores, counts


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


def search_level_match(
    level: PyramidLevel,
    center_row: int,
    center_col: int,
    radius: int,
    max_shift_pixels: int,
) -> PeakDiagnostics:
    radius = max(0, int(radius))
    level_limit = int(math.floor(max_shift_pixels / max(1, level.factor)))
    row_lags = np.arange(
        max(-level_limit, int(center_row) - radius),
        min(level_limit, int(center_row) + radius) + 1,
        dtype=np.int64,
    )
    col_lags = np.arange(
        max(-level_limit, int(center_col) - radius),
        min(level_limit, int(center_col) + radius) + 1,
        dtype=np.int64,
    )
    if row_lags.size == 0 or col_lags.size == 0:
        raise RuntimeError(
            f"No valid correlation candidates at pyramid factor {level.factor}."
        )
    shifts = [
        (int(row_shift), int(col_shift))
        for row_shift in row_lags
        for col_shift in col_lags
    ]
    support_cap = min(
        int(np.count_nonzero(np.isfinite(level.reference))),
        int(np.count_nonzero(np.isfinite(level.moving))),
    )
    min_valid = min(1000, max(30, int(support_cap * 0.002)))
    scores, counts = normalized_correlations_at_shifts(
        level.reference,
        level.moving,
        shifts,
        min_valid,
    )
    best: Optional[ShiftResult] = None
    for (row_shift, col_shift), score, count in zip(shifts, scores, counts):
        score_value = float(score)
        count_value = int(count)
        if not math.isfinite(score_value):
            continue
        if better_candidate(
            score_value,
            count_value,
            row_shift,
            col_shift,
            best,
        ):
            best = ShiftResult(
                float(row_shift),
                float(col_shift),
                score_value,
                count_value,
            )
    if best is None:
        raise RuntimeError(
            f"No valid correlation candidate at pyramid factor {level.factor}. "
            "Check that the two rasters overlap and have enough non-nodata pixels."
        )
    score_surface = np.asarray(scores, dtype=np.float64).reshape(
        row_lags.size,
        col_lags.size,
    )
    count_surface = np.asarray(counts, dtype=np.int64).reshape(
        row_lags.size,
        col_lags.size,
    )
    match = peak_diagnostics_from_windows(
        score_surface,
        count_surface,
        row_lags,
        col_lags,
        min_valid,
        support_cap,
        (
            f"No valid correlation candidate at pyramid factor {level.factor}. "
            "Check that the two rasters overlap and have enough non-nodata pixels."
        ),
        selected_result=best,
    )
    match.ncc_scores = {
        shift: (float(score), int(count))
        for shift, score, count in zip(shifts, scores, counts)
    }
    return match


def search_level(
    level: PyramidLevel,
    center_row: int,
    center_col: int,
    radius: int,
    max_shift_pixels: int,
) -> ShiftResult:
    return search_level_match(
        level,
        center_row,
        center_col,
        radius,
        max_shift_pixels,
    ).result


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def valid_search_crop(reference: np.ndarray, moving: np.ndarray, pad: int):
    valid = np.isfinite(reference) | np.isfinite(moving)
    valid_rows = np.flatnonzero(valid.any(axis=1))
    valid_cols = np.flatnonzero(valid.any(axis=0))
    if valid_rows.size == 0 or valid_cols.size == 0:
        raise RuntimeError("No valid pixels were found for phase correlation.")
    y0 = max(0, int(valid_rows[0]) - pad)
    y1 = min(reference.shape[0], int(valid_rows[-1]) + pad + 1)
    x0 = max(0, int(valid_cols[0]) - pad)
    x1 = min(reference.shape[1], int(valid_cols[-1]) + pad + 1)
    return reference[y0:y1, x0:x1], moving[y0:y1, x0:x1]


@lru_cache(maxsize=64)
def hanning_float32(length: int) -> np.ndarray:
    window = np.hanning(max(0, int(length))).astype(np.float32)
    window.setflags(write=False)
    return window


# Reference-only caches; keys retain exact view geometry and FFT settings.
_REFERENCE_PHASE_LIMIT = max(0, int(os.environ.get("TIR_BRIDGE_REFERENCE_PHASE_MB", "64"))) * 1024 ** 2
_REFERENCE_PHASE_FFT_LIMIT = max(0, int(os.environ.get("TIR_BRIDGE_REFERENCE_PHASE_FFT_MB", "128"))) * 1024 ** 2
_REFERENCE_PHASE_CACHE = OrderedDict()
_REFERENCE_PHASE_BYTES = 0
_REFERENCE_PHASE_LOCK = threading.Lock()
_REFERENCE_PHASE_FFT_CACHE = None


def _array_root(array):
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def _owned_reference_root(data):
    root = _array_root(data)
    with REFERENCE_GRID_CACHE_LOCK:
        for products in REFERENCE_GRID_CACHE.values():
            for source in (products.data, products.gradient, products.robust_gradient):
                if _array_root(source) is root:
                    return root
            with products.reference_pyramid_lock:
                for levels in products.reference_pyramids.values():
                    for source in levels.values():
                        if _array_root(source) is root:
                            return root
    return None


def cached_standardize_reference(data):
    global _REFERENCE_PHASE_BYTES
    root = _owned_reference_root(data) if _REFERENCE_PHASE_LIMIT else None
    if root is None:
        return standardize_for_phase(data)
    key = (id(root), int(data.ctypes.data), data.shape, data.strides, data.dtype.str)
    with _REFERENCE_PHASE_LOCK:
        cached = _REFERENCE_PHASE_CACHE.get(key)
        if cached is not None and cached[0]() is root:
            _REFERENCE_PHASE_CACHE.move_to_end(key)
            return cached[1], cached[2]
    phase, valid = standardize_for_phase(data)
    size = int(phase.nbytes + valid.nbytes)
    if size > _REFERENCE_PHASE_LIMIT:
        return phase, valid
    phase.setflags(write=False)
    valid.setflags(write=False)
    with _REFERENCE_PHASE_LOCK:
        cached = _REFERENCE_PHASE_CACHE.get(key)
        if cached is not None and cached[0]() is root:
            _REFERENCE_PHASE_CACHE.move_to_end(key)
            return cached[1], cached[2]
        if cached is not None:
            _REFERENCE_PHASE_BYTES -= cached[3]
            del _REFERENCE_PHASE_CACHE[key]
        while _REFERENCE_PHASE_CACHE and _REFERENCE_PHASE_BYTES + size > _REFERENCE_PHASE_LIMIT:
            _, removed = _REFERENCE_PHASE_CACHE.popitem(last=False)
            _REFERENCE_PHASE_BYTES -= removed[3]
        _REFERENCE_PHASE_CACHE[key] = (weakref.ref(root), phase, valid, size)
        _REFERENCE_PHASE_BYTES += size
    return phase, valid


def reference_phase_fft_cache():
    global _REFERENCE_PHASE_FFT_CACHE
    if not _REFERENCE_PHASE_FFT_LIMIT:
        return None
    with _REFERENCE_PHASE_LOCK:
        if _REFERENCE_PHASE_FFT_CACHE is None:
            available = available_system_memory_bytes()
            budget = min(_REFERENCE_PHASE_FFT_LIMIT, max(0, available) // 32)
            _REFERENCE_PHASE_FFT_CACHE = CPUReferenceFFTCache(budget)
        return _REFERENCE_PHASE_FFT_CACHE


def clear_reference_phase_cache():
    global _REFERENCE_PHASE_BYTES
    with _REFERENCE_PHASE_LOCK:
        _REFERENCE_PHASE_CACHE.clear()
        _REFERENCE_PHASE_BYTES = 0
        if _REFERENCE_PHASE_FFT_CACHE is not None:
            _REFERENCE_PHASE_FFT_CACHE.clear()


def standardize_for_phase(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(data)
    if int(np.count_nonzero(valid)) < 30:
        raise RuntimeError("Not enough valid pixels for phase correlation.")
    raw_values = data[valid]
    values = raw_values.astype(np.float64)
    mean = float(values.mean())
    try:
        # NumPy 2 can reuse the identical mean instead of reducing it again.
        std = float(values.std(mean=mean))
    except TypeError:
        std = float(values.std())
    if not math.isfinite(std) or std <= 0.0:
        std = 1.0
    out = np.zeros(data.shape, dtype=np.float32)
    if (
        raw_values.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and np.result_type(raw_values, mean) == raw_values.dtype
        and np.result_type(raw_values, std) == raw_values.dtype
    ):
        # Boolean selection is private. Reuse its two arithmetic destinations
        # only when scalar promotion preserves the original operation dtype.
        np.subtract(raw_values, mean, out=raw_values)
        np.divide(raw_values, std, out=raw_values)
        out[valid] = raw_values
    else:
        out[valid] = ((raw_values - mean) / std).astype(np.float32, copy=False)
    height, width = data.shape
    if height > 1:
        out *= hanning_float32(height)[:, None]
    if width > 1:
        out *= hanning_float32(width)[None, :]
    return out, valid


def lag_indices(lags: np.ndarray, size: int) -> np.ndarray:
    return np.where(lags >= 0, lags, int(size) + lags).astype(np.int64)


def best_shift_from_windows(
    scores: np.ndarray,
    counts: np.ndarray,
    row_lags: np.ndarray,
    col_lags: np.ndarray,
    minimum_valid: int,
    error_message: str,
) -> ShiftResult:
    counts_int = np.rint(np.maximum(counts, 0.0)).astype(np.int64)
    usable = np.isfinite(scores) & (counts_int >= int(minimum_valid))
    if not bool(usable.any()):
        raise RuntimeError(error_message)
    maximum = float(np.nanmax(np.where(usable, scores, np.nan)))
    candidates = usable & (scores >= maximum - 1e-12)
    distances = (
        row_lags.astype(np.float64)[:, None] ** 2
        + col_lags.astype(np.float64)[None, :] ** 2
    )
    minimum_distance = float(np.min(distances[candidates]))
    candidates &= distances == minimum_distance
    maximum_count = int(np.max(counts_int[candidates]))
    candidates &= counts_int == maximum_count
    flat_index = int(np.flatnonzero(candidates.ravel())[0])
    row_index, col_index = np.unravel_index(flat_index, scores.shape)
    return ShiftResult(
        float(row_lags[row_index]),
        float(col_lags[col_index]),
        float(scores[row_index, col_index]),
        int(counts_int[row_index, col_index]),
    )


def peak_diagnostics_from_windows(
    scores: np.ndarray,
    counts: np.ndarray,
    row_lags: np.ndarray,
    col_lags: np.ndarray,
    minimum_valid: int,
    support_cap: int,
    error_message: str,
    selected_result: Optional[ShiftResult] = None,
) -> PeakDiagnostics:
    result = selected_result
    if result is None:
        result = best_shift_from_windows(
            scores,
            counts,
            row_lags,
            col_lags,
            minimum_valid,
            error_message,
        )
    row_candidates = np.flatnonzero(row_lags == int(round(result.row_shift)))
    col_candidates = np.flatnonzero(col_lags == int(round(result.col_shift)))
    if row_candidates.size == 0 or col_candidates.size == 0:
        raise RuntimeError("The selected correlation peak was not present in its lag grid.")
    row_index = int(row_candidates[0])
    col_index = int(col_candidates[0])

    counts_int = np.rint(np.maximum(counts, 0.0)).astype(np.int64)
    usable = np.isfinite(scores) & (counts_int >= int(minimum_valid))
    background = usable.copy()
    y0 = max(0, row_index - 1)
    y1 = min(scores.shape[0], row_index + 2)
    x0 = max(0, col_index - 1)
    x1 = min(scores.shape[1], col_index + 2)
    background[y0:y1, x0:x1] = False
    background_scores = np.asarray(scores[background], dtype=np.float64)
    if background_scores.size:
        second_best = float(np.max(background_scores))
        peak_margin = float(result.score - second_best)
    else:
        peak_margin = math.nan
    if background_scores.size >= 5:
        median = float(np.median(background_scores))
        mad = float(np.median(np.abs(background_scores - median)))
        sigma = max(1e-06, 1.4826 * mad)
        peak_zscore = float((result.score - median) / sigma)
    else:
        peak_zscore = math.nan

    overlap_fraction = float(result.valid_pixels) / max(1, int(support_cap))
    edge_hit = bool(
        row_index == 0
        or row_index == len(row_lags) - 1
        or col_index == 0
        or col_index == len(col_lags) - 1
    )
    return PeakDiagnostics(
        result=result,
        peak_margin=peak_margin,
        peak_zscore=peak_zscore,
        overlap_fraction=overlap_fraction,
        search_edge_hit=edge_hit,
    )


def _cpu_correlation_lag_window(
    left_fft: np.ndarray,
    right_fft: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
) -> np.ndarray:
    surface = cpu_irfft2(
        left_fft * np.conj(right_fft), fft_shape, overwrite=True
    )
    row_indices = lag_indices(row_lags, fft_shape[0])
    col_indices = lag_indices(col_lags, fft_shape[1])
    # Advanced indexing already owns its data; another copy only adds traffic.
    window = surface[np.ix_(row_indices, col_indices)]
    del surface
    return window


def fft_correlation_lag_window(
    left_fft: np.ndarray,
    right_fft: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
) -> np.ndarray:
    return _cpu_correlation_lag_window(
        left_fft,
        right_fft,
        fft_shape,
        row_lags,
        col_lags,
    )


def _phase_windows_cpu(
    ref_phase: np.ndarray,
    mov_phase: np.ndarray,
    ref_valid: np.ndarray,
    mov_valid: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cache = reference_phase_fft_cache() if not ref_phase.flags.writeable else None
    cached_phase = cache.get(ref_phase, fft_shape) if cache is not None else None
    if cached_phase is None:
        ref_fft = cpu_rfft2(ref_phase, fft_shape)
        if cache is not None:
            cache.put(ref_phase, fft_shape, (ref_fft,))
    else:
        ref_fft = cached_phase[0]
    mov_fft = cpu_rfft2(mov_phase, fft_shape)
    cross_power = ref_fft * np.conj(mov_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    phase_surface = cpu_irfft2(cross_power, fft_shape, overwrite=True)
    row_indices = lag_indices(row_lags, fft_shape[0])
    col_indices = lag_indices(col_lags, fft_shape[1])
    phase_window = phase_surface[np.ix_(row_indices, col_indices)]
    del ref_fft, mov_fft, cross_power, phase_surface

    cached_mask = cache.get(ref_valid, fft_shape) if cache is not None else None
    if cached_mask is None:
        ref_mask_fft = cpu_rfft2(ref_valid.astype(np.float32), fft_shape)
        if cache is not None:
            cache.put(ref_valid, fft_shape, (ref_mask_fft,))
    else:
        ref_mask_fft = cached_mask[0]
    mov_mask_fft = cpu_rfft2(mov_valid.astype(np.float32), fft_shape)
    overlap_surface = cpu_irfft2(
        ref_mask_fft * np.conj(mov_mask_fft), fft_shape, overwrite=True
    )
    overlap_window = overlap_surface[np.ix_(row_indices, col_indices)]
    return phase_window, overlap_window


def _phase_windows_gpu_job(
    ref_phase: np.ndarray,
    mov_phase: np.ndarray,
    ref_valid: np.ndarray,
    mov_valid: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(phase_gpu_memory_estimate(fft_shape, ref_phase.size))
        row_indices = cp.asarray(lag_indices(row_lags, fft_shape[0]))
        col_indices = cp.asarray(lag_indices(col_lags, fft_shape[1]))
        ref_gpu = cp.asarray(ref_phase, dtype=cp.float32)
        mov_gpu = cp.asarray(mov_phase, dtype=cp.float32)
        ref_fft = cp.fft.rfft2(ref_gpu, s=fft_shape)
        mov_fft = cp.fft.rfft2(mov_gpu, s=fft_shape)
        cross_power = ref_fft * cp.conj(mov_fft)
        cross_power /= cp.maximum(cp.abs(cross_power), 1e-12)
        phase_surface = cp.fft.irfft2(cross_power, s=fft_shape)
        phase_window = phase_surface[row_indices[:, None], col_indices[None, :]].copy()
        del ref_gpu, mov_gpu, ref_fft, mov_fft, cross_power, phase_surface

        ref_mask_fft = cp.fft.rfft2(cp.asarray(ref_valid, dtype=cp.float32), s=fft_shape)
        mov_mask_fft = cp.fft.rfft2(cp.asarray(mov_valid, dtype=cp.float32), s=fft_shape)
        overlap_surface = cp.fft.irfft2(
            ref_mask_fft * cp.conj(mov_mask_fft),
            s=fft_shape,
        )
        overlap_window = overlap_surface[
            row_indices[:, None],
            col_indices[None, :],
        ].copy()
        return gpu_download_arrays(phase_window, overlap_window)


def masked_phase_correlation_match(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_pixels: int,
    center_row: int = 0,
    center_col: int = 0,
    search_radius: Optional[int] = None,
) -> PeakDiagnostics:
    ref_crop, mov_crop = valid_search_crop(reference, moving, max_shift_pixels + 4)
    ref_phase, ref_valid = cached_standardize_reference(ref_crop)
    mov_phase, mov_valid = standardize_for_phase(mov_crop)

    if search_radius is None:
        row_min, row_max = -max_shift_pixels, max_shift_pixels
        col_min, col_max = -max_shift_pixels, max_shift_pixels
    else:
        row_min = max(-max_shift_pixels, int(center_row) - int(search_radius))
        row_max = min(max_shift_pixels, int(center_row) + int(search_radius))
        col_min = max(-max_shift_pixels, int(center_col) - int(search_radius))
        col_max = min(max_shift_pixels, int(center_col) + int(search_radius))
    row_lags = np.arange(row_min, row_max + 1, dtype=np.int64)
    col_lags = np.arange(col_min, col_max + 1, dtype=np.int64)
    fft_shape = linear_fft_shape(ref_phase.shape)

    phase_window = None
    overlap_window = None
    if gpu_is_active() and ref_phase.size >= GPU_MIN_PHASE_INPUT_PIXELS:
        try:
            phase_window, overlap_window = submit_gpu_job(
                _phase_windows_gpu_job,
                ref_phase,
                mov_phase,
                ref_valid,
                mov_valid,
                fft_shape,
                row_lags,
                col_lags,
            )
        except Exception as exc:
            record_gpu_fallback(exc)
    if phase_window is None or overlap_window is None:
        phase_window, overlap_window = _phase_windows_cpu(
            ref_phase,
            mov_phase,
            ref_valid,
            mov_valid,
            fft_shape,
            row_lags,
            col_lags,
        )

    support_cap = min(
        int(np.count_nonzero(ref_valid)),
        int(np.count_nonzero(mov_valid)),
    )
    minimum = min(1000, max(30, int(support_cap * 0.002)))
    return peak_diagnostics_from_windows(
        phase_window,
        overlap_window,
        row_lags,
        col_lags,
        minimum,
        support_cap,
        "Masked phase correlation did not find a valid shift. Check raster overlap and nodata coverage.",
    )


def masked_phase_correlation_peak(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_pixels: int,
    center_row: int = 0,
    center_col: int = 0,
    search_radius: Optional[int] = None,
) -> ShiftResult:
    return masked_phase_correlation_match(
        reference,
        moving,
        max_shift_pixels,
        center_row=center_row,
        center_col=center_col,
        search_radius=search_radius,
    ).result


class CPUReferenceFFTCache:
    """Scene-local LRU; only immutable structural features are admitted."""
    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def key(self, source: np.ndarray, fft_shape: Tuple[int, int]):
        return (
            id(source), source.shape, source.strides, source.dtype.str,
            tuple(fft_shape), id(scipy_fft), active_fft_workers(),
        )

    def get(self, source: np.ndarray, fft_shape: Tuple[int, int]):
        if not self.max_bytes or source.flags.writeable:
            return None
        key = self.key(source, fft_shape)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None or entry[0] is not source:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, source: np.ndarray, fft_shape: Tuple[int, int], spectra) -> None:
        if not self.max_bytes or source.flags.writeable:
            return
        size = int(source.nbytes) + sum(int(array.nbytes) for array in spectra)
        if size > self.max_bytes:
            return
        key = self.key(source, fft_shape)
        with self.lock:
            if key in self.entries:
                return
            while self.entries and self.bytes + size > self.max_bytes:
                _key, entry = self.entries.popitem(last=False)
                self.bytes -= entry[2]
            self.entries[key] = (source, tuple(spectra), size)
            self.bytes += size

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.bytes = 0


def qa_reference_fft_cache_budget() -> int:
    available = available_system_memory_bytes()
    if available <= 0:
        return 0
    return min(
        QA_REFERENCE_FFT_CACHE_MAX_BYTES,
        available // (16 * max(1, ACTIVE_SCENE_WORKERS)),
    )


class QAReferenceFeatureStore(dict):
    def __init__(self):
        super().__init__()
        self.fft_cache = CPUReferenceFFTCache(qa_reference_fft_cache_budget())
        self.texture_scores = {}
        self.previous = None

    def can_retain(self, data: np.ndarray) -> bool:
        available = available_system_memory_bytes()
        limit = min(
            QA_REUSE_IMAGE_MAX_BYTES,
            max(0, available) // (64 * max(1, ACTIVE_SCENE_WORKERS)),
        )
        return int(data.nbytes) <= limit and limit > 0

    def remember(self, moving: np.ndarray, cache) -> None:
        self.previous = None
        if not self.can_retain(moving):
            return
        # No cycle back to this reference store.
        self.previous = (
            moving, cache.candidate_cells, cache.candidate_texture_scores,
            cache.moving_features,
            {key: value for key, value in cache.zncc_sums.items()
             if key[1] == PAIRED_RESIDUAL_SEARCH_RADIUS_PIXELS and key[-1] == "cpu"},
            cache.future_gpu_batch,
        )

    def restore_for_identical(self, moving: np.ndarray, cache) -> bool:
        previous = self.previous
        self.previous = None
        if previous is None:
            return False
        original = previous[0]
        if (
            original.shape != moving.shape or original.dtype != moving.dtype
            or not original.flags.c_contiguous or not moving.flags.c_contiguous
            or not np.array_equal(original.view(np.uint8), moving.view(np.uint8))
        ):
            return False
        cache.candidate_cells = previous[1]
        cache.candidate_texture_scores = previous[2]
        cache.moving_features = previous[3]
        cache.zncc_sums.update(previous[4])
        cache.reusable_gpu_batch = previous[5]
        return True

    def clear(self) -> None:
        super().clear()
        self.fft_cache.clear()
        self.texture_scores.clear()
        self.previous = None

def _masked_zncc_sums_cpu(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    ref_valid: np.ndarray,
    mov_valid: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
    reference_fft_cache: Optional[CPUReferenceFFTCache] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached_reference = (
        reference_fft_cache.get(ref_crop, fft_shape)
        if reference_fft_cache is not None else None
    )
    mov_mask = mov_valid.astype(np.float64)
    mov_values = np.where(mov_valid, mov_crop, 0.0).astype(np.float64)
    overwrite_temporaries = ref_crop.size <= 600_000
    if cached_reference is None:
        ref_mask = ref_valid.astype(np.float64)
        ref_values = np.where(ref_valid, ref_crop, 0.0).astype(np.float64)
        ref_mask_fft = cpu_rfft2(
            ref_mask, fft_shape, overwrite=overwrite_temporaries
        )
    else:
        ref_mask_fft, ref_fft, ref_squared_fft = cached_reference
    mov_mask_fft = cpu_rfft2(
        mov_mask, fft_shape, overwrite=overwrite_temporaries
    )
    if cached_reference is None:
        ref_fft = cpu_rfft2(ref_values, fft_shape)
    mov_fft = cpu_rfft2(mov_values, fft_shape)
    # Every sum uses the same lag grid. Conjugate each right-hand spectrum
    # once, then reuse one product buffer without changing FFT sizes or sums.
    lag_selection = np.ix_(
        lag_indices(row_lags, fft_shape[0]),
        lag_indices(col_lags, fft_shape[1]),
    )
    np.conjugate(mov_mask_fft, out=mov_mask_fft)
    np.conjugate(mov_fft, out=mov_fft)
    product = np.empty_like(ref_fft)

    def correlation_window(left_fft, conjugated_right_fft):
        np.multiply(left_fft, conjugated_right_fft, out=product)
        surface = cpu_irfft2(product, fft_shape, overwrite=True)
        return surface[lag_selection]

    overlap = correlation_window(ref_mask_fft, mov_mask_fft)
    sum_ref = correlation_window(ref_fft, mov_mask_fft)
    sum_mov = correlation_window(ref_mask_fft, mov_fft)
    sum_products = correlation_window(ref_fft, mov_fft)
    if cached_reference is None:
        ref_squared_fft = cpu_rfft2(
            ref_values * ref_values,
            fft_shape,
            overwrite=overwrite_temporaries,
        )
        if reference_fft_cache is not None:
            reference_fft_cache.put(
                ref_crop, fft_shape, (ref_mask_fft, ref_fft, ref_squared_fft)
            )
    sum_ref_squared = correlation_window(ref_squared_fft, mov_mask_fft)
    del ref_squared_fft
    mov_squared_fft = cpu_rfft2(
        mov_values * mov_values,
        fft_shape,
        overwrite=overwrite_temporaries,
    )
    np.conjugate(mov_squared_fft, out=mov_squared_fft)
    sum_mov_squared = correlation_window(ref_mask_fft, mov_squared_fft)
    return overlap, sum_ref, sum_mov, sum_products, sum_ref_squared, sum_mov_squared


def _gpu_correlation_window(
    left_fft,
    right_fft,
    fft_shape: Tuple[int, int],
    row_indices,
    col_indices,
):
    surface = cp.fft.irfft2(left_fft * cp.conj(right_fft), s=fft_shape)
    window = surface[row_indices[:, None], col_indices[None, :]].copy()
    del surface
    return window


@lru_cache(maxsize=1)
def _gpu_masked_float32_fft_inputs_kernel():
    """Form masks and FFT values without flushing float32 subnormal inputs."""
    assert cp is not None
    return cp.ElementwiseKernel(
        "float32 source", "float64 mask, float64 values",
        """
        const bool valid = isfinite(source);
        mask = valid ? 1.0 : 0.0;
        double converted;
        asm("cvt.f64.f32 %0, %1;" : "=d"(converted) : "f"(source));
        values = valid ? converted : 0.0;
        """,
        "tir_masked_float32_fft_inputs",
    )


@lru_cache(maxsize=1)
def _gpu_qa_score_kernel():
    """Evaluate QA scores with the original separately rounded float64 math."""
    assert cp is not None
    return cp.ElementwiseKernel(
        "float64 count, float64 minimum, float64 sum_ref, float64 sum_mov, "
        "float64 sum_products, float64 sum_ref_squared, float64 sum_mov_squared",
        "float64 score",
        """
        const double covariance = __dsub_rn(sum_products,
            __ddiv_rn(__dmul_rn(sum_ref, sum_mov), count));
        const double variance_ref = __dsub_rn(sum_ref_squared,
            __ddiv_rn(__dmul_rn(sum_ref, sum_ref), count));
        const double variance_mov = __dsub_rn(sum_mov_squared,
            __ddiv_rn(__dmul_rn(sum_mov, sum_mov), count));
        const double positive_ref = isnan(variance_ref)
            ? variance_ref : fmax(variance_ref, 0.0);
        const double positive_mov = isnan(variance_mov)
            ? variance_mov : fmax(variance_mov, 0.0);
        const double denominator = __dsqrt_rn(__dmul_rn(positive_ref, positive_mov));
        const double raw_score = __ddiv_rn(covariance, denominator);
        const bool usable = count >= minimum && isfinite(raw_score)
            && variance_ref > 1e-12 && variance_mov > 1e-12 && denominator > 0.0;
        score = usable ? fmin(fmax(raw_score, -1.0), 1.0)
            : __longlong_as_double(0x7ff8000000000000LL);
        """,
        "tir_qa_score_from_sums",
        options=("--fmad=false",),
    )


def _gpu_masked_fft_inputs(data: np.ndarray):
    """Upload native float32 once; retain the existing path for other dtypes."""
    if isinstance(data, np.ndarray) and data.dtype == np.dtype(np.float32):
        return _gpu_masked_float32_fft_inputs_kernel()(cp.asarray(data))
    values = _gpu_asarray_float64(data)
    valid = cp.isfinite(values)
    return (valid.astype(cp.float64), cp.where(valid, values, 0.0))


def _gpu_zncc_windows(
    reference: np.ndarray,
    moving: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
    batched: bool,
):
    """Keep every FFT unchanged while sharing input preparation and conjugates."""
    row_indices = cp.asarray(lag_indices(row_lags, fft_shape[0]))
    col_indices = cp.asarray(lag_indices(col_lags, fft_shape[1]))
    ref_mask, ref_values = _gpu_masked_fft_inputs(reference)
    mov_mask, mov_values = _gpu_masked_fft_inputs(moving)
    ref_mask_fft = cp.fft.rfft2(ref_mask, s=fft_shape, axes=(-2, -1))
    mov_mask_fft = cp.fft.rfft2(mov_mask, s=fft_shape, axes=(-2, -1))
    ref_fft = cp.fft.rfft2(ref_values, s=fft_shape, axes=(-2, -1))
    mov_fft = cp.fft.rfft2(mov_values, s=fft_shape, axes=(-2, -1))
    # These transforms are used only as right-hand correlation operands.
    # Conjugating them once avoids repeated full-sized temporary arrays.
    cp.conj(mov_mask_fft, out=mov_mask_fft)
    cp.conj(mov_fft, out=mov_fft)

    def correlation_window(left, right):
        product = left * right
        surface = cp.fft.irfft2(product, s=fft_shape, axes=(-2, -1))
        del product
        # Advanced indexing already returns independent storage.
        if batched:
            window = surface[:, row_indices[:, None], col_indices[None, :]]
        else:
            window = surface[row_indices[:, None], col_indices[None, :]]
        return window

    overlap = correlation_window(ref_mask_fft, mov_mask_fft)
    sum_ref = correlation_window(ref_fft, mov_mask_fft)
    sum_mov = correlation_window(ref_mask_fft, mov_fft)
    sum_products = correlation_window(ref_fft, mov_fft)
    ref_squared_fft = cp.fft.rfft2(ref_values * ref_values, s=fft_shape, axes=(-2, -1))
    sum_ref_squared = correlation_window(ref_squared_fft, mov_mask_fft)
    del ref_squared_fft
    mov_squared_fft = cp.fft.rfft2(mov_values * mov_values, s=fft_shape, axes=(-2, -1))
    cp.conj(mov_squared_fft, out=mov_squared_fft)
    sum_mov_squared = correlation_window(ref_mask_fft, mov_squared_fft)
    return (overlap, sum_ref, sum_mov, sum_products, sum_ref_squared, sum_mov_squared)


def _gpu_qa_scores_from_windows(sums, min_match_pixels: int, min_overlap_ratio: float):
    overlap, sum_ref, sum_mov, sum_products, sum_ref_squared, sum_mov_squared = sums
    counts = cp.rint(cp.maximum(overlap, 0.0))
    max_overlap = cp.max(counts, axis=(1, 2))
    # Preserve the original rounding, reduction, and threshold calculation.
    minimum = cp.maximum(
        float(min_match_pixels), cp.ceil(max_overlap * float(min_overlap_ratio)),
    )[:, None, None]
    scores = _gpu_qa_score_kernel()(
        counts, minimum, sum_ref, sum_mov, sum_products,
        sum_ref_squared, sum_mov_squared,
    )
    return (scores, counts, max_overlap)


def _masked_zncc_sums_gpu_job(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        require_gpu_memory(zncc_gpu_memory_estimate(fft_shape, ref_crop.size))
        sums = _gpu_zncc_windows(ref_crop, mov_crop, fft_shape, row_lags, col_lags, False)
        result = cp.asnumpy(cp.stack(sums, axis=0))
        return tuple(result[index] for index in range(6))


def zncc_scores_from_sums(
    sums: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    minimum_valid: int,
    minimum_overlap_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    overlap, sum_ref, sum_mov, sum_products, sum_ref_squared, sum_mov_squared = sums
    counts = np.rint(np.maximum(overlap, 0.0))
    max_overlap = int(counts.max()) if counts.size else 0
    minimum = max(
        int(minimum_valid),
        int(math.ceil(max_overlap * float(minimum_overlap_ratio))),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = sum_products - ((sum_ref * sum_mov) / counts)
        variance_ref = sum_ref_squared - ((sum_ref * sum_ref) / counts)
        variance_mov = sum_mov_squared - ((sum_mov * sum_mov) / counts)
        denominator = np.sqrt(
            np.maximum(variance_ref, 0.0) * np.maximum(variance_mov, 0.0)
        )
        scores = covariance / denominator
    usable = (
        (counts >= minimum)
        & np.isfinite(scores)
        & (variance_ref > 1e-12)
        & (variance_mov > 1e-12)
        & (denominator > 0.0)
    )
    scores = np.where(usable, np.clip(scores, -1.0, 1.0), np.nan)
    return scores, counts, max_overlap


def masked_zncc_fft_peak(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_pixels: int,
    center_row: int = 0,
    center_col: int = 0,
    search_radius: Optional[int] = None,
) -> ShiftResult:
    """Find exact pairwise masked ZNCC over a rectangular integer-lag search."""
    if reference.shape != moving.shape:
        raise RuntimeError("Reference and moving arrays must have the same shape for masked ZNCC.")

    ref_crop, mov_crop = valid_search_crop(reference, moving, max_shift_pixels + 4)
    ref_valid = np.isfinite(ref_crop)
    mov_valid = np.isfinite(mov_crop)
    if int(np.count_nonzero(ref_valid)) < 30 or int(np.count_nonzero(mov_valid)) < 30:
        raise RuntimeError("Not enough valid pixels for masked ZNCC.")

    if search_radius is None:
        row_min, row_max = -max_shift_pixels, max_shift_pixels
        col_min, col_max = -max_shift_pixels, max_shift_pixels
    else:
        row_min = max(-max_shift_pixels, int(center_row) - int(search_radius))
        row_max = min(max_shift_pixels, int(center_row) + int(search_radius))
        col_min = max(-max_shift_pixels, int(center_col) - int(search_radius))
        col_max = min(max_shift_pixels, int(center_col) + int(search_radius))
    row_lags = np.arange(row_min, row_max + 1, dtype=np.int64)
    col_lags = np.arange(col_min, col_max + 1, dtype=np.int64)
    fft_shape = linear_fft_shape(ref_crop.shape)

    sums = None
    if gpu_is_active() and ref_crop.size >= GPU_MIN_ZNCC_INPUT_PIXELS:
        try:
            sums = submit_gpu_job(
                _masked_zncc_sums_gpu_job,
                ref_crop,
                mov_crop,
                fft_shape,
                row_lags,
                col_lags,
            )
        except Exception as exc:
            record_gpu_fallback(exc)
    if sums is None:
        sums = _masked_zncc_sums_cpu(
            ref_crop,
            mov_crop,
            ref_valid,
            mov_valid,
            fft_shape,
            row_lags,
            col_lags,
        )
    scores, counts, _max_overlap = zncc_scores_from_sums(
        sums,
        min_valid_pixels_for_level(ref_crop, mov_crop),
        ROBUST_MIN_OVERLAP_RATIO,
    )
    return best_shift_from_windows(
        scores,
        counts,
        row_lags,
        col_lags,
        0,
        "Exact masked ZNCC did not find a valid shift.",
    )


def valid_bbox(data_a: np.ndarray, data_b: np.ndarray):
    valid = np.isfinite(data_a) | np.isfinite(data_b)
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


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


def _separable_nearest_indices(inverse: Affine, height: int, width: int):
    """Build exact axis maps for a translation/axis-aligned nearest warp.

    Retain the original separate float64 multiply/add/rint operations, including
    ties-to-even at half-pixel coordinates. Ordinary translations reduce to
    two slices; half-pixel ties and scaling retain the explicit index maps.
    """
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    source_cols = np.rint((inverse.a * columns + 0.0) + inverse.c).astype(np.int64)
    source_rows = np.rint((0.0 + inverse.e * rows) + inverse.f).astype(np.int64)
    dest_cols = np.flatnonzero((source_cols >= 0) & (source_cols < width))
    dest_rows = np.flatnonzero((source_rows >= 0) & (source_rows < height))
    source_cols = source_cols[dest_cols]
    source_rows = source_rows[dest_rows]
    axes = (dest_rows, dest_cols, source_rows, source_cols)
    if all(axis.size and bool(np.all(np.diff(axis) == 1)) for axis in axes):
        slices = tuple(slice(int(axis[0]), int(axis[-1]) + 1) for axis in axes)
        return slices[:2], slices[2:]
    return np.ix_(dest_rows, dest_cols), np.ix_(source_rows, source_cols)


def _warp_array_nearest_cpu(
    data: np.ndarray,
    pixel_affine: Affine,
    block_rows: int,
) -> np.ndarray:
    height, width = data.shape
    try:
        inverse = ~pixel_affine
    except Exception as exc:
        raise RuntimeError(f"The pixel affine is not invertible: {exc}") from exc

    if inverse.b == 0.0 and inverse.d == 0.0:
        out = np.full((height, width), np.nan, dtype=np.float32)
        destination_index, source_index = _separable_nearest_indices(inverse, height, width)
        out[destination_index] = data[source_index]
        return out

    if NATIVE_KERNELS_ACTIVE and data.dtype == np.dtype(np.float32) and data.flags.writeable:
        return _native_warp(data, inverse.a, inverse.b, inverse.c, inverse.d, inverse.e, inverse.f)

    out = np.full((height, width), np.nan, dtype=np.float32)
    destination_cols = np.arange(width, dtype=np.float64)[None, :]
    column_terms = inverse.a * destination_cols
    row_terms = inverse.d * destination_cols
    block_rows = max(1, int(block_rows))
    for row0 in range(0, height, block_rows):
        row1 = min(height, row0 + block_rows)
        destination_rows = np.arange(row0, row1, dtype=np.float64)[:, None]
        source_cols = (
            column_terms
            + inverse.b * destination_rows
            + inverse.c
        )
        source_rows = (
            row_terms
            + inverse.e * destination_rows
            + inverse.f
        )
        source_cols = np.rint(source_cols).astype(np.int64)
        source_rows = np.rint(source_rows).astype(np.int64)
        usable = (
            (source_rows >= 0)
            & (source_rows < height)
            & (source_cols >= 0)
            & (source_cols < width)
        )
        block = out[row0:row1]
        block[usable] = data[source_rows[usable], source_cols[usable]]
    return out


@lru_cache(maxsize=2)
def _gpu_nearest_warp_kernel(paired: bool = False):
    """Gather nearest pixels without dynamic boolean-index compaction.

    Explicit rounded double operations preserve the original separate CuPy
    multiply/add kernels, including nearest-even decisions at half pixels.
    """
    assert cp is not None
    source_parameters = "raw float32 first_source"
    output_parameters = "float32 first_out"
    initialize_outputs = "first_out = __int_as_float(0x7fc00000);"
    gather_outputs = "first_out = first_source[source_index];"
    if paired:
        source_parameters += ", raw float32 second_source"
        output_parameters += ", float32 second_out"
        initialize_outputs += " second_out = __int_as_float(0x7fc00000);"
        gather_outputs += " second_out = second_source[source_index];"
    return cp.ElementwiseKernel(
        source_parameters
        + ", int64 width, int64 height, float64 a, float64 b, float64 c"
        + ", float64 d, float64 e, float64 f",
        output_parameters,
        """
        const double destination_col = (double)(i % width);
        const double destination_row = (double)(i / width);
        const double source_col = rint(__dadd_rn(
            __dadd_rn(__dmul_rn(a, destination_col),
                      __dmul_rn(b, destination_row)), c));
        const double source_row = rint(__dadd_rn(
            __dadd_rn(__dmul_rn(d, destination_col),
                      __dmul_rn(e, destination_row)), f));
        """
        + initialize_outputs
        + """
        if (source_row >= 0.0 && source_row < (double)height &&
            source_col >= 0.0 && source_col < (double)width) {
            const long long source_index =
                (long long)source_row * width + (long long)source_col;
        """
        + gather_outputs
        + "\n}",
        "tir_nearest_warp_pair" if paired else "tir_nearest_warp",
        options=("--fmad=false",),
    )


def _warp_array_nearest_gpu_job(
    data: np.ndarray,
    inverse_coefficients: Tuple[float, float, float, float, float, float],
    block_rows: int,
) -> np.ndarray:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        height, width = data.shape
        block_rows = max(1, min(int(block_rows), height))
        require_gpu_memory(warp_gpu_memory_estimate(data.shape, block_rows))
        source = cp.asarray(data, dtype=cp.float32, order="C")
        out = _gpu_nearest_warp_kernel(False)(
            source,
            np.int64(width),
            np.int64(height),
            *(np.float64(value) for value in inverse_coefficients),
            size=int(source.size),
        )
        return cp.asnumpy(out.reshape((height, width)))


def warp_array_nearest(
    data: np.ndarray,
    pixel_affine: Affine,
    block_rows: int = 256,
) -> np.ndarray:
    """Apply a source-to-destination pixel affine with one nearest-neighbor pass."""
    if (
        pixel_affine.a == 1.0
        and pixel_affine.b == 0.0
        and pixel_affine.c == 0.0
        and pixel_affine.d == 0.0
        and pixel_affine.e == 1.0
        and pixel_affine.f == 0.0
    ):
        return np.asarray(data, dtype=np.float32).copy()
    try:
        inverse = ~pixel_affine
    except Exception as exc:
        raise RuntimeError(f"The pixel affine is not invertible: {exc}") from exc
    if gpu_is_active() and data.size >= GPU_MIN_WARP_PIXELS:
        try:
            return submit_gpu_job(
                _warp_array_nearest_gpu_job,
                data,
                (
                    float(inverse.a),
                    float(inverse.b),
                    float(inverse.c),
                    float(inverse.d),
                    float(inverse.e),
                    float(inverse.f),
                ),
                max(int(block_rows), GPU_WARP_BLOCK_ROWS),
            )
        except Exception as exc:
            record_gpu_fallback(exc)
    return _warp_array_nearest_cpu(data, pixel_affine, block_rows)


def _warp_array_nearest_pair_cpu(
    first: np.ndarray,
    second: np.ndarray,
    pixel_affine: Affine,
    block_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = first.shape
    if (EXACT_CPU_KERNELS_ACTIVE and first.dtype == np.float32 and second.dtype == np.float32
            and first.flags.writeable and second.flags.writeable
            and (pixel_affine.b != 0.0 or pixel_affine.d != 0.0)):
        try:
            inverse = ~pixel_affine
        except Exception as exc:
            raise RuntimeError(f"The pixel affine is not invertible: {exc}") from exc
        return _exact_warp_pair(first, second, *tuple(inverse)[:6])
    first_out = np.full((height, width), np.nan, dtype=np.float32)
    second_out = np.full((height, width), np.nan, dtype=np.float32)
    try:
        inverse = ~pixel_affine
    except Exception as exc:
        raise RuntimeError(f"The pixel affine is not invertible: {exc}") from exc

    if inverse.b == 0.0 and inverse.d == 0.0:
        destination_index, source_index = _separable_nearest_indices(inverse, height, width)
        first_out[destination_index] = first[source_index]
        second_out[destination_index] = second[source_index]
        return first_out, second_out

    destination_cols = np.arange(width, dtype=np.float64)[None, :]
    column_terms = inverse.a * destination_cols
    row_terms = inverse.d * destination_cols
    block_rows = max(1, int(block_rows))
    for row0 in range(0, height, block_rows):
        row1 = min(height, row0 + block_rows)
        destination_rows = np.arange(row0, row1, dtype=np.float64)[:, None]
        source_cols = np.rint(
            column_terms
            + inverse.b * destination_rows
            + inverse.c
        ).astype(np.int64)
        source_rows = np.rint(
            row_terms
            + inverse.e * destination_rows
            + inverse.f
        ).astype(np.int64)
        usable = (
            (source_rows >= 0)
            & (source_rows < height)
            & (source_cols >= 0)
            & (source_cols < width)
        )
        first_block = first_out[row0:row1]
        second_block = second_out[row0:row1]
        source_index = source_rows[usable], source_cols[usable]
        first_block[usable] = first[source_index]
        second_block[usable] = second[source_index]
    return first_out, second_out


def _warp_array_nearest_pair_gpu_job(
    first: np.ndarray,
    second: np.ndarray,
    inverse_coefficients: Tuple[float, float, float, float, float, float],
    block_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        height, width = first.shape
        block_rows = max(1, min(int(block_rows), height))
        require_gpu_memory(warp_pair_gpu_memory_estimate(first.shape, block_rows))
        first_source = cp.asarray(first, dtype=cp.float32, order="C")
        second_source = cp.asarray(second, dtype=cp.float32, order="C")
        first_out, second_out = _gpu_nearest_warp_kernel(True)(
            first_source,
            second_source,
            np.int64(width),
            np.int64(height),
            *(np.float64(value) for value in inverse_coefficients),
            size=int(first_source.size),
        )
        return gpu_download_arrays(
            first_out.reshape((height, width)),
            second_out.reshape((height, width)),
        )


def warp_array_nearest_pair(
    first: np.ndarray,
    second: np.ndarray,
    pixel_affine: Affine,
    block_rows: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp two same-grid arrays with one shared nearest-neighbor index map."""
    if first.shape != second.shape:
        raise RuntimeError("Paired warp arrays must have identical shapes.")
    if pixel_affine == Affine.identity():
        return (
            np.asarray(first, dtype=np.float32).copy(),
            np.asarray(second, dtype=np.float32).copy(),
        )
    try:
        inverse = ~pixel_affine
    except Exception as exc:
        raise RuntimeError(f"The pixel affine is not invertible: {exc}") from exc
    if gpu_is_active() and first.size >= GPU_MIN_WARP_PIXELS:
        try:
            return submit_gpu_job(
                _warp_array_nearest_pair_gpu_job,
                first,
                second,
                (
                    float(inverse.a),
                    float(inverse.b),
                    float(inverse.c),
                    float(inverse.d),
                    float(inverse.e),
                    float(inverse.f),
                ),
                max(int(block_rows), GPU_WARP_BLOCK_ROWS),
            )
        except Exception as exc:
            record_gpu_fallback(exc)
    return _warp_array_nearest_pair_cpu(
        first,
        second,
        pixel_affine,
        block_rows,
    )


def _warp_array_continuous_numpy(data, inverse, block_rows=256):
    """Bilinear fallback with bounded coordinate buffers."""
    source = np.asarray(data, dtype=np.float32)
    height, width = source.shape
    output = np.full(source.shape, np.nan, dtype=np.float32)
    block_rows = max(1, int(block_rows))
    for y0 in range(0, height, block_rows):
        y1 = min(height, y0 + block_rows)
        yy = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        for x0 in range(0, width, 2048):
            x1 = min(width, x0 + 2048)
            xx = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
            sy = inverse.d * xx + inverse.e * yy + inverse.f - 0.5
            sx = inverse.a * xx + inverse.b * yy + inverse.c - 0.5
            inside = ((sy >= 0.0) & (sy <= height - 1)
                      & (sx >= 0.0) & (sx <= width - 1))
            sy = np.clip(sy, 0.0, height - 1)
            sx = np.clip(sx, 0.0, width - 1)
            iy, ix = np.floor(sy).astype(np.intp), np.floor(sx).astype(np.intp)
            jy, jx = np.minimum(iy + 1, height - 1), np.minimum(ix + 1, width - 1)
            fy, fx = sy - iy, sx - ix
            total = np.zeros(sy.shape, dtype=np.float64)
            support = np.zeros(sy.shape, dtype=np.float64)
            for rr, cc, weight in ((iy, ix, (1.0 - fy) * (1.0 - fx)),
                                   (iy, jx, (1.0 - fy) * fx),
                                   (jy, ix, fy * (1.0 - fx)),
                                   (jy, jx, fy * fx)):
                values = source[rr, cc]
                valid = np.isfinite(values)
                total += np.where(valid, values, 0.0) * weight
                support += valid * weight
            total[~inside | (support < 0.9999)] = np.nan
            output[y0:y1, x0:x1] = total
    return output


def warp_array_continuous(data, pixel_affine, block_rows=256):
    """Native-grid bilinear warp; pixel-corner affine, strict valid support."""
    source = np.asarray(data, dtype=np.float32)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("The continuous warp requires a nonempty 2-D raster.")
    coefficients = tuple(pixel_affine)[:6]
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise ValueError("The pixel affine contains nonfinite coefficients.")
    if (pixel_affine.a == 1.0 and pixel_affine.b == 0.0
            and pixel_affine.d == 0.0 and pixel_affine.e == 1.0
            and pixel_affine.c == round(pixel_affine.c)
            and pixel_affine.f == round(pixel_affine.f)):
        output = np.full(source.shape, np.nan, dtype=np.float32)
        selection = shifted_slices(source.shape, int(pixel_affine.f), int(pixel_affine.c))
        if selection is not None:
            ref_slice, mov_slice = selection
            output[ref_slice] = source[mov_slice]
            output[~np.isfinite(output)] = np.nan
        return output
    try:
        inverse = ~pixel_affine
    except Exception as exc:
        raise RuntimeError(f"The pixel affine is not invertible: {exc}") from exc
    if scipy_ndimage is None:
        return _warp_array_continuous_numpy(source, inverse, block_rows)
    matrix = np.asarray([[inverse.e, inverse.d], [inverse.b, inverse.a]], dtype=np.float64)
    offset = np.asarray([inverse.f + 0.5 * (inverse.d + inverse.e) - 0.5,
                         inverse.c + 0.5 * (inverse.a + inverse.b) - 0.5])
    valid = np.isfinite(source)
    options = dict(matrix=matrix, offset=offset, output_shape=source.shape,
                   order=1, mode="constant", prefilter=False)
    if valid.all():
        return scipy_ndimage.affine_transform(source, cval=np.nan, **options)
    output = scipy_ndimage.affine_transform(np.where(valid, source, 0.0), cval=0.0, **options)
    support = scipy_ndimage.affine_transform(valid.astype(np.float32), cval=0.0, **options)
    output[support < 0.9999] = np.nan
    return output


def warp_array_continuous_pair(first, second, pixel_affine, block_rows=256):
    if first.shape != second.shape:
        raise RuntimeError("Paired warp arrays must have identical shapes.")
    return (warp_array_continuous(first, pixel_affine, block_rows),
            warp_array_continuous(second, pixel_affine, block_rows))


def affine_for_downsample(pixel_affine: Affine, factor: int) -> Affine:
    factor = max(1, int(factor))
    if factor == 1:
        return pixel_affine
    return Affine.scale(1.0 / factor) * pixel_affine * Affine.scale(factor)


def compose_pixel_affines(current: Affine, incremental: Affine) -> Affine:
    """Compose original->current with current->next into original->next."""
    return incremental * current


def affine_shift_at(pixel_affine: Affine, row: float, col: float) -> Tuple[float, float]:
    moved_col, moved_row = pixel_affine * (float(col), float(row))
    return float(moved_row - row), float(moved_col - col)


def affine_deviation_pixels(
    candidate: Affine,
    baseline: Affine,
    shape: Tuple[int, int],
) -> float:
    """Maximum candidate-minus-baseline displacement over a 5x5 image grid."""
    height, width = shape
    rows = np.linspace(0.0, float(max(0, height - 1)), 5)
    cols = np.linspace(0.0, float(max(0, width - 1)), 5)
    maximum = 0.0
    for row in rows:
        for col in cols:
            candidate_row, candidate_col = affine_shift_at(candidate, row, col)
            baseline_row, baseline_col = affine_shift_at(baseline, row, col)
            maximum = max(
                maximum,
                math.hypot(
                    candidate_row - baseline_row,
                    candidate_col - baseline_col,
                ),
            )
    return float(maximum)


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


def zero_shift_ncc(reference: np.ndarray, moving: np.ndarray) -> Tuple[float, int]:
    min_valid = min_valid_pixels_for_level(reference, moving)
    return normalized_correlation(reference, moving, 0, 0, min_valid)


def validate_alignment_candidate(
    reference: np.ndarray,
    moving: np.ndarray,
    baseline_affine: Affine,
    candidate_affine: Affine,
    minimum_score_gain: Optional[float] = None,
) -> AlignmentValidation:
    """Compare global and regional NCC after two candidate transforms."""
    factor = max(1, int(WARP_VALIDATION_DOWNSAMPLE))
    reference_levels = cached_reference_pyramid_levels(reference, factor)
    reference_small = (
        reference_levels[factor]
        if reference_levels is not None and factor in reference_levels
        else nanmean_downsample_power_of_two(reference, factor)
    )
    moving_small = nanmean_downsample_power_of_two(moving, factor)
    baseline_small = affine_for_downsample(baseline_affine, factor)
    candidate_small = affine_for_downsample(candidate_affine, factor)

    baseline_aligned = warp_array_continuous(moving_small, baseline_small)
    candidate_aligned = warp_array_continuous(moving_small, candidate_small)
    common = np.isfinite(reference_small) & np.isfinite(baseline_aligned) & np.isfinite(candidate_aligned)
    reference_small = np.where(common, reference_small, np.nan)
    baseline_aligned[~common] = np.nan
    candidate_aligned[~common] = np.nan
    baseline_score, _ = zero_shift_ncc(reference_small, baseline_aligned)
    candidate_score, _ = zero_shift_ncc(reference_small, candidate_aligned)
    score_gain = candidate_score - baseline_score

    # Silent acceptance guards cannot accept a candidate that fails the global
    # score check, regardless of regional scores. Detailed/logged validation
    # leaves the optional threshold unset and still evaluates every window.
    if minimum_score_gain is not None and (
        not math.isfinite(score_gain) or score_gain < minimum_score_gain
    ):
        return AlignmentValidation(
            float(baseline_score), float(candidate_score), float(score_gain),
            0, 1.0, -math.inf, -math.inf,
        )

    local_gains = []
    windows = local_window_slices(reference_small, baseline_aligned, 0)

    def local_gain(window):
        wy0, wy1, wx0, wx1 = window
        ref_win = reference_small[wy0:wy1, wx0:wx1]
        baseline_win = baseline_aligned[wy0:wy1, wx0:wx1]
        candidate_win = candidate_aligned[wy0:wy1, wx0:wx1]
        baseline_local, _ = zero_shift_ncc(ref_win, baseline_win)
        candidate_local, _ = zero_shift_ncc(ref_win, candidate_win)
        if math.isfinite(baseline_local) and math.isfinite(candidate_local):
            return candidate_local - baseline_local
        return None

    gain_results = [local_gain(window) for window in windows]
    local_gains.extend(gain for gain in gain_results if gain is not None)

    if local_gains:
        gains = np.asarray(local_gains, dtype=np.float64)
        worsened_fraction = float(np.mean(gains < -0.005))
        median_local_gain = float(np.median(gains))
        worst_local_gain = float(np.min(gains))
    else:
        worsened_fraction = 1.0
        median_local_gain = -math.inf
        worst_local_gain = -math.inf

    return AlignmentValidation(
        float(baseline_score),
        float(candidate_score),
        float(score_gain),
        len(local_gains),
        worsened_fraction,
        median_local_gain,
        worst_local_gain,
    )


def guarded_warp_choice(
    reference: np.ndarray,
    moving: np.ndarray,
    baseline: WarpModel,
    candidate: WarpModel,
    shape: Tuple[int, int],
    stage_label: str,
    logger: Optional[ConsoleLog] = None,
    confidence_multiplier: float = 1.0,
) -> WarpModel:
    """Accept a warp only when its evidence scales with its nonrigid magnitude."""
    if candidate.is_translation_only():
        if logger is not None:
            logger.info(f"{stage_label}: no nonrigid candidate; retaining translation")
        return baseline

    multiplier = max(1.0, float(confidence_multiplier))
    deformation = affine_deviation_pixels(
        candidate.pixel_affine(), baseline.pixel_affine(), shape
    )
    required_controls = min(
        LOCAL_WARP_GRID * LOCAL_WARP_GRID,
        LOCAL_WARP_MIN_CONTROLS
        + int(math.ceil(
            multiplier
            * WARP_CONTROL_POINTS_PER_5_PIXELS
            * deformation
            / 5.0
        )),
    )
    maximum_rmse = max(
        WARP_MIN_MAX_RMSE_PIXELS,
        WARP_BASE_MAX_RMSE_PIXELS
        - multiplier * WARP_RMSE_TIGHTEN_PER_PIXEL * deformation,
    )
    minimum_mean_score = (
        WARP_BASE_MIN_MEAN_CONTROL_SCORE
        + multiplier * WARP_MIN_MEAN_CONTROL_SCORE_PER_PIXEL * deformation
    )
    minimum_q25_score = (
        WARP_BASE_MIN_Q25_CONTROL_SCORE
        + multiplier * WARP_MIN_Q25_CONTROL_SCORE_PER_PIXEL * deformation
    )
    required_gain = multiplier * (
        WARP_BASE_MIN_GLOBAL_SCORE_GAIN
        + WARP_MIN_GLOBAL_SCORE_GAIN_PER_PIXEL * deformation
    )
    maximum_worsened_fraction = max(
        WARP_MIN_MAX_WORSENED_FRACTION,
        WARP_BASE_MAX_WORSENED_FRACTION
        - multiplier * WARP_WORSENED_FRACTION_TIGHTEN_PER_PIXEL * deformation,
    )
    maximum_local_drop = max(
        WARP_MIN_MAX_LOCAL_SCORE_DROP,
        WARP_BASE_MAX_LOCAL_SCORE_DROP
        - multiplier * WARP_LOCAL_SCORE_DROP_TIGHTEN_PER_PIXEL * deformation,
    )

    reasons = []
    if not math.isfinite(deformation) or deformation > WARP_MAX_NONRIGID_DEVIATION_PIXELS:
        reasons.append(
            f"nonrigid deviation {deformation:.2f}px exceeds "
            f"{WARP_MAX_NONRIGID_DEVIATION_PIXELS:.1f}px"
        )
    if candidate.control_points < required_controls:
        reasons.append(
            f"controls {candidate.control_points} < required {required_controls}"
        )
    if not math.isfinite(candidate.rmse_px) or candidate.rmse_px > maximum_rmse:
        reasons.append(
            f"RMSE {candidate.rmse_px:.2f}px > allowed {maximum_rmse:.2f}px"
        )
    if not math.isfinite(candidate.mean_score) or candidate.mean_score < minimum_mean_score:
        reasons.append(
            f"mean control score {candidate.mean_score:.4f} < {minimum_mean_score:.4f}"
        )
    if not math.isfinite(candidate.score_q25) or candidate.score_q25 < minimum_q25_score:
        reasons.append(
            f"Q25 control score {candidate.score_q25:.4f} < {minimum_q25_score:.4f}"
        )

    validation = None
    if not reasons:
        try:
            validation = validate_alignment_candidate(
                reference,
                moving,
                baseline.pixel_affine(),
                candidate.pixel_affine(),
                minimum_score_gain=required_gain if logger is None else None,
            )
        except Exception as exc:
            reasons.append(f"validation failed: {exc}")

    if validation is not None:
        if validation.valid_windows < WARP_MIN_VALIDATION_WINDOWS:
            reasons.append(
                f"regional validation windows {validation.valid_windows} < "
                f"{WARP_MIN_VALIDATION_WINDOWS}"
            )
        if not math.isfinite(validation.score_gain) or validation.score_gain < required_gain:
            reasons.append(
                f"global NCC gain {validation.score_gain:+.5f} < required {required_gain:.5f}"
            )
        if validation.worsened_fraction > maximum_worsened_fraction:
            reasons.append(
                f"worsened-window fraction {validation.worsened_fraction:.1%} > "
                f"{maximum_worsened_fraction:.1%}"
            )
        if validation.worst_local_gain < -maximum_local_drop:
            reasons.append(
                f"worst local NCC change {validation.worst_local_gain:+.5f} < "
                f"{-maximum_local_drop:+.5f}"
            )

    if logger is not None:
        validation_text = "not run"
        if validation is not None:
            validation_text = (
                f"global gain={validation.score_gain:+.5f}; "
                f"local median={validation.median_local_gain:+.5f}; "
                f"worsened={validation.worsened_fraction:.1%}; "
                f"worst={validation.worst_local_gain:+.5f}; "
                f"windows={validation.valid_windows}"
            )
        logger.info(
            f"{stage_label} confidence: deformation={deformation:.2f}px; "
            f"controls={candidate.control_points}/{required_controls}; "
            f"rmse={candidate.rmse_px:.2f}/{maximum_rmse:.2f}px; "
            f"mean={candidate.mean_score:.4f}/{minimum_mean_score:.4f}; "
            f"q25={candidate.score_q25:.4f}/{minimum_q25_score:.4f}; "
            f"{validation_text}"
        )

    if reasons:
        if logger is not None:
            logger.info(f"{stage_label} rejected: " + "; ".join(reasons))
        fallback_method = f"{baseline.method}; {stage_label.lower()} rejected"
        return WarpModel.from_pixel_affine(
            baseline.pixel_affine(),
            fallback_method,
            control_points=baseline.control_points,
            rmse_px=baseline.rmse_px,
            mean_score=baseline.mean_score,
            score_q25=baseline.score_q25,
            min_score=baseline.min_score,
        )

    candidate.method = f"{candidate.method} ({stage_label.lower()} accepted)"
    if logger is not None:
        logger.info(f"{stage_label} accepted")
    return candidate


def guarded_residual_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    candidate: ShiftResult,
    stage_label: str,
    logger: Optional[ConsoleLog] = None,
) -> WarpModel:
    magnitude = math.hypot(float(candidate.row_shift), float(candidate.col_shift))
    identity = WarpModel.from_pixel_affine(Affine.identity(), f"{stage_label.lower()} skipped")
    if magnitude < RESIDUAL_TRANSLATION_MIN_MAGNITUDE_PIXELS:
        if logger is not None:
            logger.info(
                f"{stage_label}: residual {magnitude:.2f}px is below "
                f"{RESIDUAL_TRANSLATION_MIN_MAGNITUDE_PIXELS:.2f}px"
            )
        return identity

    candidate_model = WarpModel.from_translation(candidate, stage_label.lower())
    required_gain = (
        RESIDUAL_TRANSLATION_BASE_MIN_SCORE_GAIN
        + RESIDUAL_TRANSLATION_MIN_SCORE_GAIN_PER_PIXEL * magnitude
    )
    validation = validate_alignment_candidate(
        reference,
        moving,
        Affine.identity(),
        candidate_model.pixel_affine(),
        minimum_score_gain=required_gain if logger is None else None,
    )
    maximum_worsened_fraction = max(
        RESIDUAL_TRANSLATION_MIN_MAX_WORSENED_FRACTION,
        RESIDUAL_TRANSLATION_BASE_MAX_WORSENED_FRACTION
        - RESIDUAL_TRANSLATION_WORSENED_TIGHTEN_PER_PIXEL * magnitude,
    )
    maximum_local_drop = max(
        RESIDUAL_TRANSLATION_MIN_MAX_LOCAL_SCORE_DROP,
        RESIDUAL_TRANSLATION_BASE_MAX_LOCAL_SCORE_DROP
        - RESIDUAL_TRANSLATION_LOCAL_DROP_TIGHTEN_PER_PIXEL * magnitude,
    )
    reasons = []
    if validation.valid_windows < WARP_MIN_VALIDATION_WINDOWS:
        reasons.append(
            f"regional validation windows {validation.valid_windows} < "
            f"{WARP_MIN_VALIDATION_WINDOWS}"
        )
    if not math.isfinite(validation.score_gain) or validation.score_gain < required_gain:
        reasons.append(
            f"global NCC gain {validation.score_gain:+.5f} < required {required_gain:.5f}"
        )
    if validation.worsened_fraction > maximum_worsened_fraction:
        reasons.append(
            f"worsened-window fraction {validation.worsened_fraction:.1%} > "
            f"{maximum_worsened_fraction:.1%}"
        )
    if validation.worst_local_gain < -maximum_local_drop:
        reasons.append(
            f"worst local NCC change {validation.worst_local_gain:+.5f} < "
            f"{-maximum_local_drop:+.5f}"
        )

    if logger is not None:
        logger.info(
            f"{stage_label} confidence: residual={magnitude:.2f}px; "
            f"global gain={validation.score_gain:+.5f}/{required_gain:.5f}; "
            f"local median={validation.median_local_gain:+.5f}; "
            f"worsened={validation.worsened_fraction:.1%}/"
            f"{maximum_worsened_fraction:.1%}; "
            f"worst={validation.worst_local_gain:+.5f}"
        )
    if reasons:
        if logger is not None:
            logger.info(f"{stage_label} rejected: " + "; ".join(reasons))
        return identity
    if logger is not None:
        logger.info(f"{stage_label} accepted")
    candidate_model.method = f"{stage_label.lower()} accepted"
    return candidate_model


def summarize_composite_model(
    pixel_affine: Affine,
    stage_methods: List[str],
    accepted_warps: List[WarpModel],
) -> WarpModel:
    if accepted_warps:
        control_points = sum(model.control_points for model in accepted_warps)
        rmse_px = max(model.rmse_px for model in accepted_warps)
        finite_means = [model.mean_score for model in accepted_warps if math.isfinite(model.mean_score)]
        finite_q25 = [model.score_q25 for model in accepted_warps if math.isfinite(model.score_q25)]
        finite_min = [model.min_score for model in accepted_warps if math.isfinite(model.min_score)]
        mean_score = min(finite_means) if finite_means else math.nan
        score_q25 = min(finite_q25) if finite_q25 else math.nan
        min_score = min(finite_min) if finite_min else math.nan
    else:
        control_points = 0
        rmse_px = 0.0
        mean_score = math.nan
        score_q25 = math.nan
        min_score = math.nan
    return WarpModel.from_pixel_affine(
        pixel_affine,
        " -> ".join(stage_methods),
        control_points=control_points,
        rmse_px=rmse_px,
        mean_score=mean_score,
        score_q25=score_q25,
        min_score=min_score,
    )


def fit_warp_model(control_points: List[LocalShift], method: str) -> WarpModel:
    if len(control_points) < 3:
        raise RuntimeError("At least three local shifts are required to fit a linear warp.")

    # Reference-window centers -> source pixel-corner coordinates.
    a = np.array(
        [[1.0, point.col - point.col_shift + 0.5,
          point.row - point.row_shift + 0.5] for point in control_points],
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
    scores = np.array([point.score for point in control_points], dtype=np.float64)
    mean_score = float(np.mean(scores)) if scores.size else math.nan
    score_q25 = float(np.percentile(scores, 25.0)) if scores.size else math.nan
    min_score = float(np.min(scores)) if scores.size else math.nan

    return WarpModel(
        float(row_coef[0]), float(row_coef[1]), float(row_coef[2]),
        float(col_coef[0]), float(col_coef[1]), float(col_coef[2]),
        len(control_points), rmse, mean_score, method, score_q25, min_score,
    )


def robust_fit_warp_model(control_points: List[LocalShift]) -> WarpModel:
    if len(control_points) < LOCAL_WARP_MIN_CONTROLS:
        raise RuntimeError("Not enough local shifts for a robust linear warp.")

    model = fit_warp_model(control_points, "affine local warp")
    a = np.array(
        [[1.0, point.col - point.col_shift + 0.5,
          point.row - point.row_shift + 0.5] for point in control_points],
        dtype=np.float64,
    )
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


def warp_control_residual(point: LocalShift, model: WarpModel) -> float:
    row_prediction, col_prediction = model.shift_at(
        point.row - point.row_shift + 0.5,
        point.col - point.col_shift + 0.5,
    )
    return math.hypot(
        row_prediction - point.row_shift,
        col_prediction - point.col_shift,
    )


def cloud_warp_plausibility_error(
    model: WarpModel,
    controls: List[LocalShift],
    global_shift: ShiftResult,
    shape: Tuple[int, int],
) -> Optional[str]:
    if not math.isfinite(model.rmse_px) or model.rmse_px > LOCAL_CLOUD_MAX_RMSE_PIXELS:
        return (
            f"fit RMSE {model.rmse_px:.2f}px exceeds "
            f"{LOCAL_CLOUD_MAX_RMSE_PIXELS:.1f}px"
        )

    inliers = [
        point
        for point in controls
        if warp_control_residual(point, model) <= LOCAL_CLOUD_MAX_RMSE_PIXELS
    ]
    if len(inliers) < LOCAL_WARP_MIN_CONTROLS:
        return "too few controls remain within the cloud-tolerant residual gate"
    height, width = shape
    row_span = max(point.row for point in inliers) - min(point.row for point in inliers)
    col_span = max(point.col for point in inliers) - min(point.col for point in inliers)
    if row_span < LOCAL_CLOUD_MIN_SPAN_FRACTION * max(1, height - 1):
        return "reliable controls do not span enough rows"
    if col_span < LOCAL_CLOUD_MIN_SPAN_FRACTION * max(1, width - 1):
        return "reliable controls do not span enough columns"

    corners = (
        (0.0, 0.0),
        (0.0, float(max(0, width - 1))),
        (float(max(0, height - 1)), 0.0),
        (float(max(0, height - 1)), float(max(0, width - 1))),
    )
    maximum_corner_deviation = 0.0
    for row, col in corners:
        row_shift, col_shift = model.shift_at(row, col)
        maximum_corner_deviation = max(
            maximum_corner_deviation,
            math.hypot(
                row_shift - float(global_shift.row_shift),
                col_shift - float(global_shift.col_shift),
            ),
        )
    if maximum_corner_deviation > LOCAL_CLOUD_MAX_CORNER_DEVIATION_PIXELS:
        return (
            f"corner deviation {maximum_corner_deviation:.2f}px exceeds "
            f"{LOCAL_CLOUD_MAX_CORNER_DEVIATION_PIXELS:.1f}px"
        )
    return None


def match_linear_warp_window(
    reference: np.ndarray,
    prealigned_moving: np.ndarray,
    window: Tuple[int, int, int, int],
    base_row_shift: int,
    base_col_shift: int,
    local_max_shift_pixels: int,
    residual_search_radius: int,
    cloud_tolerant: bool,
) -> Tuple[Optional[LocalShift], str]:
    wy0, wy1, wx0, wx1 = window
    ref_win = reference[wy0:wy1, wx0:wx1]
    mov_win = prealigned_moving[wy0:wy1, wx0:wx1]
    try:
        if cloud_tolerant:
            ref_valid = np.isfinite(ref_win)
            mov_valid = np.isfinite(mov_win)
            common = ref_valid & mov_valid
            common_count = int(np.count_nonzero(common))
            minimum_count = max(
                LOCAL_CLOUD_MIN_VALID_PIXELS,
                int(math.ceil(ref_win.size * LOCAL_CLOUD_MIN_VALID_FRACTION)),
            )
            available = min(
                int(np.count_nonzero(ref_valid)),
                int(np.count_nonzero(mov_valid)),
            )
            overlap_fraction = common_count / max(1, available)
            if (
                common_count < minimum_count
                or overlap_fraction < LOCAL_CLOUD_MIN_OVERLAP_FRACTION
            ):
                return None, "support"
            ref_values = ref_win[common].astype(np.float64)
            mov_values = mov_win[common].astype(np.float64)
            if ref_values.std() <= 1e-12 or mov_values.std() <= 1e-12:
                return None, "support"
            local_best = masked_zncc_fft_peak(
                ref_win,
                mov_win,
                residual_search_radius,
            )
            refined = subpixel_refine_with_ncc(
                ref_win,
                mov_win,
                int(round(local_best.row_shift)),
                int(round(local_best.col_shift)),
                residual_search_radius,
            )
            if (
                refined.valid_pixels < minimum_count
                or refined.score < LOCAL_CLOUD_MIN_SCORE
            ):
                return None, "score"
        else:
            phase = masked_phase_correlation_peak(
                ref_win,
                mov_win,
                local_max_shift_pixels,
                center_row=0,
                center_col=0,
                search_radius=residual_search_radius,
            )
            local_match = search_level_match(
                PyramidLevel(1, ref_win, mov_win),
                int(round(phase.row_shift)),
                int(round(phase.col_shift)),
                1,
                local_max_shift_pixels,
            )
            local_best = local_match.result
            refined = subpixel_refine_with_ncc(
                ref_win,
                mov_win,
                int(round(local_best.row_shift)),
                int(round(local_best.col_shift)),
                local_max_shift_pixels,
                cached_scores=local_match.ncc_scores,
            )
    except Exception:
        return None, "match"

    if not math.isfinite(refined.score):
        return None, ""
    return (
        LocalShift(
            row=0.5 * (wy0 + wy1 - 1),
            col=0.5 * (wx0 + wx1 - 1),
            row_shift=base_row_shift + refined.row_shift,
            col_shift=base_col_shift + refined.col_shift,
            score=refined.score,
            valid_pixels=refined.valid_pixels,
        ),
        "",
    )


def estimate_linear_warp(
    reference: np.ndarray,
    moving: np.ndarray,
    global_shift: ShiftResult,
    local_max_shift_pixels: int,
    logger: Optional[ConsoleLog] = None,
    cloud_tolerant: bool = False,
    residual_search_radius: int = LOCAL_WARP_RESIDUAL_RADIUS,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> WarpModel:
    residual_search_radius = max(1, int(residual_search_radius))
    base_row_shift = int(round(float(global_shift.row_shift)))
    base_col_shift = int(round(float(global_shift.col_shift)))
    prealigned_moving = shift_array_nearest(moving, base_row_shift, base_col_shift)
    windows = local_window_slices(reference, prealigned_moving, local_max_shift_pixels)
    controls = []
    rejected_support = 0
    rejected_score = 0
    rejected_match = 0

    def run_window(window):
        return match_linear_warp_window(
            reference,
            prealigned_moving,
            window,
            base_row_shift,
            base_col_shift,
            local_max_shift_pixels,
            residual_search_radius,
            cloud_tolerant,
        )

    parallel_windows = (
        ACTIVE_LOCAL_WINDOW_WORKERS > 1
        and len(windows) > 1
        and (cloud_tolerant or gpu_is_active())
    )
    if parallel_windows and local_executor is not None:
        window_results = list(local_executor.map(run_window, windows))
    elif parallel_windows:
        with ThreadPoolExecutor(
            max_workers=min(ACTIVE_LOCAL_WINDOW_WORKERS, len(windows)),
            thread_name_prefix="tir_bridge_local",
            initializer=configure_local_cpu_worker,
        ) as executor:
            window_results = list(executor.map(run_window, windows))
    else:
        window_results = [run_window(window) for window in windows]

    for control, rejection in window_results:
        if control is not None:
            controls.append(control)
        elif rejection == "support":
            rejected_support += 1
        elif rejection == "score":
            rejected_score += 1
        elif rejection == "match":
            rejected_match += 1

    if logger is not None:
        logger.info(
            f"Linear warp controls: {len(controls)}/{len(windows)} local windows "
            f"(residual search +/-{residual_search_radius}px; "
            f"matcher={'exact masked ZNCC' if cloud_tolerant else 'legacy phase/NCC'})"
        )
        if cloud_tolerant:
            logger.info(
                f"Cloud local rejects: support={rejected_support}; "
                f"score={rejected_score}; match={rejected_match}"
            )

    if len(controls) < LOCAL_WARP_MIN_CONTROLS:
        if logger is not None:
            logger.info("Linear warp fallback: not enough reliable local controls")
        method = (
            "cloud-tolerant translation fallback"
            if cloud_tolerant
            else "translation fallback"
        )
        return WarpModel.from_translation(global_shift, method)

    try:
        model = robust_fit_warp_model(controls)
    except Exception as exc:
        if logger is not None:
            logger.info(f"Linear warp fallback: {exc}")
        method = (
            "cloud-tolerant translation fallback"
            if cloud_tolerant
            else "translation fallback"
        )
        return WarpModel.from_translation(global_shift, method)

    if cloud_tolerant:
        plausibility_error = cloud_warp_plausibility_error(
            model,
            controls,
            global_shift,
            reference.shape,
        )
        if plausibility_error is not None:
            if logger is not None:
                logger.info(f"Linear warp fallback: {plausibility_error}")
            return WarpModel.from_translation(
                global_shift,
                "cloud-tolerant translation fallback",
            )
        model.method = f"cloud-tolerant {model.method}"

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
    cached_scores: Optional[Dict[Tuple[int, int], Tuple[float, int]]] = None,
) -> ShiftResult:
    candidate_shifts = [
        (row_shift, col_shift),
        (row_shift - 1, col_shift),
        (row_shift + 1, col_shift),
        (row_shift, col_shift - 1),
        (row_shift, col_shift + 1),
    ]
    valid_positions = [
        abs(row) <= max_shift_pixels and abs(col) <= max_shift_pixels
        for row, col in candidate_shifts
    ]
    scores = [-math.inf] * len(candidate_shifts)
    counts = [0] * len(candidate_shifts)
    missing_indices = []
    missing_shifts = []
    for index, (shift, use) in enumerate(zip(candidate_shifts, valid_positions)):
        if not use:
            continue
        cached = cached_scores.get(shift) if cached_scores is not None else None
        if cached is not None:
            scores[index] = float(cached[0])
            counts[index] = int(cached[1])
        else:
            missing_indices.append(index)
            missing_shifts.append(shift)

    if missing_shifts:
        min_valid = min_valid_pixels_for_level(reference, moving)
        computed_scores, computed_counts = normalized_correlations_at_shifts(
            reference,
            moving,
            missing_shifts,
            min_valid,
        )
        for index, score, count in zip(
            missing_indices,
            computed_scores,
            computed_counts,
        ):
            scores[index] = float(score)
            counts[index] = int(count)

    center_score = scores[0]
    center_count = counts[0]
    if not math.isfinite(center_score):
        raise RuntimeError("Could not compute normalized correlation at the phase-correlation peak.")

    row_offset = parabolic_subpixel_offset(scores[1], center_score, scores[2])
    col_offset = parabolic_subpixel_offset(scores[3], center_score, scores[4])
    refined_row = max(-max_shift_pixels, min(max_shift_pixels, float(row_shift) + row_offset))
    refined_col = max(-max_shift_pixels, min(max_shift_pixels, float(col_shift) + col_offset))
    return ShiftResult(refined_row, refined_col, center_score, center_count)


def adaptive_peak_guard_reasons(
    match: PeakDiagnostics,
    require_phase_confidence: bool,
    require_ncc_score: bool = False,
) -> List[str]:
    reasons = []
    result = match.result
    if not math.isfinite(result.score):
        reasons.append("non-finite peak score")
    if match.search_edge_hit:
        reasons.append("search-edge peak")
    if (
        not math.isfinite(match.overlap_fraction)
        or match.overlap_fraction < ADAPTIVE_MIN_OVERLAP_FRACTION
    ):
        reasons.append(
            f"overlap {match.overlap_fraction:.3f} < "
            f"{ADAPTIVE_MIN_OVERLAP_FRACTION:.3f}"
        )
    if require_phase_confidence:
        relative_margin = (
            match.peak_margin / max(abs(float(result.score)), 1e-06)
            if math.isfinite(match.peak_margin)
            else math.nan
        )
        if (
            not math.isfinite(relative_margin)
            or relative_margin < ADAPTIVE_PHASE_MIN_RELATIVE_MARGIN
        ):
            reasons.append(
                f"relative peak margin {relative_margin:.4f} < "
                f"{ADAPTIVE_PHASE_MIN_RELATIVE_MARGIN:.4f}"
            )
        if (
            not math.isfinite(match.peak_zscore)
            or match.peak_zscore < ADAPTIVE_PHASE_MIN_PEAK_ZSCORE
        ):
            reasons.append(
                f"peak z-score {match.peak_zscore:.2f} < "
                f"{ADAPTIVE_PHASE_MIN_PEAK_ZSCORE:.2f}"
            )
    if require_ncc_score and (
        not math.isfinite(result.score)
        or result.score < ADAPTIVE_MIN_NATIVE_NCC_SCORE
    ):
        reasons.append(
            f"NCC {result.score:.4f} < {ADAPTIVE_MIN_NATIVE_NCC_SCORE:.4f}"
        )
    return reasons


def native_phase_translation_refine(
    reference: np.ndarray,
    moving: np.ndarray,
    global_max_shift_pixels: int,
    center_row: int,
    center_col: int,
    search_radius: int,
    method: str,
    logger: Optional[ConsoleLog] = None,
    describe_shift=None,
    preparation_cache: Optional[RegistrationPreparationCache] = None,
) -> ShiftResult:
    # A retry sometimes requests exactly the full native search already used by
    # the initial pass. Reuse only results for this scene's immutable prepared
    # arrays, with identical search settings and no intervening GPU fallback.
    cache_key = None
    prepared = preparation_cache.prepared if preparation_cache is not None else None
    fallback_generation = GPU_FALLBACK_GENERATION
    if (
        logger is None
        and prepared is not None
        and reference is prepared.legacy_reference
        and moving is prepared.legacy_moving
    ):
        cache_key = (
            id(reference), id(moving), int(global_max_shift_pixels),
            int(center_row), int(center_col), int(search_radius),
            int(ADAPTIVE_PYRAMID_REFINE_RADIUS), gpu_is_active(),
            fallback_generation,
        )
        cached = preparation_cache.native_phase_results.get(cache_key)
        if cached is not None:
            return ShiftResult(
                cached.row_shift, cached.col_shift, cached.score,
                cached.valid_pixels, method=method,
            )
    phase_peak = masked_phase_correlation_peak(
        reference,
        moving,
        global_max_shift_pixels,
        center_row=int(center_row),
        center_col=int(center_col),
        search_radius=int(search_radius),
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

    integer_match = search_level_match(
        PyramidLevel(1, reference, moving),
        int(round(phase_peak.row_shift)),
        int(round(phase_peak.col_shift)),
        ADAPTIVE_PYRAMID_REFINE_RADIUS,
        global_max_shift_pixels,
    )
    integer_best = integer_match.result
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
        cached_scores=integer_match.ncc_scores,
    )
    refined.method = method
    if cache_key is not None and fallback_generation == GPU_FALLBACK_GENERATION:
        preparation_cache.native_phase_results[cache_key] = ShiftResult(
            refined.row_shift, refined.col_shift, refined.score,
            refined.valid_pixels, method=method,
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


def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    global_max_shift_pixels: int,
    native_refine_radius_pixels: int,
    coarse_downsample: int,
    logger: Optional[ConsoleLog] = None,
    describe_shift=None,
    force_full_native: bool = False,
    preparation_cache: Optional[RegistrationPreparationCache] = None,
) -> ShiftResult:
    if reference.shape != moving.shape:
        raise RuntimeError("Reference and moving arrays must already be on the same grid.")

    global_max_shift_pixels = max(0, int(global_max_shift_pixels))
    native_refine_radius_pixels = max(1, int(native_refine_radius_pixels))
    coarse_downsample = max(1, int(coarse_downsample))

    # Retry 01 intentionally remains a full-range native search.
    if coarse_downsample == 1:
        return native_phase_translation_refine(
            reference,
            moving,
            global_max_shift_pixels,
            0,
            0,
            min(global_max_shift_pixels, native_refine_radius_pixels),
            "full-range native masked phase",
            logger=logger,
            describe_shift=describe_shift,
            preparation_cache=preparation_cache,
        )

    if force_full_native:
        return native_phase_translation_refine(
            reference,
            moving,
            global_max_shift_pixels,
            0,
            0,
            global_max_shift_pixels,
            "full-range native confirmation after adaptive/robust disagreement",
            logger=logger,
            describe_shift=describe_shift,
            preparation_cache=preparation_cache,
        )

    levels = build_requested_pyramid_levels(
        reference,
        moving,
        coarse_downsample,
    )
    coarse_level = levels[coarse_downsample]
    coarse_max = int(math.ceil(global_max_shift_pixels / coarse_downsample))
    try:
        coarse_match = masked_phase_correlation_match(
            coarse_level.reference,
            coarse_level.moving,
            coarse_max,
        )
        coarse_peak = coarse_match.result
    except Exception:
        if logger is not None:
            logger.info(
                f"Coarse {coarse_downsample}x catch failed; using full-range native phase"
            )
        return native_phase_translation_refine(
            reference,
            moving,
            global_max_shift_pixels,
            0,
            0,
            global_max_shift_pixels,
            "full-range native phase after coarse failure",
            logger=logger,
            describe_shift=describe_shift,
            preparation_cache=preparation_cache,
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

    def native_fallback(reason_text: str) -> ShiftResult:
        if logger is not None:
            logger.info(f"Adaptive refinement fallback: {reason_text}")
        return native_phase_translation_refine(
            reference,
            moving,
            global_max_shift_pixels,
            0,
            0,
            global_max_shift_pixels,
            f"guarded full-range native-FFT fallback: {reason_text}",
            logger=logger,
            describe_shift=describe_shift,
            preparation_cache=preparation_cache,
        )

    if not ENABLE_ADAPTIVE_PHASE_REFINEMENT:
        return native_phase_translation_refine(
            reference,
            moving,
            global_max_shift_pixels,
            int(round(coarse_row_native)),
            int(round(coarse_col_native)),
            min(global_max_shift_pixels, native_refine_radius_pixels),
            "legacy coarse-centered native masked phase",
            logger=logger,
            describe_shift=describe_shift,
            preparation_cache=preparation_cache,
        )

    coarse_reasons = adaptive_peak_guard_reasons(
        coarse_match,
        require_phase_confidence=True,
    )
    if coarse_reasons:
        return native_fallback(", ".join(coarse_reasons))

    try:
        # Normal path: 16x global phase -> 4x local phase -> 2x/native NCC.
        if coarse_downsample > ADAPTIVE_PHASE_DOWNSAMPLE:
            phase_level = levels[ADAPTIVE_PHASE_DOWNSAMPLE]
            phase_max = int(
                math.ceil(global_max_shift_pixels / ADAPTIVE_PHASE_DOWNSAMPLE)
            )
            phase_radius = int(
                math.ceil(native_refine_radius_pixels / ADAPTIVE_PHASE_DOWNSAMPLE)
            )
            phase_match = masked_phase_correlation_match(
                phase_level.reference,
                phase_level.moving,
                phase_max,
                center_row=int(round(coarse_row_native / ADAPTIVE_PHASE_DOWNSAMPLE)),
                center_col=int(round(coarse_col_native / ADAPTIVE_PHASE_DOWNSAMPLE)),
                search_radius=phase_radius,
            )
            phase_reasons = adaptive_peak_guard_reasons(
                phase_match,
                require_phase_confidence=True,
            )
            if phase_reasons:
                return native_fallback(", ".join(phase_reasons))

            factor2_level = levels[ADAPTIVE_NCC_DOWNSAMPLE]
            phase_to_ncc_scale = (
                ADAPTIVE_PHASE_DOWNSAMPLE / ADAPTIVE_NCC_DOWNSAMPLE
            )
            factor2_match = search_level_match(
                factor2_level,
                int(round(phase_match.result.row_shift * phase_to_ncc_scale)),
                int(round(phase_match.result.col_shift * phase_to_ncc_scale)),
                ADAPTIVE_PYRAMID_REFINE_RADIUS,
                global_max_shift_pixels,
            )
            factor2_reasons = adaptive_peak_guard_reasons(
                factor2_match,
                require_phase_confidence=False,
                require_ncc_score=True,
            )
            if factor2_reasons:
                return native_fallback(", ".join(factor2_reasons))
            native_center_row = int(
                round(factor2_match.result.row_shift * ADAPTIVE_NCC_DOWNSAMPLE)
            )
            native_center_col = int(
                round(factor2_match.result.col_shift * ADAPTIVE_NCC_DOWNSAMPLE)
            )
            method = (
                f"adaptive {coarse_downsample}x/{ADAPTIVE_PHASE_DOWNSAMPLE}x/"
                "2x/native phase-NCC"
            )

        # Post-warp path: its 2x global phase already covers the full residual range.
        elif (
            coarse_downsample == ADAPTIVE_NCC_DOWNSAMPLE
            and global_max_shift_pixels <= ADAPTIVE_POST_WARP_MAX_SHIFT_PIXELS
        ):
            native_center_row = int(
                round(coarse_peak.row_shift * ADAPTIVE_NCC_DOWNSAMPLE)
            )
            native_center_col = int(
                round(coarse_peak.col_shift * ADAPTIVE_NCC_DOWNSAMPLE)
            )
            method = "adaptive 2x/native residual phase-NCC"
        else:
            return native_fallback("unsupported adaptive pyramid configuration")

        native_match = search_level_match(
            levels[1],
            native_center_row,
            native_center_col,
            ADAPTIVE_PYRAMID_REFINE_RADIUS,
            global_max_shift_pixels,
        )
        native_reasons = adaptive_peak_guard_reasons(
            native_match,
            require_phase_confidence=False,
            require_ncc_score=True,
        )
        if native_reasons:
            return native_fallback(", ".join(native_reasons))

        refined = subpixel_refine_with_ncc(
            reference,
            moving,
            int(round(native_match.result.row_shift)),
            int(round(native_match.result.col_shift)),
            global_max_shift_pixels,
            cached_scores=native_match.ncc_scores,
        )
        refined.method = method
        if logger is not None:
            shift_text = (
                describe_shift(refined.row_shift, refined.col_shift)
                if describe_shift is not None
                else f"row={refined.row_shift:+.2f}, col={refined.col_shift:+.2f}"
            )
            logger.info(
                f"Adaptive native NCC/sub-pixel fit: {shift_text}; "
                f"score={refined.score:.5f}; valid={refined.valid_pixels:,}"
            )
        return refined
    except Exception as exc:
        return native_fallback(str(exc))


def score_shift_on_features(
    reference: np.ndarray,
    moving: np.ndarray,
    shift: ShiftResult,
) -> Tuple[float, int]:
    min_valid = min_valid_pixels_for_level(reference, moving)
    return normalized_correlation(
        reference,
        moving,
        int(round(float(shift.row_shift))),
        int(round(float(shift.col_shift))),
        min_valid,
    )


def estimate_translation_cloud_robust(
    reference: np.ndarray,
    moving: np.ndarray,
    global_max_shift_pixels: int,
    logger: Optional[ConsoleLog] = None,
    describe_shift=None,
) -> ShiftResult:
    """Cloud-tolerant global match: exact masked ZNCC then direct NCC refinement."""
    factor = ROBUST_GLOBAL_DOWNSAMPLE
    reference_levels = cached_reference_pyramid_levels(reference, max(2, factor))
    factor2_reference = (
        reference_levels[2] if reference_levels is not None
        else nanmean_downsample_2x(reference)
    )
    factor2_moving = nanmean_downsample_2x(moving)
    if factor <= 1:
        coarse_reference = reference
        coarse_moving = moving
    elif factor == 2:
        coarse_reference = factor2_reference
        coarse_moving = factor2_moving
    else:
        coarse_reference = (
            reference_levels[factor]
            if reference_levels is not None and factor in reference_levels
            else nanmean_downsample_power_of_two(factor2_reference, factor // 2)
        )
        coarse_moving = nanmean_downsample_power_of_two(
            factor2_moving, factor // 2
        )
    coarse_max = int(math.ceil(global_max_shift_pixels / factor))
    coarse = masked_zncc_fft_peak(
        coarse_reference,
        coarse_moving,
        coarse_max,
    )
    coarse_row_native = coarse.row_shift * factor
    coarse_col_native = coarse.col_shift * factor
    if logger is not None:
        shift_text = (
            describe_shift(coarse_row_native, coarse_col_native)
            if describe_shift is not None
            else f"row={coarse_row_native:+.0f}, col={coarse_col_native:+.0f}"
        )
        logger.info(
            f"Cloud robust {factor}x exact masked ZNCC: {shift_text}; "
            f"score={coarse.score:.5f}; valid={coarse.valid_pixels:,}"
        )

    factor2 = search_level(
        PyramidLevel(2, factor2_reference, factor2_moving),
        int(round(coarse.row_shift * (factor / 2))),
        int(round(coarse.col_shift * (factor / 2))),
        ROBUST_PYRAMID_REFINE_RADIUS,
        global_max_shift_pixels,
    )
    native_cpu_only = not gpu_is_active()
    native_match = search_level_match(
        PyramidLevel(1, reference, moving),
        int(round(factor2.row_shift * 2)),
        int(round(factor2.col_shift * 2)),
        ROBUST_PYRAMID_REFINE_RADIUS,
        global_max_shift_pixels,
    )
    native = native_match.result
    refined = subpixel_refine_with_ncc(
        reference,
        moving,
        int(round(native.row_shift)),
        int(round(native.col_shift)),
        global_max_shift_pixels,
        # CPU searches and refinement use the same scalar NCC reduction. GPU
        # batch dispatch depends on candidate count, so preserve its existing
        # independent refinement calculation and numerical backend choice.
        cached_scores=(
            native_match.ncc_scores
            if native_cpu_only and not gpu_is_active()
            else None
        ),
    )
    if logger is not None:
        shift_text = (
            describe_shift(refined.row_shift, refined.col_shift)
            if describe_shift is not None
            else f"row={refined.row_shift:+.2f}, col={refined.col_shift:+.2f}"
        )
        logger.info(
            f"Cloud robust factor-2/native masked NCC: {shift_text}; "
            f"score={refined.score:.5f}; valid={refined.valid_pixels:,}"
        )
    return refined


def estimate_translation_hybrid(
    legacy_reference: np.ndarray,
    legacy_moving: np.ndarray,
    robust_reference: np.ndarray,
    robust_moving: np.ndarray,
    support_loss: float,
    global_max_shift_pixels: int,
    native_refine_radius_pixels: int,
    coarse_downsample: int,
    logger: Optional[ConsoleLog] = None,
    describe_shift=None,
    force_robust: bool = False,
    robust_result_cache: Optional[RegistrationPreparationCache] = None,
) -> TranslationChoice:
    legacy = None
    robust = None
    legacy_error = None
    robust_error = None

    def robust_feature_score() -> Tuple[float, int]:
        if (
            robust_result_cache is not None
            and robust_result_cache.robust_global_result is robust
            and math.isfinite(robust_result_cache.robust_global_feature_score)
        ):
            return (
                float(robust_result_cache.robust_global_feature_score),
                int(robust_result_cache.robust_global_feature_count),
            )
        score, count = score_shift_on_features(
            robust_reference, robust_moving, robust
        )
        if robust_result_cache is not None:
            robust_result_cache.robust_global_feature_score = float(score)
            robust_result_cache.robust_global_feature_count = int(count)
        return float(score), int(count)

    def legacy_method_label(suffix: str = "") -> str:
        base = (
            legacy.method
            if legacy is not None and legacy.method
            else "legacy masked phase"
        )
        return f"{base} ({suffix})" if suffix else base

    try:
        legacy = estimate_translation(
            legacy_reference,
            legacy_moving,
            global_max_shift_pixels,
            native_refine_radius_pixels,
            coarse_downsample,
            logger=logger,
            describe_shift=describe_shift,
            preparation_cache=robust_result_cache,
        )
    except Exception as exc:
        legacy_error = exc
        if logger is not None:
            logger.info(f"Legacy phase candidate unavailable: {exc}")

    # Under the existing selector, a successful legacy result cannot be replaced
    # when support loss is below the robust-evidence threshold. Avoid the costly
    # full masked-ZNCC candidate in that deterministic branch.
    if (
        legacy is not None
        and support_loss < ROBUST_SUPPORT_LOSS_TRIGGER
        and not force_robust
    ):
        if logger is not None:
            logger.info(
                "Global selector: legacy result; robust ZNCC skipped because "
                f"support loss {support_loss:.3f} < {ROBUST_SUPPORT_LOSS_TRIGGER:.3f}"
            )
        return TranslationChoice(
            legacy,
            False,
            legacy_method_label("robust skipped: low support loss"),
            float(legacy.score),
            math.nan,
        )

    try:
        cached_robust = (
            robust_result_cache.robust_global_result
            if robust_result_cache is not None
            and robust_result_cache.robust_global_max_shift_pixels
            == int(global_max_shift_pixels)
            else None
        )
        if cached_robust is None:
            robust = estimate_translation_cloud_robust(
                robust_reference,
                robust_moving,
                global_max_shift_pixels,
                logger=logger,
                describe_shift=describe_shift,
            )
            if robust_result_cache is not None:
                robust_result_cache.robust_global_result = robust
                robust_result_cache.robust_global_max_shift_pixels = int(
                    global_max_shift_pixels
                )
                robust_result_cache.robust_global_feature_score = math.nan
                robust_result_cache.robust_global_feature_count = 0
        else:
            robust = cached_robust
            if logger is not None:
                logger.info("Cloud-robust global candidate reused from this scene.")
    except Exception as exc:
        robust_error = exc
        if logger is not None:
            logger.info(f"Cloud-robust candidate unavailable: {exc}")

    if legacy is None and robust is None:
        raise RuntimeError(
            f"Both global matchers failed. Legacy: {legacy_error}; cloud robust: {robust_error}"
        )
    if legacy is None:
        robust_score, _ = robust_feature_score()
        if logger is not None:
            logger.info("Global selector: cloud-robust candidate (legacy matcher failed)")
        return TranslationChoice(
            robust, True, "cloud-tolerant masked ZNCC", math.nan, robust_score
        )
    if robust is None:
        legacy_score, _ = score_shift_on_features(robust_reference, robust_moving, legacy)
        if logger is not None:
            logger.info("Global selector: legacy phase candidate (robust matcher failed)")
        return TranslationChoice(
            legacy, False, legacy_method_label(), legacy_score, math.nan
        )

    initial_separation = math.hypot(
        float(legacy.row_shift) - float(robust.row_shift),
        float(legacy.col_shift) - float(robust.col_shift),
    )
    if (
        legacy.method.startswith("adaptive ")
        and initial_separation > ROBUST_AGREEMENT_TOLERANCE_PIXELS
    ):
        if logger is not None:
            logger.info(
                f"Adaptive/robust separation {initial_separation:.2f}px exceeds "
                f"{ROBUST_AGREEMENT_TOLERANCE_PIXELS:.2f}px; confirming with native FFT"
            )
        try:
            legacy = estimate_translation(
                legacy_reference,
                legacy_moving,
                global_max_shift_pixels,
                native_refine_radius_pixels,
                coarse_downsample,
                logger=logger,
                describe_shift=describe_shift,
                force_full_native=True,
                preparation_cache=robust_result_cache,
            )
        except Exception as exc:
            if logger is not None:
                logger.info(f"Native confirmation unavailable: {exc}")

    legacy_score, legacy_count = score_shift_on_features(
        robust_reference, robust_moving, legacy
    )
    robust_score, robust_count = robust_feature_score()
    distance = math.hypot(
        float(legacy.row_shift) - float(robust.row_shift),
        float(legacy.col_shift) - float(robust.col_shift),
    )
    advantage = robust_score - legacy_score
    missing_evidence = (
        support_loss >= ROBUST_SUPPORT_LOSS_TRIGGER
        or force_robust
    )
    if logger is not None:
        logger.info(
            f"Global candidate check: separation={distance:.2f}px; "
            f"legacy robust-feature NCC={legacy_score:.5f} ({legacy_count:,} px); "
            f"cloud robust NCC={robust_score:.5f} ({robust_count:,} px); "
            f"advantage={advantage:+.5f}"
        )

    if distance <= ROBUST_AGREEMENT_TOLERANCE_PIXELS:
        if logger is not None:
            logger.info("Global selector: candidates agree; retaining exact legacy result")
        return TranslationChoice(
            legacy,
            False,
            legacy_method_label("robust agreement"),
            legacy_score,
            robust_score,
        )
    if (
        math.isfinite(robust_score)
        and advantage >= ROBUST_SCORE_ADVANTAGE
        and missing_evidence
    ):
        if logger is not None:
            logger.info(
                f"Global selector: cloud-robust candidate (internal support loss; "
                f"NCC advantage >= {ROBUST_SCORE_ADVANTAGE:.2f})"
            )
        return TranslationChoice(
            robust, True, "cloud-tolerant masked ZNCC", legacy_score, robust_score
        )

    if logger is not None:
        logger.info(
            "Global selector: retaining legacy result; robust disagreement lacked "
            "the required score/evidence margin"
        )
    return TranslationChoice(
        legacy,
        False,
        legacy_method_label("guarded"),
        legacy_score,
        robust_score,
    )


def shifted_transform(transform: Affine, row_shift: float, col_shift: float) -> Affine:
    return transform * Affine.translation(col_shift, row_shift)


def load_raster_geometry(path: Path) -> RasterGeometry:
    with rasterio.open(path) as src:
        crs = require_crs(src, path)
        return RasterGeometry(src.transform, crs, src.width, src.height)


def apply_translation_like_geoviewer(eco_path: Path, output_path: Path, row_shift: float, col_shift: float):
    with rasterio.open(eco_path) as src:
        require_crs(src, eco_path)
        source_data = raster_band_to_nan(src)
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
            num_threads=ACTIVE_GDAL_WARP_THREADS,
            warp_mem_limit=GDAL_WARP_MEMORY_MB,
        )
        destination[~np.isfinite(destination) | (destination == 0)] = np.nan
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = src.tags(1)

    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        zlevel=TIFF_DEFLATE_LEVEL,
        NUM_THREADS=str(ACTIVE_TIFF_THREADS),
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
        source_data = raster_band_to_nan(src)
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
            num_threads=ACTIVE_GDAL_WARP_THREADS,
            warp_mem_limit=GDAL_WARP_MEMORY_MB,
        )
        destination[~np.isfinite(destination) | (destination == 0)] = np.nan
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = src.tags(1)

    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        zlevel=TIFF_DEFLATE_LEVEL,
        NUM_THREADS=str(ACTIVE_TIFF_THREADS),
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


@lru_cache(maxsize=64)
def native_crs_measurement(crs_wkt: str):
    """Native units and distance basis; no coordinate transformation."""
    crs = PyprojCRS.from_wkt(crs_wkt)
    axes = crs.axis_info[:2]
    if len(axes) != 2:
        raise ValueError("Source CRS must define two horizontal axes")
    factor = float(axes[0].unit_conversion_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Source CRS has no valid horizontal unit conversion")
    if not math.isclose(factor, float(axes[1].unit_conversion_factor), rel_tol=1e-12):
        raise ValueError("Source CRS horizontal axes use different units")
    if crs.is_projected:
        return "projected_grid", "grid_north", axes[0].unit_name, factor, None
    if crs.is_geographic:
        return "native_ellipsoid", "true_north", axes[0].unit_name, factor, crs.get_geod()
    raise ValueError("Source CRS must be projected or geographic")


def native_displacement_metrics(crs, x0, y0, x1, y1):
    """Measure coordinate differences within the original CRS."""
    basis, north, units, factor, geod = native_crs_measurement(crs.to_wkt())
    x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("Source-center coordinates must be finite")
    dx, dy = x1 - x0, y1 - y0
    if geod is None:
        dx_m, dy_m = dx * factor, dy * factor
        distance = math.hypot(dx_m, dy_m)
        azimuth = math.degrees(math.atan2(dx_m, dy_m))
    else:
        azimuth, _back, distance = geod.inv(
            x0 * factor, y0 * factor, x1 * factor, y1 * factor, radians=True
        )
        azimuth = math.degrees(azimuth)
    if not math.isfinite(azimuth) or not math.isfinite(distance) or distance < 0.0:
        raise ValueError("Source coordinates do not yield finite movement metrics")
    azimuth = float(azimuth) % 360.0
    return {
        "center_x_before": x0,
        "center_y_before": y0,
        "center_x_after": x1,
        "center_y_after": y1,
        "x_shift_map_units": dx,
        "y_shift_map_units": dy,
        "azimuth_deg": 0.0 if distance == 0.0 or azimuth >= 360.0 else azimuth,
        "distance_m": float(distance),
        "translation_distance_basis": basis,
        "translation_azimuth_reference": north,
        "source_xy_units": units,
        "source_units_to_m": factor if geod is None else "",
    }


def translation_metrics_from_geometry(geometry: RasterGeometry, row_shift: float, col_shift: float):
    original_transform = geometry.transform
    moved_transform = shifted_transform(geometry.transform, row_shift, col_shift)
    x0, y0 = raster_center(original_transform, geometry.width, geometry.height)
    x1, y1 = raster_center(moved_transform, geometry.width, geometry.height)

    return native_displacement_metrics(geometry.crs, x0, y0, x1, y1)


def translation_metrics(eco_path: Path, row_shift: float, col_shift: float):
    return translation_metrics_from_geometry(load_raster_geometry(eco_path), row_shift, col_shift)


def format_shift_report(row_shift: float, col_shift: float, movement_info: dict) -> str:
    return (
        f"row={float(row_shift):+.2f}px col={float(col_shift):+.2f}px | "
        f"{movement_info['distance_m']:.2f} m @ {movement_info['azimuth_deg']:.3f} deg"
    )




# ---------------------- INTEGRATED ASSESSMENT SETTINGS ----------------------

QA_GRID_ROWS = 5
QA_GRID_COLS = 5
QA_WINDOW_SIZE_PIXELS = 384
QA_CANDIDATES_PER_CELL_AXIS = 3
MAX_RESIDUAL_SEARCH_PIXELS = 8
PAIRED_RESIDUAL_SEARCH_RADIUS_PIXELS = 16
PAIRED_RESIDUAL_MIN_WINDOWS = 1
STRUCTURE_SMOOTH_RADIUS_PIXELS = 3
STRUCTURE_SUPPRESS_PERCENTILE = 55.0
STRUCTURE_CLIP_PERCENTILE = 99.0

MIN_WINDOW_COMMON_VALID_FRACTION = 0.25
MIN_MATCH_PIXELS = 1500
MIN_MATCH_OVERLAP_RATIO = 0.70
MIN_RELIABLE_MATCH_SCORE = 0.10
MIN_RELIABLE_PEAK_MARGIN = 0.010
MIN_RELIABLE_PEAK_ZSCORE = 1.8
MIN_RELIABLE_WINDOWS = 6
MIN_RELIABLE_WINDOW_FRACTION = 0.25
NEAR_ZERO_RESIDUAL_PIXELS = 0.75

@dataclass(frozen=True)
class QAThresholdProfile:
    name: str
    min_window_common_valid_fraction: float
    min_match_pixels: int
    min_match_overlap_ratio: float
    min_reliable_match_score: float
    min_reliable_peak_margin: float
    min_reliable_peak_zscore: float

@dataclass
class QAWindowMatch:
    grid_row: int
    grid_col: int
    row_shift_px: float
    col_shift_px: float
    residual_px: float
    best_score: float
    zero_score: float
    score_gain: float
    second_best_score: float
    peak_margin: float
    peak_zscore: float
    peak_curvature: float
    overlap_pixels: int
    overlap_fraction: float
    search_edge_hit: bool
    subpixel_method: str
    reliable: bool
    rejection_reason: str

@dataclass(frozen=True)
class QAWindowCandidate:
    bounds: Tuple[int, int, int, int]
    common_fraction: float
    texture_score: float


@dataclass
class QAStructuralFeatureVariants:
    raw: np.ndarray
    finite_count: int
    positive_count: int
    normalized: Optional[np.ndarray] = None
    normalized_ready: bool = False

    def select(self, min_match_pixels: int) -> Tuple[np.ndarray, str]:
        if self.positive_count < int(min_match_pixels):
            return self.raw, "raw"
        if not self.normalized_ready:
            positive = self.raw[np.isfinite(self.raw) & (self.raw > 0)]
            normalized = None
            if positive.size >= int(min_match_pixels):
                clip_value = float(np.percentile(
                    positive, STRUCTURE_CLIP_PERCENTILE, overwrite_input=True
                ))
                if math.isfinite(clip_value) and clip_value > 0.0:
                    normalized = self.raw / clip_value
                    np.clip(normalized, 0.0, 1.0, out=normalized)
                    normalized = normalized.astype(np.float32, copy=False)
                    normalized_finite = np.isfinite(normalized)
                    finite_positive = normalized[
                        normalized_finite & (normalized > 0)
                    ]
                    if finite_positive.size:
                        suppress = float(
                            np.percentile(
                                finite_positive,
                                STRUCTURE_SUPPRESS_PERCENTILE,
                                overwrite_input=True,
                            )
                        )
                        normalized[
                            normalized_finite & (normalized < suppress)
                        ] = 0.0
            if normalized is not None:
                normalized.setflags(write=False)
            self.normalized = normalized
            self.normalized_ready = True
        if self.normalized is None:
            return self.raw, "raw"
        return self.normalized, "normalized"


@dataclass
class QAAssessmentCache:
    candidate_cells: Optional[
        List[Tuple[int, int, List[QAWindowCandidate]]]
    ] = None
    reference_features: Dict[
        Tuple[int, int, int, int], QAStructuralFeatureVariants
    ] = field(default_factory=dict)
    moving_features: Dict[
        Tuple[int, int, int, int], QAStructuralFeatureVariants
    ] = field(default_factory=dict)
    zncc_sums: Dict[Tuple[object, ...], np.ndarray] = field(default_factory=dict)
    reference_fft_cache: Optional[CPUReferenceFFTCache] = None
    reference_texture_scores: Dict[Tuple[int, int, int, int], float] = field(default_factory=dict)
    capture_radius: int = 0
    future_gpu_batch: Optional[tuple] = None
    reusable_gpu_batch: Optional[tuple] = None
    candidate_texture_scores: Dict[
        Tuple[int, int, int, int], float
    ] = field(default_factory=dict)

QA_THRESHOLD_PROFILES = (
    QAThresholdProfile(
        "strict",
        MIN_WINDOW_COMMON_VALID_FRACTION,
        MIN_MATCH_PIXELS,
        MIN_MATCH_OVERLAP_RATIO,
        MIN_RELIABLE_MATCH_SCORE,
        MIN_RELIABLE_PEAK_MARGIN,
        MIN_RELIABLE_PEAK_ZSCORE,
    ),
    QAThresholdProfile("relaxed-1", 0.20, 1200, 0.65, 0.080, 0.008, 1.5),
    QAThresholdProfile("relaxed-2", 0.15, 900, 0.60, 0.060, 0.006, 1.2),
    QAThresholdProfile("relaxed-3", 0.10, 600, 0.55, 0.040, 0.004, 0.9),
    QAThresholdProfile("relaxed-4", 0.05, 300, 0.50, 0.020, 0.002, 0.6),
)

def qa_box_sum_same(data: np.ndarray, radius: int) -> np.ndarray:
    return box_sum_same(data, radius)


def qa_nan_box_smooth(data: np.ndarray, radius: int) -> np.ndarray:
    if (EXACT_CPU_KERNELS_ACTIVE and data.dtype == np.float32
            and data.flags.writeable and 1024 <= data.size <= np.iinfo(np.int32).max):
        return _exact_smooth(data, max(0, int(radius)), 0.35)
    valid = np.isfinite(data)
    totals = box_sum_same(data, radius, valid=valid)
    counts = box_count_same(valid, radius)
    possible = box_possible_same(data.shape, radius)
    minimum = np.maximum(9.0, 0.35 * possible)
    usable = valid & (counts >= minimum)
    out = np.full(data.shape, np.nan, dtype=np.float32)
    np.divide(totals, counts, out=out, where=usable)
    out[~usable] = np.nan
    return out

def qa_gradient_magnitude(data: np.ndarray) -> np.ndarray:
    if (EXACT_CPU_KERNELS_ACTIVE and data.dtype == np.float32
            and data.flags.writeable and data.size >= 1024):
        return _exact_gradient(data, False)
    out = np.full(data.shape, np.nan, dtype=np.float32)
    if min(data.shape) < 3:
        return out
    center = data[1:-1, 1:-1]
    north = data[:-2, 1:-1]
    south = data[2:, 1:-1]
    west = data[1:-1, :-2]
    east = data[1:-1, 2:]
    valid = np.isfinite(center) & np.isfinite(north) & np.isfinite(south) & np.isfinite(west) & np.isfinite(east)
    gx = 0.5 * (east - west)
    gy = 0.5 * (south - north)
    np.multiply(gx, gx, out=gx)
    np.multiply(gy, gy, out=gy)
    np.add(gx, gy, out=gx)
    np.sqrt(gx, out=gx)
    magnitude = gx.astype(np.float32, copy=False)
    inner = out[1:-1, 1:-1]
    np.copyto(inner, magnitude, where=valid)
    return out

def qa_build_structural_variants(data: np.ndarray) -> QAStructuralFeatureVariants:
    smoothed = qa_nan_box_smooth(data, STRUCTURE_SMOOTH_RADIUS_PIXELS)
    feature = qa_gradient_magnitude(smoothed)
    finite = np.isfinite(feature)
    feature.setflags(write=False)
    return QAStructuralFeatureVariants(
        raw=feature,
        finite_count=int(np.count_nonzero(finite)),
        positive_count=int(np.count_nonzero(finite & (feature > 0))),
    )


def qa_structural_feature(data: np.ndarray, thresholds: QAThresholdProfile) -> np.ndarray:
    variants = qa_build_structural_variants(data)
    return variants.select(thresholds.min_match_pixels)[0]


def qa_valid_bbox(data: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    valid = np.isfinite(data)
    rows = np.flatnonzero(valid.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(valid.any(axis=0))
    return (
        int(rows[0]),
        int(rows[-1]) + 1,
        int(cols[0]),
        int(cols[-1]) + 1,
    )

def qa_robust_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 32:
        return 0.0
    q10, q90 = np.percentile(finite, [10.0, 90.0], overwrite_input=True)
    return max(0.0, float(q90 - q10))

def qa_coarse_texture(data: np.ndarray) -> float:
    sampled = data[::4, ::4]
    valid = np.isfinite(sampled)
    if int(np.count_nonzero(valid)) < 64:
        return 0.0
    value_range = qa_robust_range(sampled)
    dx_valid = valid[:, 1:] & valid[:, :-1]
    dy_valid = valid[1:, :] & valid[:-1, :]
    gradients = []
    if bool(dx_valid.any()):
        gradients.append(np.abs(sampled[:, 1:] - sampled[:, :-1])[dx_valid])
    if bool(dy_valid.any()):
        gradients.append(np.abs(sampled[1:, :] - sampled[:-1, :])[dy_valid])
    if not gradients:
        return 0.0
    gradient_values = np.concatenate(gradients)
    gradient_scale = float(np.percentile(
        gradient_values, 75.0, overwrite_input=True
    )) if gradient_values.size else 0.0
    return max(0.0, value_range) * max(0.0, gradient_scale)

def qa_clamp_window(center_y: int, center_x: int, size: int, height: int, width: int) -> Tuple[int, int, int, int]:
    size_y = min(max(32, int(size)), height)
    size_x = min(max(32, int(size)), width)
    y0 = max(0, min(height - size_y, int(center_y) - size_y // 2))
    x0 = max(0, min(width - size_x, int(center_x) - size_x // 2))
    return (y0, y0 + size_y, x0, x0 + size_x)

def qa_candidate_centers(start: int, stop: int, half_size: int, count: int) -> np.ndarray:
    low = start + half_size
    high = stop - half_size
    if low > high:
        return np.array([0.5 * (start + stop)], dtype=np.float64)
    if count <= 1 or low == high:
        return np.array([0.5 * (low + high)], dtype=np.float64)
    return np.linspace(low, high, count)

def qa_choose_window_in_cell(reference: np.ndarray, moving: np.ndarray, cell: Tuple[int, int, int, int], thresholds: QAThresholdProfile) -> Optional[Tuple[int, int, int, int]]:
    y0, y1, x0, x1 = cell
    height, width = reference.shape
    size = min(QA_WINDOW_SIZE_PIXELS, height, width)
    half = size // 2
    centers_y = qa_candidate_centers(y0, y1, half, QA_CANDIDATES_PER_CELL_AXIS)
    centers_x = qa_candidate_centers(x0, x1, half, QA_CANDIDATES_PER_CELL_AXIS)
    best_bounds = None
    best_score = -math.inf
    for cy in centers_y:
        for cx in centers_x:
            bounds = qa_clamp_window(int(round(cy)), int(round(cx)), size, height, width)
            wy0, wy1, wx0, wx1 = bounds
            ref_win = reference[wy0:wy1, wx0:wx1]
            mov_win = moving[wy0:wy1, wx0:wx1]
            common = np.isfinite(ref_win) & np.isfinite(mov_win)
            common_fraction = float(np.count_nonzero(common)) / max(1, ref_win.size)
            if common_fraction < thresholds.min_window_common_valid_fraction:
                continue
            ref_texture = qa_coarse_texture(ref_win)
            mov_texture = qa_coarse_texture(mov_win)
            if ref_texture <= 0.0 or mov_texture <= 0.0:
                continue
            score = common_fraction * math.sqrt(ref_texture * mov_texture)
            if score > best_score:
                best_score = score
                best_bounds = bounds
    return best_bounds

def qa_prepare_candidate_cells(
    reference: np.ndarray,
    moving: np.ndarray,
    defer_texture: bool = False,
) -> List[Tuple[int, int, List[QAWindowCandidate]]]:
    bbox = qa_valid_bbox(moving)
    if bbox is None:
        return []
    y0, y1, x0, x1 = bbox
    row_edges = np.linspace(y0, y1, QA_GRID_ROWS + 1, dtype=int)
    col_edges = np.linspace(x0, x1, QA_GRID_COLS + 1, dtype=int)
    height, width = reference.shape
    size = min(QA_WINDOW_SIZE_PIXELS, height, width)
    half = size // 2
    metrics: Dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
    cells: List[Tuple[int, int, List[QAWindowCandidate]]] = []
    # Overlapping candidates otherwise reread both float rasters for every
    # validity check. Reuse a compact mask only when those repeated scans are
    # expected to cover substantially more pixels than a single full pass.
    candidate_rows = QA_GRID_ROWS * (
        QA_CANDIDATES_PER_CELL_AXIS
        if (y1 - y0) > QA_GRID_ROWS * size
        else 1
    )
    candidate_cols = QA_GRID_COLS * (
        QA_CANDIDATES_PER_CELL_AXIS
        if (x1 - x0) > QA_GRID_COLS * size
        else 1
    )
    common_mask = None
    if candidate_rows * candidate_cols * size * size > 2 * reference.size:
        common_mask = np.isfinite(reference) & np.isfinite(moving)

    for grid_row in range(QA_GRID_ROWS):
        for grid_col in range(QA_GRID_COLS):
            cell_y0 = int(row_edges[grid_row])
            cell_y1 = int(row_edges[grid_row + 1])
            cell_x0 = int(col_edges[grid_col])
            cell_x1 = int(col_edges[grid_col + 1])
            centers_y = qa_candidate_centers(
                cell_y0,
                cell_y1,
                half,
                QA_CANDIDATES_PER_CELL_AXIS,
            )
            centers_x = qa_candidate_centers(
                cell_x0,
                cell_x1,
                half,
                QA_CANDIDATES_PER_CELL_AXIS,
            )
            candidates: List[QAWindowCandidate] = []
            for cy in centers_y:
                for cx in centers_x:
                    bounds = qa_clamp_window(
                        int(round(cy)),
                        int(round(cx)),
                        size,
                        height,
                        width,
                    )
                    metric = metrics.get(bounds)
                    if metric is None:
                        wy0, wy1, wx0, wx1 = bounds
                        ref_win = reference[wy0:wy1, wx0:wx1]
                        mov_win = moving[wy0:wy1, wx0:wx1]
                        common = (
                            common_mask[wy0:wy1, wx0:wx1]
                            if common_mask is not None
                            else np.isfinite(ref_win) & np.isfinite(mov_win)
                        )
                        common_fraction = float(np.count_nonzero(common)) / max(1, ref_win.size)
                        texture_score = math.nan
                        if not defer_texture:
                            ref_texture = qa_coarse_texture(ref_win)
                            mov_texture = qa_coarse_texture(mov_win)
                            texture_score = -math.inf
                            if ref_texture > 0.0 and mov_texture > 0.0:
                                texture_score = common_fraction * math.sqrt(
                                    ref_texture * mov_texture
                                )
                        metric = (common_fraction, texture_score)
                        metrics[bounds] = metric
                    candidates.append(
                        QAWindowCandidate(
                            bounds=bounds,
                            common_fraction=metric[0],
                            texture_score=metric[1],
                        )
                    )
            cells.append((grid_row, grid_col, candidates))
    return cells


def qa_make_qa_windows(
    reference: np.ndarray,
    moving: np.ndarray,
    thresholds: QAThresholdProfile,
    cache: Optional[QAAssessmentCache] = None,
) -> List[Tuple[int, int, int, int, int, int]]:
    active_cache = cache if cache is not None else QAAssessmentCache()
    if active_cache.candidate_cells is None:
        # Defer percentile-based texture work until a profile can select the
        # candidate. Relaxed/custom profiles may still evaluate it later.
        active_cache.candidate_cells = qa_prepare_candidate_cells(
            reference, moving, defer_texture=True
        )

    windows = []
    seen = set()
    for grid_row, grid_col, candidates in active_cache.candidate_cells:
        best_bounds = None
        best_score = -math.inf
        for candidate in candidates:
            if (
                candidate.common_fraction
                < thresholds.min_window_common_valid_fraction
            ):
                continue
            texture_score = candidate.texture_score
            if math.isnan(texture_score):
                texture_score = active_cache.candidate_texture_scores.get(
                    candidate.bounds
                )
                if texture_score is None:
                    wy0, wy1, wx0, wx1 = candidate.bounds
                    ref_texture = active_cache.reference_texture_scores.get(candidate.bounds)
                    if ref_texture is None:
                        ref_texture = qa_coarse_texture(reference[wy0:wy1, wx0:wx1])
                        active_cache.reference_texture_scores[candidate.bounds] = ref_texture
                    mov_texture = qa_coarse_texture(moving[wy0:wy1, wx0:wx1])
                    texture_score = -math.inf
                    if ref_texture > 0.0 and mov_texture > 0.0:
                        texture_score = candidate.common_fraction * math.sqrt(
                            ref_texture * mov_texture
                        )
                    active_cache.candidate_texture_scores[
                        candidate.bounds
                    ] = texture_score
            if texture_score > best_score:
                best_score = texture_score
                best_bounds = candidate.bounds
        if best_bounds is None or best_bounds in seen:
            continue
        seen.add(best_bounds)
        windows.append((*best_bounds, grid_row, grid_col))
    return windows

def qa_next_power_of_two(value: int) -> int:
    return fast_fft_length(value)


def qa_fft_correlation_lag_window(
    left_fft: np.ndarray,
    right_fft: np.ndarray,
    fft_shape: Tuple[int, int],
    row_lags: np.ndarray,
    col_lags: np.ndarray,
) -> np.ndarray:
    return _cpu_correlation_lag_window(
        left_fft,
        right_fft,
        fft_shape,
        row_lags,
        col_lags,
    )


def qa_masked_zncc_surface(
    reference: np.ndarray,
    moving: np.ndarray,
    radius: int,
    thresholds: QAThresholdProfile,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if reference.shape != moving.shape:
        raise RuntimeError('Reference and moving windows must have identical shapes.')
    ref_valid = np.isfinite(reference)
    mov_valid = np.isfinite(moving)
    if (
        int(np.count_nonzero(ref_valid)) < thresholds.min_match_pixels
        or int(np.count_nonzero(mov_valid)) < thresholds.min_match_pixels
    ):
        raise RuntimeError('Insufficient valid structural pixels.')
    row_lags = np.arange(-radius, radius + 1, dtype=np.int64)
    col_lags = np.arange(-radius, radius + 1, dtype=np.int64)
    fft_shape = linear_fft_shape(reference.shape)
    sums = _masked_zncc_sums_cpu(
        reference,
        moving,
        ref_valid,
        mov_valid,
        fft_shape,
        row_lags,
        col_lags,
    )
    scores, counts, max_overlap = zncc_scores_from_sums(
        sums,
        thresholds.min_match_pixels,
        thresholds.min_match_overlap_ratio,
    )
    return scores, counts, row_lags, col_lags, max_overlap


def qa_parabolic_offset(left: float, center: float, right: float) -> float:
    if not all((math.isfinite(value) for value in (left, center, right))):
        return 0.0
    denominator = left - 2.0 * center + right
    if denominator >= -1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    if not math.isfinite(offset):
        return 0.0
    return max(-0.75, min(0.75, float(offset)))

@lru_cache(maxsize=1)
def qa_quadratic_design() -> np.ndarray:
    design = np.asarray(
        [
            [x * x, y * y, x * y, x, y, 1.0]
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ],
        dtype=np.float64,
    )
    design.setflags(write=False)
    return design


def qa_quadratic_subpixel(scores: np.ndarray, row_index: int, col_index: int) -> Tuple[float, float, str]:
    if row_index <= 0 or row_index >= scores.shape[0] - 1 or col_index <= 0 or (col_index >= scores.shape[1] - 1):
        return (0.0, 0.0, 'integer-edge')
    neighborhood = scores[row_index - 1:row_index + 2, col_index - 1:col_index + 2]
    if not bool(np.isfinite(neighborhood).all()):
        row_offset = qa_parabolic_offset(scores[row_index - 1, col_index], scores[row_index, col_index], scores[row_index + 1, col_index])
        col_offset = qa_parabolic_offset(scores[row_index, col_index - 1], scores[row_index, col_index], scores[row_index, col_index + 1])
        return (row_offset, col_offset, 'separable-parabola')
    design = qa_quadratic_design()
    response = np.asarray(neighborhood, dtype=np.float64).reshape(-1)
    try:
        a, b, c, d, e, _f = np.linalg.lstsq(design, response, rcond=None)[0]
        hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(hessian)
        if bool(np.all(eigenvalues < -1e-10)):
            offset_xy = -np.linalg.solve(hessian, np.array([d, e], dtype=np.float64))
            col_offset = float(offset_xy[0])
            row_offset = float(offset_xy[1])
            if abs(row_offset) <= 0.9 and abs(col_offset) <= 0.9:
                return (row_offset, col_offset, 'quadratic-2d')
    except Exception:
        pass
    row_offset = qa_parabolic_offset(scores[row_index - 1, col_index], scores[row_index, col_index], scores[row_index + 1, col_index])
    col_offset = qa_parabolic_offset(scores[row_index, col_index - 1], scores[row_index, col_index], scores[row_index, col_index + 1])
    return (row_offset, col_offset, 'separable-parabola')

def qa_second_best_score(scores: np.ndarray, row_index: int, col_index: int) -> float:
    mask = np.isfinite(scores)
    y0 = max(0, row_index - 1)
    y1 = min(scores.shape[0], row_index + 2)
    x0 = max(0, col_index - 1)
    x1 = min(scores.shape[1], col_index + 2)
    mask[y0:y1, x0:x1] = False
    values = scores[mask]
    return float(np.max(values)) if values.size else math.nan

def qa_peak_zscore(scores: np.ndarray, row_index: int, col_index: int, best_score: float) -> float:
    mask = np.isfinite(scores)
    y0 = max(0, row_index - 1)
    y1 = min(scores.shape[0], row_index + 2)
    x0 = max(0, col_index - 1)
    x1 = min(scores.shape[1], col_index + 2)
    mask[y0:y1, x0:x1] = False
    background = scores[mask]
    if background.size < 5:
        return math.nan
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    sigma = max(1e-06, 1.4826 * mad)
    return float((best_score - median) / sigma)


def qa_peak_background_metrics(
    scores: np.ndarray,
    row_index: int,
    col_index: int,
    best_score: float,
) -> Tuple[float, float]:
    mask = np.isfinite(scores)
    y0 = max(0, row_index - 1)
    y1 = min(scores.shape[0], row_index + 2)
    x0 = max(0, col_index - 1)
    x1 = min(scores.shape[1], col_index + 2)
    mask[y0:y1, x0:x1] = False
    background = scores[mask]
    second = float(np.max(background)) if background.size else math.nan
    if background.size < 5:
        return second, math.nan
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    sigma = max(1e-06, 1.4826 * mad)
    return second, float((best_score - median) / sigma)

def qa_peak_curvature(scores: np.ndarray, row_index: int, col_index: int) -> float:
    if row_index <= 0 or row_index >= scores.shape[0] - 1 or col_index <= 0 or (col_index >= scores.shape[1] - 1):
        return 0.0
    center = float(scores[row_index, col_index])
    neighbors = np.array([scores[row_index - 1, col_index], scores[row_index + 1, col_index], scores[row_index, col_index - 1], scores[row_index, col_index + 1]], dtype=np.float64)
    finite = neighbors[np.isfinite(neighbors)]
    if finite.size == 0:
        return 0.0
    return max(0.0, center - float(finite.mean()))

def qa_match_from_surface(
    scores: np.ndarray,
    counts: np.ndarray,
    row_lags: np.ndarray,
    col_lags: np.ndarray,
    max_overlap: int,
    grid_row: int,
    grid_col: int,
    thresholds: QAThresholdProfile,
    search_radius: int,
) -> QAWindowMatch:
    if not bool(np.isfinite(scores).any()):
        raise RuntimeError('No valid correlation peak.')
    flat_index = int(np.nanargmax(scores))
    row_index, col_index = np.unravel_index(flat_index, scores.shape)
    best_score = float(scores[row_index, col_index])
    integer_row = int(row_lags[row_index])
    integer_col = int(col_lags[col_index])
    edge_hit = abs(integer_row) == search_radius or abs(integer_col) == search_radius
    row_offset, col_offset, subpixel_method = qa_quadratic_subpixel(
        scores,
        row_index,
        col_index,
    )
    if edge_hit:
        row_offset = 0.0
        col_offset = 0.0
    row_shift = float(integer_row + row_offset)
    col_shift = float(integer_col + col_offset)
    residual = float(math.hypot(row_shift, col_shift))
    zero_value = float(scores[search_radius, search_radius])
    zero_score = zero_value if math.isfinite(zero_value) else math.nan
    gain = best_score - zero_score if math.isfinite(zero_score) else math.nan
    second, zscore = qa_peak_background_metrics(
        scores, row_index, col_index, best_score
    )
    margin = best_score - second if math.isfinite(second) else math.nan
    curvature = qa_peak_curvature(scores, row_index, col_index)
    overlap_pixels = int(round(float(counts[row_index, col_index])))
    overlap_fraction = overlap_pixels / max(1, int(max_overlap))
    rejection = []
    if best_score < thresholds.min_reliable_match_score:
        rejection.append('low_match_score')
    if not math.isfinite(margin) or margin < thresholds.min_reliable_peak_margin:
        rejection.append('ambiguous_peak')
    if not math.isfinite(zscore) or zscore < thresholds.min_reliable_peak_zscore:
        rejection.append('weak_peak_zscore')
    if overlap_pixels < thresholds.min_match_pixels:
        rejection.append('low_overlap_pixels')
    if overlap_fraction < thresholds.min_match_overlap_ratio:
        rejection.append('low_overlap_fraction')
    if edge_hit:
        rejection.append('search_edge_hit')
    reliable = not rejection
    return QAWindowMatch(
        grid_row=grid_row,
        grid_col=grid_col,
        row_shift_px=row_shift,
        col_shift_px=col_shift,
        residual_px=residual,
        best_score=best_score,
        zero_score=zero_score,
        score_gain=gain,
        second_best_score=second,
        peak_margin=margin,
        peak_zscore=zscore,
        peak_curvature=curvature,
        overlap_pixels=overlap_pixels,
        overlap_fraction=overlap_fraction,
        search_edge_hit=edge_hit,
        subpixel_method=subpixel_method,
        reliable=reliable,
        rejection_reason=';'.join(rejection),
    )


def qa_match_window(
    reference: np.ndarray,
    moving: np.ndarray,
    grid_row: int,
    grid_col: int,
    thresholds: QAThresholdProfile,
    search_radius: int = MAX_RESIDUAL_SEARCH_PIXELS,
) -> QAWindowMatch:
    search_radius = max(1, int(search_radius))
    ref_feature = qa_structural_feature(reference, thresholds)
    mov_feature = qa_structural_feature(moving, thresholds)
    scores, counts, row_lags, col_lags, max_overlap = qa_masked_zncc_surface(
        ref_feature,
        moving=mov_feature,
        radius=search_radius,
        thresholds=thresholds,
    )
    return qa_match_from_surface(
        scores,
        counts,
        row_lags,
        col_lags,
        max_overlap,
        grid_row,
        grid_col,
        thresholds,
        search_radius,
    )


def _qa_batch_zncc_sums_gpu_job(
    reference_stack: np.ndarray,
    moving_stack: np.ndarray,
    radius: int,
) -> np.ndarray:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        batch_size, height, width = reference_stack.shape
        fft_shape = linear_fft_shape((height, width))
        require_gpu_memory(qa_batch_gpu_memory_estimate(batch_size, (height, width), fft_shape))
        row_lags = np.arange(-radius, radius + 1, dtype=np.int64)
        col_lags = np.arange(-radius, radius + 1, dtype=np.int64)
        sums = _gpu_zncc_windows(reference_stack, moving_stack, fft_shape, row_lags, col_lags, True)
        return cp.asnumpy(cp.stack(sums, axis=1))


def _qa_scores_from_sums_gpu_job(
    sums_batch: np.ndarray,
    min_match_pixels: int,
    min_overlap_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        sums = cp.asarray(sums_batch, dtype=cp.float64)
        values = tuple((sums[:, index] for index in range(6)))
        scores, counts, maximum = _gpu_qa_scores_from_windows(values, min_match_pixels, min_overlap_ratio)
        scores_host, counts_host, maximum_host = gpu_download_arrays(scores, counts, maximum)
        return scores_host, counts_host, maximum_host.astype(np.int64, copy=False)


def _qa_batch_zncc_gpu_job(
    reference_stack: np.ndarray,
    moving_stack: np.ndarray,
    radius: int,
    min_match_pixels: int,
    min_overlap_ratio: float,
    storage_radius: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert cp is not None
    with cp.cuda.Device(GPU_DEVICE_ID):
        batch_size, height, width = reference_stack.shape
        fft_shape = linear_fft_shape((height, width))
        require_gpu_memory(qa_batch_gpu_memory_estimate(batch_size, (height, width), fft_shape))
        retained_radius = max(radius, int(storage_radius or radius))
        row_lags = np.arange(-retained_radius, retained_radius + 1, dtype=np.int64)
        col_lags = np.arange(-retained_radius, retained_radius + 1, dtype=np.int64)
        sums = _gpu_zncc_windows(reference_stack, moving_stack, fft_shape, row_lags, col_lags, True)
        visible_sums = sums
        if retained_radius > radius:
            offset = retained_radius - radius
            stop = offset + 2 * radius + 1
            visible_sums = tuple(cp.ascontiguousarray(
                value[:, offset:stop, offset:stop]
            ) for value in sums)
        scores, counts, maximum = _gpu_qa_scores_from_windows(visible_sums, min_match_pixels, min_overlap_ratio)
        sums_host, scores_host, counts_host, maximum_host = gpu_download_arrays(
            cp.stack(sums, axis=1), scores, counts, maximum
        )
        return sums_host, scores_host, counts_host, maximum_host.astype(np.int64, copy=False)


_SHARED_REFERENCE_FEATURES = OrderedDict()
_SHARED_REFERENCE_FEATURE_BYTES = 0
_SHARED_REFERENCE_FEATURE_LOCK = threading.Lock()
_SHARED_REFERENCE_FEATURE_LIMIT = max(0, int(os.environ.get("TIR_BRIDGE_SHARED_QA_MB", "256"))) * 1024 ** 2


def clear_shared_reference_features() -> None:
    global _SHARED_REFERENCE_FEATURE_BYTES
    with _SHARED_REFERENCE_FEATURE_LOCK:
        _SHARED_REFERENCE_FEATURES.clear()
        _SHARED_REFERENCE_FEATURE_BYTES = 0


def shared_reference_variants(data, bounds):
    """Share only internally owned, unchanged reference-grid windows."""
    global _SHARED_REFERENCE_FEATURE_BYTES
    if not _SHARED_REFERENCE_FEATURE_LIMIT:
        return None
    with REFERENCE_GRID_CACHE_LOCK:
        owned = any(data is item.data for item in REFERENCE_GRID_CACHE.values())
    if not owned:
        return None
    key = (id(data), bounds, STRUCTURE_SMOOTH_RADIUS_PIXELS,
           STRUCTURE_CLIP_PERCENTILE, STRUCTURE_SUPPRESS_PERCENTILE)
    with _SHARED_REFERENCE_FEATURE_LOCK:
        cached = _SHARED_REFERENCE_FEATURES.get(key)
        if cached is not None and cached[0]() is data:
            _SHARED_REFERENCE_FEATURES.move_to_end(key)
            return cached[1]
    y0, y1, x0, x1 = bounds
    variants = qa_build_structural_variants(data[y0:y1, x0:x1])
    size = 2 * int(variants.raw.nbytes)
    if size > _SHARED_REFERENCE_FEATURE_LIMIT:
        return variants
    with _SHARED_REFERENCE_FEATURE_LOCK:
        cached = _SHARED_REFERENCE_FEATURES.get(key)
        if cached is not None and cached[0]() is data:
            return cached[1]
        if cached is not None:
            _SHARED_REFERENCE_FEATURE_BYTES -= cached[2]
            del _SHARED_REFERENCE_FEATURES[key]
        while _SHARED_REFERENCE_FEATURES and _SHARED_REFERENCE_FEATURE_BYTES + size > _SHARED_REFERENCE_FEATURE_LIMIT:
            _, removed = _SHARED_REFERENCE_FEATURES.popitem(last=False)
            _SHARED_REFERENCE_FEATURE_BYTES -= removed[2]
        _SHARED_REFERENCE_FEATURES[key] = (weakref.ref(data), variants, size)
        _SHARED_REFERENCE_FEATURE_BYTES += size
    return variants


def qa_feature_variants_for_bounds(
    data: np.ndarray,
    bounds: Tuple[int, int, int, int],
    store: Dict[Tuple[int, int, int, int], QAStructuralFeatureVariants],
) -> QAStructuralFeatureVariants:
    variants = store.get(bounds)
    if variants is None:
        variants = shared_reference_variants(data, bounds) if isinstance(store, QAReferenceFeatureStore) else None
        if variants is None:
            wy0, wy1, wx0, wx1 = bounds
            variants = qa_build_structural_variants(data[wy0:wy1, wx0:wx1])
        store[bounds] = variants
    return variants


def qa_cpu_zncc_sums(
    reference: np.ndarray,
    moving: np.ndarray,
    radius: int,
    reference_fft_cache: Optional[CPUReferenceFFTCache] = None,
) -> np.ndarray:
    ref_valid = np.isfinite(reference)
    mov_valid = np.isfinite(moving)
    row_lags = np.arange(-radius, radius + 1, dtype=np.int64)
    col_lags = np.arange(-radius, radius + 1, dtype=np.int64)
    sums = _masked_zncc_sums_cpu(
        reference,
        moving,
        ref_valid,
        mov_valid,
        linear_fft_shape(reference.shape),
        row_lags,
        col_lags,
        reference_fft_cache=reference_fft_cache,
    )
    return np.stack(sums, axis=0)


def _qa_match_windows_with_executor(
    reference: np.ndarray,
    moving: np.ndarray,
    windows: Sequence[Tuple[int, int, int, int, int, int]],
    thresholds: QAThresholdProfile,
    search_radius: int = MAX_RESIDUAL_SEARCH_PIXELS,
    cache: Optional[QAAssessmentCache] = None,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[List[Tuple[Tuple[int, int, int, int, int, int], QAWindowMatch]], int]:
    active_cache = cache if cache is not None else QAAssessmentCache()
    search_radius = max(1, int(search_radius))
    prepared = []
    rejected = 0

    def cpu_sums_for_item(item):
        radius = max(search_radius, int(active_cache.capture_radius))
        sums = qa_cpu_zncc_sums(
            item[2], item[3], radius, active_cache.reference_fft_cache
        )
        if radius > search_radius:
            key = (item[1], radius, item[4], item[5], "cpu")
            active_cache.zncc_sums[key] = sums
            offset = radius - search_radius
            stop = offset + 2 * search_radius + 1
            return np.ascontiguousarray(sums[:, offset:stop, offset:stop])
        return sums

    def prepare_window(window):
        bounds = tuple(int(value) for value in window[:4])
        try:
            ref_variants = qa_feature_variants_for_bounds(
                reference,
                bounds,
                active_cache.reference_features,
            )
            mov_variants = qa_feature_variants_for_bounds(
                moving,
                bounds,
                active_cache.moving_features,
            )
            ref_feature, ref_variant = ref_variants.select(
                thresholds.min_match_pixels
            )
            mov_feature, mov_variant = mov_variants.select(
                thresholds.min_match_pixels
            )
            if (
                ref_variants.finite_count < thresholds.min_match_pixels
                or mov_variants.finite_count < thresholds.min_match_pixels
            ):
                raise RuntimeError('Insufficient valid structural pixels.')
            return (
                window,
                bounds,
                ref_feature,
                mov_feature,
                ref_variant,
                mov_variant,
            )
        except Exception:
            return None

    unique_bounds = len(
        {tuple(int(value) for value in window[:4]) for window in windows}
    ) == len(windows)
    parallel_windows = (
        local_executor is not None
        and ACTIVE_LOCAL_WINDOW_WORKERS > 1
        and len(windows) > 1
        and unique_bounds
    )
    pipeline_cpu = parallel_windows and not gpu_is_active()

    if pipeline_cpu:
        def prepare_cpu_window(window):
            item = prepare_window(window)
            if item is None:
                return None, None, None
            key = (
                item[1],
                search_radius,
                item[4],
                item[5],
                "cpu",
            )
            sums_array = (
                None
                if key in active_cache.zncc_sums
                else cpu_sums_for_item(item)
            )
            return item, key, sums_array

        pipelined_results = list(
            local_executor.map(prepare_cpu_window, windows)
        )
        for item, key, sums_array in pipelined_results:
            if item is None:
                rejected += 1
                continue
            prepared.append(item)
            if sums_array is not None:
                active_cache.zncc_sums[key] = sums_array
    else:
        if parallel_windows:
            prepared_results = list(local_executor.map(prepare_window, windows))
        else:
            prepared_results = [prepare_window(window) for window in windows]

        for item in prepared_results:
            if item is None:
                rejected += 1
            else:
                prepared.append(item)

    results = []
    row_lags = np.arange(-search_radius, search_radius + 1, dtype=np.int64)
    col_lags = np.arange(-search_radius, search_radius + 1, dtype=np.int64)
    can_batch_gpu = (
        gpu_is_active()
        and len(prepared) >= GPU_QA_BATCH_MIN_WINDOWS
        and len({item[2].shape for item in prepared}) == 1
    )

    if can_batch_gpu:
        try:
            keys = [
                (
                    item[1],
                    search_radius,
                    item[4],
                    item[5],
                    "gpu",
                )
                for item in prepared
            ]
            batch_signature = tuple((item[1], item[4], item[5]) for item in prepared)
            reusable = active_cache.reusable_gpu_batch
            if (
                reusable is not None and reusable[0] == batch_signature
                and reusable[1] == search_radius
                and reusable[2] == GPU_FALLBACK_GENERATION
                and all(key not in active_cache.zncc_sums for key in keys)
            ):
                # Same pixels, window order, feature variants, and FFT batch shape.
                for index, key in enumerate(keys):
                    active_cache.zncc_sums[key] = reusable[3][index]
                active_cache.reusable_gpu_batch = None
            missing = [
                index
                for index, key in enumerate(keys)
                if key not in active_cache.zncc_sums
            ]
            scores_batch = None
            counts_batch = None
            max_overlap_batch = None
            if missing:
                ref_stack = np.stack(
                    [prepared[index][2] for index in missing],
                    axis=0,
                )
                mov_stack = np.stack(
                    [prepared[index][3] for index in missing],
                    axis=0,
                )
                if len(missing) == len(prepared):
                    retained_radius = (
                        max(search_radius, int(active_cache.capture_radius))
                        if active_cache.future_gpu_batch is None
                        else search_radius
                    )
                    generation = GPU_FALLBACK_GENERATION
                    (
                        sums_batch,
                        scores_batch,
                        counts_batch,
                        max_overlap_batch,
                    ) = submit_gpu_job(
                        _qa_batch_zncc_gpu_job,
                        ref_stack,
                        mov_stack,
                        search_radius,
                        int(thresholds.min_match_pixels),
                        float(thresholds.min_match_overlap_ratio),
                        retained_radius,
                    )
                    if retained_radius > search_radius:
                        if generation == GPU_FALLBACK_GENERATION:
                            active_cache.future_gpu_batch = (
                                batch_signature, retained_radius, generation, sums_batch,
                            )
                        offset = retained_radius - search_radius
                        stop = offset + 2 * search_radius + 1
                        sums_batch = np.ascontiguousarray(
                            sums_batch[:, :, offset:stop, offset:stop]
                        )
                else:
                    sums_batch = submit_gpu_job(
                        _qa_batch_zncc_sums_gpu_job,
                        ref_stack,
                        mov_stack,
                        search_radius,
                    )
                for batch_index, prepared_index in enumerate(missing):
                    active_cache.zncc_sums[keys[prepared_index]] = sums_batch[
                        batch_index
                    ]

            if scores_batch is None:
                cached_sums = np.stack(
                    [active_cache.zncc_sums[key] for key in keys],
                    axis=0,
                )
                scores_batch, counts_batch, max_overlap_batch = submit_gpu_job(
                    _qa_scores_from_sums_gpu_job,
                    cached_sums,
                    int(thresholds.min_match_pixels),
                    float(thresholds.min_match_overlap_ratio),
                )
            for index, item in enumerate(prepared):
                window = item[0]
                try:
                    match = qa_match_from_surface(
                        scores_batch[index],
                        counts_batch[index],
                        row_lags,
                        col_lags,
                        int(max_overlap_batch[index]),
                        int(window[4]),
                        int(window[5]),
                        thresholds,
                        search_radius,
                    )
                    results.append((window, match))
                except Exception:
                    rejected += 1
            return results, rejected
        except Exception as exc:
            record_gpu_fallback(exc)

    cpu_keys = [
        (
            item[1],
            search_radius,
            item[4],
            item[5],
            "cpu",
        )
        for item in prepared
    ]
    missing_cpu = [
        index
        for index, key in enumerate(cpu_keys)
        if key not in active_cache.zncc_sums
    ]

    def compute_cpu_sums(prepared_index: int) -> Tuple[int, np.ndarray]:
        item = prepared[prepared_index]
        return (
            prepared_index,
            cpu_sums_for_item(item),
        )

    if (
        local_executor is not None
        and ACTIVE_LOCAL_WINDOW_WORKERS > 1
        and len(missing_cpu) > 1
    ):
        computed_cpu = list(local_executor.map(compute_cpu_sums, missing_cpu))
    else:
        computed_cpu = [compute_cpu_sums(index) for index in missing_cpu]
    for prepared_index, sums_array in computed_cpu:
        active_cache.zncc_sums[cpu_keys[prepared_index]] = sums_array

    for item, key in zip(prepared, cpu_keys):
        window = item[0]
        try:
            sums_array = active_cache.zncc_sums[key]
            scores, counts, max_overlap = zncc_scores_from_sums(
                tuple(sums_array[index] for index in range(6)),
                thresholds.min_match_pixels,
                thresholds.min_match_overlap_ratio,
            )
            match = qa_match_from_surface(
                scores,
                counts,
                row_lags,
                col_lags,
                max_overlap,
                int(window[4]),
                int(window[5]),
                thresholds,
                search_radius,
            )
            results.append((window, match))
        except Exception:
            rejected += 1
    return results, rejected


def qa_match_windows(
    reference: np.ndarray,
    moving: np.ndarray,
    windows: Sequence[Tuple[int, int, int, int, int, int]],
    thresholds: QAThresholdProfile,
    search_radius: int = MAX_RESIDUAL_SEARCH_PIXELS,
    cache: Optional[QAAssessmentCache] = None,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[List[Tuple[Tuple[int, int, int, int, int, int], QAWindowMatch]], int]:
    if local_executor is not None or ACTIVE_LOCAL_WINDOW_WORKERS <= 1:
        return _qa_match_windows_with_executor(
            reference,
            moving,
            windows,
            thresholds,
            search_radius=search_radius,
            cache=cache,
            local_executor=local_executor,
        )
    with ThreadPoolExecutor(
        max_workers=ACTIVE_LOCAL_WINDOW_WORKERS,
        thread_name_prefix="tir_bridge_qa",
        initializer=configure_local_cpu_worker,
    ) as executor:
        return _qa_match_windows_with_executor(
            reference,
            moving,
            windows,
            thresholds,
            search_radius=search_radius,
            cache=cache,
            local_executor=executor,
        )


def qa_finite_stat(values: Sequence[float], function, default: float=math.nan) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return default
    return float(function(finite))

def qa_percentile(values: Sequence[float], q: float) -> float:
    return qa_finite_stat(values, lambda array: np.percentile(array, q))

def qa_directional_coherence(rows: np.ndarray, cols: np.ndarray) -> float:
    magnitudes = np.hypot(rows, cols)
    use = magnitudes > 0.1
    if int(np.count_nonzero(use)) < 2:
        return 0.0
    unit_rows = rows[use] / magnitudes[use]
    unit_cols = cols[use] / magnitudes[use]
    return float(math.hypot(float(unit_rows.mean()), float(unit_cols.mean())))

def qa_confidence_score(matches: Sequence[QAWindowMatch], reliable: Sequence[QAWindowMatch]) -> float:
    total = max(1, len(matches))
    count = len(reliable)
    reliable_fraction = count / total
    median_score = qa_percentile([m.best_score for m in reliable], 50)
    median_margin = qa_percentile([m.peak_margin for m in reliable], 50)
    median_zscore = qa_percentile([m.peak_zscore for m in reliable], 50)
    p90_residual = qa_percentile([m.residual_px for m in reliable], 90)
    edge_fraction = sum((m.search_edge_hit for m in matches)) / total
    evidence_component = min(1.0, count / max(MIN_RELIABLE_WINDOWS, 10))
    coverage_component = min(1.0, reliable_fraction / 0.6)
    score_component = 0.0 if not math.isfinite(median_score) else np.clip((median_score - 0.05) / 0.35, 0.0, 1.0)
    margin_component = 0.0 if not math.isfinite(median_margin) else np.clip(median_margin / 0.05, 0.0, 1.0)
    z_component = 0.0 if not math.isfinite(median_zscore) else np.clip(median_zscore / 8.0, 0.0, 1.0)
    residual_component = 0.0 if not math.isfinite(p90_residual) else np.clip(1.0 - p90_residual / MAX_RESIDUAL_SEARCH_PIXELS, 0.0, 1.0)
    edge_component = np.clip(1.0 - edge_fraction, 0.0, 1.0)
    score = 100.0 * (0.15 * evidence_component + 0.15 * coverage_component + 0.15 * score_component + 0.2 * margin_component + 0.15 * z_component + 0.15 * residual_component + 0.05 * edge_component)
    return float(np.clip(score, 0.0, 100.0))

def qa_unreliable_reason_counts(matches: Sequence[QAWindowMatch]) -> str:
    counts = {}
    for match in matches:
        if match.reliable or not match.rejection_reason:
            continue
        for reason in match.rejection_reason.split(';'):
            counts[reason] = counts.get(reason, 0) + 1
    return ';'.join((f'{key}={counts[key]}' for key in sorted(counts)))

def qa_serialize_window_vector(matches: Sequence[QAWindowMatch]) -> str:
    ordered = sorted(matches, key=lambda item: (item.grid_row, item.grid_col))
    values = []
    for match in ordered:
        flag = 'R' if match.reliable else 'X'
        values.append(f'r{match.grid_row}c{match.grid_col}:{match.row_shift_px:+.3f},{match.col_shift_px:+.3f},{match.residual_px:.3f},{flag}')
    return '|'.join(values)


# --------------------------- INTEGRATED PIPELINE ---------------------------

@dataclass
class AssessmentResult:
    status: str
    threshold_profile: str
    attempt_number: int
    systematic_row_px: float
    systematic_col_px: float
    systematic_magnitude_px: float
    p90_residual_px: float
    local_scatter_p90_px: float
    directional_coherence: float
    confidence_0_100: float
    reliable_windows: int
    evaluated_windows: int
    candidate_windows: int
    median_match_score: float
    median_peak_margin: float
    median_peak_zscore: float
    median_overlap_fraction: float
    search_edge_hit_fraction: float
    unreliable_reason_counts: str
    window_vectors: str
    processing_seconds: float
    error: str = ""


@dataclass
class RegistrationState:
    reference: np.ndarray
    moving: np.ndarray
    geometry: RasterGeometry
    nan_stats: NaNFootprintStats
    eco_nan_dilation_pixels: int
    support_loss: float
    overlap_pixels: int
    gradient_overlap_pixels: int
    robust_gradient_overlap_pixels: int
    choice: TranslationChoice
    global_result: ShiftResult
    warp_model: WarpModel
    processing_seconds: float
    fine_texture: Optional["FineTextureEvidence"] = None
    local_texture: Optional["LocalTextureWarp"] = None


@dataclass
class PipelineResult:
    index: int
    input_file: str
    output_file: str
    status: str
    confidence_percent: int
    row: Dict[str, object]
    performance: dict = field(default_factory=dict)


def qa_assess_registered(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_feature_cache: Optional[
        Dict[Tuple[int, int, int, int], QAStructuralFeatureVariants]
    ] = None,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> AssessmentResult:
    start = time.perf_counter()
    last_error = "No assessment attempt produced a finite residual vector."
    qa_cache = QAAssessmentCache(
        reference_features=(
            reference_feature_cache
            if reference_feature_cache is not None
            else {}
        ),
        reference_fft_cache=getattr(reference_feature_cache, "fft_cache", None),
        reference_texture_scores=getattr(reference_feature_cache, "texture_scores", {}),
    )

    if isinstance(reference_feature_cache, QAReferenceFeatureStore):
        if reference_feature_cache.can_retain(moving):
            qa_cache.capture_radius = PAIRED_RESIDUAL_SEARCH_RADIUS_PIXELS

    for attempt_number, thresholds in enumerate(QA_THRESHOLD_PROFILES, start=1):
        windows = qa_make_qa_windows(
            reference,
            moving,
            thresholds,
            cache=qa_cache,
        )
        matched_windows, rejected_windows = qa_match_windows(
            reference,
            moving,
            windows,
            thresholds,
            search_radius=MAX_RESIDUAL_SEARCH_PIXELS,
            cache=qa_cache,
            local_executor=local_executor,
        )
        matches: List[QAWindowMatch] = [
            match for _window, match in matched_windows
        ]

        reliable = [match for match in matches if match.reliable]
        rows = np.asarray([match.row_shift_px for match in reliable], dtype=np.float64)
        cols = np.asarray([match.col_shift_px for match in reliable], dtype=np.float64)
        residuals = np.asarray([match.residual_px for match in reliable], dtype=np.float64)
        systematic_row = qa_percentile(rows, 50)
        systematic_col = qa_percentile(cols, 50)

        if not (
            math.isfinite(systematic_row)
            and math.isfinite(systematic_col)
            and residuals.size > 0
        ):
            last_error = (
                f"{thresholds.name}: no reliable residual vector "
                f"({len(reliable)}/{QA_GRID_ROWS * QA_GRID_COLS} windows; "
                f"{rejected_windows} match exceptions)"
            )
            continue

        systematic_magnitude = math.hypot(systematic_row, systematic_col)
        local_scatter = np.hypot(rows - systematic_row, cols - systematic_col)
        total = max(1, len(matches))
        if isinstance(reference_feature_cache, QAReferenceFeatureStore):
            reference_feature_cache.remember(moving, qa_cache)
        return AssessmentResult(
            status="OK",
            threshold_profile=thresholds.name,
            attempt_number=attempt_number,
            systematic_row_px=float(systematic_row),
            systematic_col_px=float(systematic_col),
            systematic_magnitude_px=float(systematic_magnitude),
            p90_residual_px=qa_percentile(residuals, 90),
            local_scatter_p90_px=qa_percentile(local_scatter, 90),
            directional_coherence=qa_directional_coherence(rows, cols),
            confidence_0_100=qa_confidence_score(matches, reliable),
            reliable_windows=len(reliable),
            evaluated_windows=len(matches),
            candidate_windows=QA_GRID_ROWS * QA_GRID_COLS,
            median_match_score=qa_percentile([m.best_score for m in reliable], 50),
            median_peak_margin=qa_percentile([m.peak_margin for m in reliable], 50),
            median_peak_zscore=qa_percentile([m.peak_zscore for m in reliable], 50),
            median_overlap_fraction=qa_percentile(
                [m.overlap_fraction for m in reliable], 50
            ),
            search_edge_hit_fraction=(
                sum(match.search_edge_hit for match in matches) / total
            ),
            unreliable_reason_counts=qa_unreliable_reason_counts(matches),
            window_vectors=qa_serialize_window_vector(matches),
            processing_seconds=time.perf_counter() - start,
            error="",
        )

    return AssessmentResult(
        status="FAILED",
        threshold_profile="",
        attempt_number=len(QA_THRESHOLD_PROFILES),
        systematic_row_px=math.nan,
        systematic_col_px=math.nan,
        systematic_magnitude_px=math.nan,
        p90_residual_px=math.nan,
        local_scatter_p90_px=math.nan,
        directional_coherence=math.nan,
        confidence_0_100=0.0,
        reliable_windows=0,
        evaluated_windows=0,
        candidate_windows=QA_GRID_ROWS * QA_GRID_COLS,
        median_match_score=math.nan,
        median_peak_margin=math.nan,
        median_peak_zscore=math.nan,
        median_overlap_fraction=math.nan,
        search_edge_hit_fraction=math.nan,
        unreliable_reason_counts="",
        window_vectors="",
        processing_seconds=time.perf_counter() - start,
        error=last_error,
    )


@dataclass
class PairedResidualResult:
    status: str
    before_p90_px: float
    after_p90_px: float
    improvement_p90_px: float
    paired_windows: int
    residual_quality: str
    threshold_profile: str
    error: str = ""


def paired_residual_quality_label(paired_windows: int) -> str:
    count = max(0, int(paired_windows))
    if count == 0:
        return "NO_SUPPORT"
    if count == 1:
        return "LOCAL_ONLY"
    if count <= 5:
        return "LOW_SUPPORT"
    if count <= 9:
        return "MODERATE_SUPPORT"
    return "STRONG_SUPPORT"


def qa_assess_before_after_residuals(
    reference: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    total_affine: Affine,
    reference_feature_cache: Optional[
        Dict[Tuple[int, int, int, int], QAStructuralFeatureVariants]
    ] = None,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> PairedResidualResult:
    """Estimate paired initial and final local displacement on identical windows."""
    if reference.shape != before.shape or reference.shape != after.shape:
        return PairedResidualResult(
            "FAILED", math.nan, math.nan, math.nan, 0, "NO_SUPPORT", "",
            "Reference, before, and after arrays must share one grid.",
        )

    try:
        inverse_affine = ~total_affine
    except Exception as exc:
        return PairedResidualResult(
            "FAILED", math.nan, math.nan, math.nan, 0, "NO_SUPPORT", "",
            f"The final affine is not invertible: {exc}",
        )

    last_error = "No threshold profile produced enough reliable paired windows."
    qa_cache = QAAssessmentCache(
        reference_features=(
            reference_feature_cache
            if reference_feature_cache is not None
            else {}
        ),
        reference_fft_cache=getattr(reference_feature_cache, "fft_cache", None),
        reference_texture_scores=getattr(reference_feature_cache, "texture_scores", {}),
    )
    if isinstance(reference_feature_cache, QAReferenceFeatureStore):
        reference_feature_cache.restore_for_identical(after, qa_cache)
    for thresholds in QA_THRESHOLD_PROFILES:
        windows = qa_make_qa_windows(
            reference,
            after,
            thresholds,
            cache=qa_cache,
        )
        before_residuals: List[float] = []
        after_residuals: List[float] = []

        matched_windows, _rejected_windows = qa_match_windows(
            reference,
            after,
            windows,
            thresholds,
            search_radius=PAIRED_RESIDUAL_SEARCH_RADIUS_PIXELS,
            cache=qa_cache,
            local_executor=local_executor,
        )
        for window, after_match in matched_windows:
            if not after_match.reliable:
                continue
            wy0, wy1, wx0, wx1, _grid_row, _grid_col = window
            destination_row = 0.5 * (wy0 + wy1 - 1)
            destination_col = 0.5 * (wx0 + wx1 - 1)
            source_col, source_row = inverse_affine * (
                destination_col + 0.5, destination_row + 0.5
            )
            source_col -= 0.5
            source_row -= 0.5
            if not (
                math.isfinite(source_row)
                and math.isfinite(source_col)
                and -0.5 <= source_row < before.shape[0] - 0.5
                and -0.5 <= source_col < before.shape[1] - 0.5
            ):
                continue

            matched_reference_row = destination_row + after_match.row_shift_px
            matched_reference_col = destination_col + after_match.col_shift_px
            before_residual = math.hypot(
                matched_reference_row - source_row,
                matched_reference_col - source_col,
            )
            after_residual = float(after_match.residual_px)
            if math.isfinite(before_residual) and math.isfinite(after_residual):
                before_residuals.append(float(before_residual))
                after_residuals.append(after_residual)

        if len(after_residuals) < PAIRED_RESIDUAL_MIN_WINDOWS:
            last_error = (
                f"{thresholds.name}: only {len(after_residuals)}/"
                f"{QA_GRID_ROWS * QA_GRID_COLS} paired windows were reliable."
            )
            continue

        before_p90 = float(np.percentile(before_residuals, 90.0))
        after_p90 = float(np.percentile(after_residuals, 90.0))
        paired_count = len(after_residuals)
        return PairedResidualResult(
            status="OK",
            before_p90_px=before_p90,
            after_p90_px=after_p90,
            improvement_p90_px=before_p90 - after_p90,
            paired_windows=paired_count,
            residual_quality=paired_residual_quality_label(paired_count),
            threshold_profile=thresholds.name,
            error="",
        )

    return PairedResidualResult(
        "FAILED", math.nan, math.nan, math.nan, 0, "NO_SUPPORT", "", last_error
    )


@dataclass
class FineTextureMatch:
    bounds: Tuple[int, int, int, int]
    row_shift: float
    col_shift: float
    score: float
    margin: float
    zscore: float
    pixels: int


@dataclass
class FineTextureEvidence:
    row_shift: float
    col_shift: float
    fit_windows: int
    validation_windows: int
    scatter_p90_px: float
    median_gain: float
    confidence: float
    matches: List[FineTextureMatch]
    processing_seconds: float


def fine_texture_features(data: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Local contrast normalization without cloud-edge or zero-gradient features."""
    valid = np.isfinite(data)
    empty = lambda: (np.full(data.shape, np.nan, np.float32),) * 2
    if scipy_ndimage is None or np.count_nonzero(valid) < 256:
        return empty()
    values = np.where(valid, data - float(np.median(data[valid])), 0.0).astype(np.float32)
    if float(np.ptp(values[valid])) <= 1e-7:
        return empty()
    weight = valid.astype(np.float32)

    def smooth(array, sigma):
        denominator = scipy_ndimage.gaussian_filter(weight, sigma, mode="constant")
        numerator = scipy_ndimage.gaussian_filter(np.where(valid, array, 0.0), sigma, mode="constant")
        return numerator / np.maximum(denominator, 1e-6)

    background = smooth(values, 8.0)
    # Keep missing-data edges outside both derivative supports.
    padded_valid = np.pad(valid, 1, constant_values=False)
    distance = scipy_ndimage.distance_transform_edt(padded_valid)[1:-1, 1:-1]
    usable = valid & (distance > 7.0)
    if np.count_nonzero(usable) < 256:
        return empty()
    features = []
    for sigma in (0.9, 1.8):
        detail = smooth(values, sigma) - background
        dy, dx = np.gradient(detail)
        magnitude = np.hypot(dx, dy).astype(np.float32)
        energy = smooth(magnitude * magnitude, 8.0)
        floor = max(1e-12, 0.25 * float(np.median(energy[usable])))
        normalized = magnitude / np.sqrt(np.maximum(energy, floor))
        feature = np.where(usable, np.minimum(normalized, 3.0), np.nan).astype(np.float32)
        features.append(feature)
    return tuple(features)


def fine_texture_windows(shape: Tuple[int, int]):
    """Separated cores; checkerboard fit and validation partitions."""
    height, width = shape
    ye = np.linspace(0, height, FINE_TEXTURE_GRID + 1, dtype=int)
    xe = np.linspace(0, width, FINE_TEXTURE_GRID + 1, dtype=int)
    windows = []
    for iy in range(FINE_TEXTURE_GRID):
        for ix in range(FINE_TEXTURE_GRID):
            h = min(FINE_TEXTURE_WINDOW_MAX, int(ye[iy + 1] - ye[iy]) - 16)
            w = min(FINE_TEXTURE_WINDOW_MAX, int(xe[ix + 1] - xe[ix]) - 16)
            if min(h, w) < FINE_TEXTURE_MIN_CORE:
                continue
            cy = (int(ye[iy]) + int(ye[iy + 1])) // 2
            cx = (int(xe[ix]) + int(xe[ix + 1])) // 2
            bounds = (cy - h // 2, cy - h // 2 + h, cx - w // 2, cx - w // 2 + w)
            windows.append((bounds, (iy + ix) % 2 == 0))
    return windows


def fine_texture_patch(data: np.ndarray, bounds, extra: int = 0):
    y0, y1, x0, x1 = bounds
    pad = FINE_TEXTURE_HALO + extra
    sy0, sy1 = max(0, y0 - pad), min(data.shape[0], y1 + pad)
    sx0, sx1 = max(0, x0 - pad), min(data.shape[1], x1 + pad)
    features = fine_texture_features(data[sy0:sy1, sx0:sx1])
    return features, (sy0, sx0)


def fine_texture_match(reference: np.ndarray, moving: np.ndarray, bounds):
    radius = FINE_TEXTURE_SEARCH_RADIUS
    y0, y1, x0, x1 = bounds
    ref_features, (ry, rx) = fine_texture_patch(reference, bounds)
    mov_features, (my, mx) = fine_texture_patch(moving, bounds)
    thresholds = QAThresholdProfile("fine_native", 0.40, 512, 0.70,
                                    FINE_TEXTURE_MIN_SCORE, FINE_TEXTURE_MIN_MARGIN,
                                    FINE_TEXTURE_MIN_ZSCORE)
    surfaces, counts_list = [], []
    for ref, mov in zip(ref_features, mov_features):
        r = ref[y0 - ry:y1 - ry, x0 - rx:x1 - rx]
        m = mov[y0 - my:y1 - my, x0 - mx:x1 - mx]
        try:
            scores, counts, _, _, _ = qa_masked_zncc_surface(r, m, radius, thresholds)
        except RuntimeError:
            return None
        if not np.isfinite(scores).any():
            return None
        surfaces.append(scores)
        counts_list.append(counts)
    peaks = [np.asarray(np.unravel_index(np.nanargmax(s), s.shape)) for s in surfaces]
    if float(np.linalg.norm(peaks[0] - peaks[1])) > 1.5:
        return None
    scores = 0.5 * (surfaces[0] + surfaces[1])
    if not np.isfinite(scores).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(scores), scores.shape)
    if iy in (0, 2 * radius) or ix in (0, 2 * radius):
        return None
    score = float(scores[iy, ix])
    second, zscore = qa_peak_background_metrics(scores, iy, ix, score)
    margin = score - second
    if (not math.isfinite(margin) or not math.isfinite(zscore)
            or score < FINE_TEXTURE_MIN_SCORE or margin < FINE_TEXTURE_MIN_MARGIN
            or zscore < FINE_TEXTURE_MIN_ZSCORE):
        return None
    dr, dc, _ = qa_quadratic_subpixel(scores, iy, ix)
    return FineTextureMatch(bounds, float(iy - radius + dr), float(ix - radius + dc),
                            score, margin, zscore,
                            int(min(c[iy, ix] for c in counts_list)))


def fine_texture_sample(feature, origin, bounds, transform, order=1):
    y0, y1, x0, x1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    inverse = ~transform
    # Affines act on pixel corners; sample at pixel centers.
    sy = inverse.d * (xx + 0.5) + inverse.e * (yy + 0.5) + inverse.f - 0.5 - origin[0]
    sx = inverse.a * (xx + 0.5) + inverse.b * (yy + 0.5) + inverse.c - 0.5 - origin[1]
    valid = np.isfinite(feature)
    coords = np.asarray([sy, sx])
    support = scipy_ndimage.map_coordinates(valid.astype(np.float32), coords,
                                            order=order, mode="constant", cval=0.0, prefilter=False)
    sampled = scipy_ndimage.map_coordinates(np.where(valid, feature, 0.0), coords,
                                            order=order, mode="constant", cval=0.0, prefilter=False)
    sampled[support < 0.9999] = np.nan
    return sampled


def fine_texture_correlations(ref, before, after):
    valid = np.isfinite(ref) & np.isfinite(before) & np.isfinite(after)
    count = int(np.count_nonzero(valid))
    if count < max(512, int(ref.size * 0.40)):
        return None
    a = ref[valid].astype(np.float64)
    a -= a.mean()
    scores = []
    for values in (before, after):
        b = values[valid].astype(np.float64)
        b -= b.mean()
        denominator = math.sqrt(float(a @ a) * float(b @ b))
        if denominator <= 1e-12:
            return None
        scores.append(float(np.clip((a @ b) / denominator, -1.0, 1.0)))
    return scores[0], scores[1]


def fine_texture_validate_window(prepared, bounds, baseline, candidate):
    y0, y1, x0, x1 = bounds
    reference, ro = fine_texture_patch(prepared.reference, bounds)
    corners = [(x0 + 0.5, y0 + 0.5), (x1 - 0.5, y0 + 0.5),
               (x0 + 0.5, y1 - 0.5), (x1 - 0.5, y1 - 0.5)]
    source = [(~a) * p for a in (baseline, candidate) for p in corners]
    source_bounds = (int(math.floor(min(p[1] for p in source))),
                     int(math.ceil(max(p[1] for p in source))) + 1,
                     int(math.floor(min(p[0] for p in source))),
                     int(math.ceil(max(p[0] for p in source))) + 1)
    # Bound temporary work for implausible affine candidates.
    if max(source_bounds[1] - source_bounds[0], source_bounds[3] - source_bounds[2]) > 1024:
        return None
    h, w = prepared.moving.shape
    sy0, sy1 = max(0, source_bounds[0] - FINE_TEXTURE_HALO), min(h, source_bounds[1] + FINE_TEXTURE_HALO)
    sx0, sx1 = max(0, source_bounds[2] - FINE_TEXTURE_HALO), min(w, source_bounds[3] + FINE_TEXTURE_HALO)
    if min(sy1 - sy0, sx1 - sx0) < FINE_TEXTURE_MIN_CORE:
        return None
    moving = fine_texture_features(prepared.fine_texture_moving[sy0:sy1, sx0:sx1])
    pairs = []
    for r, m in zip(reference, moving):
        rc = r[y0 - ro[0]:y1 - ro[0], x0 - ro[1]:x1 - ro[1]]
        before = fine_texture_sample(m, (sy0, sx0), bounds, baseline)
        after = fine_texture_sample(m, (sy0, sx0), bounds, candidate)
        pair = fine_texture_correlations(rc, before, after)
        if pair is None:
            return None
        pairs.append(pair)
    # Protect strong original edges on the same native support.
    r = prepared.legacy_reference[y0:y1, x0:x1]
    m = prepared.legacy_moving[sy0:sy1, sx0:sx1]
    broad = fine_texture_correlations(
        r, fine_texture_sample(m, (sy0, sx0), bounds, baseline),
        fine_texture_sample(m, (sy0, sx0), bounds, candidate))
    gain = float(np.mean([b - a for a, b in pairs]))
    return gain, broad, float(np.mean([b for _, b in pairs]))


def fine_texture_consensus(matches: List[FineTextureMatch]):
    if not matches:
        return [], None
    points = np.asarray([(m.row_shift, m.col_shift) for m in matches])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    seed = int(np.argmax(np.sum(distances <= FINE_TEXTURE_CONSENSUS_RADIUS, axis=1)))
    keep = distances[seed] <= FINE_TEXTURE_CONSENSUS_RADIUS
    center = np.median(points[keep], axis=0)
    keep = np.linalg.norm(points - center, axis=1) <= FINE_TEXTURE_CONSENSUS_RADIUS
    return [m for m, use in zip(matches, keep) if use], np.median(points[keep], axis=0)


def fine_texture_spread(matches, shape):
    if not matches:
        return False
    centers = np.asarray([((m.bounds[0] + m.bounds[1]) / 2,
                           (m.bounds[2] + m.bounds[3]) / 2) for m in matches])
    spans = np.ptp(centers, axis=0) / np.asarray(shape)
    return bool(np.all(spans >= 0.35))


def refine_fine_texture(prepared, baseline_affine) -> Optional[FineTextureEvidence]:
    """Fit small absolute translation; accept only independent native validation."""
    start = time.perf_counter()
    if not FINE_TEXTURE_ENABLED or scipy_ndimage is None:
        return None
    windows = fine_texture_windows(prepared.reference.shape)
    fit_bounds = [b for b, fit in windows if fit]
    validation_bounds = [b for b, fit in windows if not fit]
    fit_matches = [m for b in fit_bounds
                   if (m := fine_texture_match(prepared.reference, prepared.fine_texture_moving, b)) is not None]
    fit, shift = fine_texture_consensus(fit_matches)
    if (len(fit) < FINE_TEXTURE_MIN_FIT_WINDOWS
            or len(fit) < FINE_TEXTURE_MIN_CONSENSUS_FRACTION * len(fit_matches)
            or len(fit) < 0.5 * len(fit_bounds)
            or not fine_texture_spread(fit, prepared.reference.shape)):
        return None
    candidate = Affine.translation(float(shift[1]), float(shift[0]))
    validation, gains, broad_pairs, compared_centers = [], [], [], []
    checked, reliable = 0, 0
    for bounds in validation_bounds:
        match = fine_texture_match(prepared.reference, prepared.fine_texture_moving, bounds)
        pair = fine_texture_validate_window(prepared, bounds, baseline_affine, candidate)
        if pair is not None:
            checked += 1
            compared_centers.append(((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2))
            gain, broad, _ = pair
            gains.append(gain)
            if broad is not None:
                broad_pairs.append(broad)
        if match is not None:
            reliable += 1
            if (math.hypot(match.row_shift - shift[0], match.col_shift - shift[1])
                    <= FINE_TEXTURE_CONSENSUS_RADIUS):
                validation.append(match)
    if (len(validation) < FINE_TEXTURE_MIN_VALIDATION_WINDOWS
            or len(validation) < FINE_TEXTURE_MIN_CONSENSUS_FRACTION * max(1, reliable)
            or len(validation) < 0.5 * len(validation_bounds)
            or not fine_texture_spread(validation, prepared.reference.shape)):
        return None
    if (checked < FINE_TEXTURE_MIN_VALIDATION_WINDOWS
            or np.any(np.ptp(np.asarray(compared_centers), axis=0) / np.asarray(prepared.reference.shape) < 0.35)):
        return None
    if (float(np.median(gains)) < FINE_TEXTURE_MIN_GAIN
            or np.mean(np.asarray(gains) >= 0.0) < 0.75
            or float(np.percentile(gains, 10)) < -0.015):
        return None
    if broad_pairs:
        broad_gain = np.asarray([b - a for a, b in broad_pairs])
        if (np.median(broad_gain) < -FINE_TEXTURE_MAX_LEGACY_LOSS
                or np.mean(broad_gain < -0.05) > 0.25):
            return None
    residuals = [math.hypot(m.row_shift - shift[0], m.col_shift - shift[1]) for m in validation]
    scatter = float(np.percentile(residuals, 90))
    score = float(np.median([m.score for m in validation]))
    evidence_fraction = (len(fit) + len(validation)) / max(1, len(windows))
    confidence = 100.0 * (0.35 * evidence_fraction + 0.35 * min(1.0, score / 0.65)
                         + 0.30 * max(0.0, 1.0 - scatter / FINE_TEXTURE_CONSENSUS_RADIUS))
    return FineTextureEvidence(float(shift[0]), float(shift[1]), len(fit), len(validation),
                               scatter, float(np.median(gains)), min(95.0, confidence),
                               validation, time.perf_counter() - start)


def fine_texture_should_try(state, assessment):
    if not FINE_TEXTURE_ENABLED or scipy_ndimage is None:
        return False
    strong_baseline = (
        state is not None and state.global_result.score >= 0.35
        and assessment is not None and assessment.status == "OK"
        and assessment.reliable_windows >= MIN_RELIABLE_WINDOWS
        and assessment.median_match_score >= 0.35
        and assessment.local_scatter_p90_px <= 0.75
    )
    return not strong_baseline


def fine_texture_assessment(evidence):
    matches = evidence.matches
    rows = np.asarray([m.row_shift - evidence.row_shift for m in matches])
    cols = np.asarray([m.col_shift - evidence.col_shift for m in matches])
    row, col = float(np.median(rows)), float(np.median(cols))
    return AssessmentResult(
        status="OK", threshold_profile="fine_native_heldout", attempt_number=1,
        systematic_row_px=row, systematic_col_px=col,
        systematic_magnitude_px=math.hypot(row, col),
        p90_residual_px=evidence.scatter_p90_px,
        local_scatter_p90_px=float(np.percentile(np.hypot(rows - row, cols - col), 90)),
        directional_coherence=qa_directional_coherence(rows, cols),
        confidence_0_100=evidence.confidence,
        reliable_windows=len(matches), evaluated_windows=FINE_TEXTURE_GRID ** 2 // 2,
        candidate_windows=FINE_TEXTURE_GRID ** 2 // 2,
        median_match_score=float(np.median([m.score for m in matches])),
        median_peak_margin=float(np.median([m.margin for m in matches])),
        median_peak_zscore=float(np.median([m.zscore for m in matches])),
        median_overlap_fraction=math.nan, search_edge_hit_fraction=0.0,
        unreliable_reason_counts="", window_vectors="",
        processing_seconds=evidence.processing_seconds,
    )


def fine_texture_final_residuals(reference, after, total_affine):
    """Measure final raster residuals; never substitute fitting scatter."""
    matches = [m for b, fit in fine_texture_windows(reference.shape) if not fit
               if (m := fine_texture_match(reference, after, b)) is not None]
    coherent, _ = fine_texture_consensus(matches)
    if (len(coherent) < FINE_TEXTURE_MIN_VALIDATION_WINDOWS
            or len(coherent) < FINE_TEXTURE_MIN_CONSENSUS_FRACTION * len(matches)
            or not fine_texture_spread(coherent, reference.shape)):
        return PairedResidualResult("FAILED", math.nan, math.nan, math.nan, 0,
                                    "FINE_TEXTURE_NO_SUPPORT", "fine_native_final",
                                    "Insufficient coherent fine-texture windows on final raster.")
    inverse = ~total_affine
    before_residuals, after_residuals = [], []
    for match in coherent:
        y0, y1, x0, x1 = match.bounds
        y, x = 0.5 * (y0 + y1 - 1), 0.5 * (x0 + x1 - 1)
        sx, sy = inverse * (x + 0.5, y + 0.5)
        before_residuals.append(math.hypot(y + 0.5 + match.row_shift - sy,
                                         x + 0.5 + match.col_shift - sx))
        after_residuals.append(math.hypot(match.row_shift, match.col_shift))
    before_p90, after_p90 = float(np.percentile(before_residuals, 90)), float(np.percentile(after_residuals, 90))
    return PairedResidualResult("OK", before_p90, after_p90, before_p90 - after_p90,
                                len(coherent), "FINE_TEXTURE_" + paired_residual_quality_label(len(coherent)),
                                "fine_native_final", "")


def prepare_registration_scene(
    landsat_cache: CachedRaster,
    eco_path: Path,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> RegistrationPreprocessing:
    eco_data, geometry = read_eco_scene(eco_path)
    reference_products = get_reference_grid_products(
        landsat_cache,
        geometry,
        local_executor=local_executor,
    )
    landsat_on_eco = reference_products.data
    nan_stats = nan_footprint_statistics(eco_data)
    dilation_pixels = automatic_eco_nan_dilation_pixels(
        nan_stats.internal_nan_percent
    )
    (
        eco_match_data,
        eco_gradient,
        eco_robust_gradient,
        _dilated_count,
    ) = prepare_moving_alignment_features(
        eco_data,
        dilation_pixels,
        invalid_mask=nan_stats.internal_nan_mask,
        footprint_mask=nan_stats.footprint_mask,
        local_executor=local_executor,
    )

    reference_valid = np.isfinite(landsat_on_eco)
    moving_valid = np.isfinite(eco_match_data)
    overlap = reference_valid & moving_valid
    overlap_pixels = int(np.count_nonzero(overlap))
    if overlap_pixels < 30:
        raise RuntimeError(
            "Reference and scene have fewer than 30 overlapping valid pixels after masking."
        )
    support_loss = support_loss_fraction(
        landsat_on_eco,
        eco_match_data,
        reference_valid_mask=reference_valid,
        moving_valid_mask=moving_valid,
        overlap_mask=overlap,
    )

    landsat_gradient = reference_products.gradient
    landsat_robust_gradient = reference_products.robust_gradient
    gradient_overlap_pixels = int(
        np.count_nonzero(np.isfinite(landsat_gradient) & np.isfinite(eco_gradient))
    )
    robust_gradient_overlap_pixels = int(
        np.count_nonzero(
            np.isfinite(landsat_robust_gradient)
            & np.isfinite(eco_robust_gradient)
        )
    )
    if gradient_overlap_pixels < 30 and robust_gradient_overlap_pixels < 30:
        raise RuntimeError("Alignment features have fewer than 30 overlapping valid pixels.")

    return RegistrationPreprocessing(
        reference=landsat_on_eco,
        moving=eco_data,
        geometry=geometry,
        nan_stats=nan_stats,
        eco_nan_dilation_pixels=dilation_pixels,
        legacy_reference=landsat_gradient,
        legacy_moving=eco_gradient,
        robust_reference=landsat_robust_gradient,
        robust_moving=eco_robust_gradient,
        support_loss=float(support_loss),
        overlap_pixels=overlap_pixels,
        gradient_overlap_pixels=gradient_overlap_pixels,
        robust_gradient_overlap_pixels=robust_gradient_overlap_pixels,
        fine_texture_masked_moving=(eco_match_data if (FINE_TEXTURE_ENABLED or LOCAL_TEXTURE_ENABLED or PRECISION_ENABLED) and scipy_ndimage is not None else None),
    )


def register_scene(
    landsat_cache: CachedRaster,
    eco_path: Path,
    coarse_downsample: int = COARSE_GLOBAL_DOWNSAMPLE,
    native_refine_radius_pixels: int = NATIVE_REFINE_RADIUS_PIXELS,
    force_robust: bool = False,
    method_prefix: str = "",
    preparation_cache: Optional[RegistrationPreparationCache] = None,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> RegistrationState:
    start = time.perf_counter()
    prepared = (
        preparation_cache.prepared
        if preparation_cache is not None
        else None
    )
    if prepared is None:
        prepared = prepare_registration_scene(
            landsat_cache,
            eco_path,
            local_executor=local_executor,
        )
        if preparation_cache is not None:
            preparation_cache.prepared = prepared

    eco_data = prepared.moving
    geometry = prepared.geometry
    landsat_on_eco = prepared.reference
    nan_stats = prepared.nan_stats
    dilation_pixels = prepared.eco_nan_dilation_pixels
    landsat_gradient = prepared.legacy_reference
    eco_gradient = prepared.legacy_moving
    landsat_robust_gradient = prepared.robust_reference
    eco_robust_gradient = prepared.robust_moving
    support_loss = prepared.support_loss
    overlap_pixels = prepared.overlap_pixels
    gradient_overlap_pixels = prepared.gradient_overlap_pixels
    robust_gradient_overlap_pixels = prepared.robust_gradient_overlap_pixels

    choice = estimate_translation_hybrid(
        landsat_gradient,
        eco_gradient,
        landsat_robust_gradient,
        eco_robust_gradient,
        support_loss,
        GLOBAL_MAX_ECO_SHIFT_PIXELS,
        native_refine_radius_pixels,
        coarse_downsample,
        logger=None,
        describe_shift=None,
        force_robust=force_robust,
        robust_result_cache=preparation_cache,
    )
    if method_prefix:
        choice.method = f"{method_prefix}; {choice.method}"
    global_result = choice.result
    downstream_key = (
        float(global_result.row_shift),
        float(global_result.col_shift),
        float(global_result.score),
        int(global_result.valid_pixels),
        bool(choice.cloud_tolerant),
    )
    cached_downstream = (
        preparation_cache.downstream_warp_model
        if preparation_cache is not None
        and preparation_cache.downstream_key == downstream_key
        else None
    )
    if cached_downstream is not None:
        cached_method = " -> ".join(
            (f"initial translation [{choice.method}]",)
            + preparation_cache.downstream_stage_tail
        )
        warp_model = WarpModel(
            row_offset=float(cached_downstream.row_offset),
            row_col_coef=float(cached_downstream.row_col_coef),
            row_row_coef=float(cached_downstream.row_row_coef),
            col_offset=float(cached_downstream.col_offset),
            col_col_coef=float(cached_downstream.col_col_coef),
            col_row_coef=float(cached_downstream.col_row_coef),
            control_points=int(cached_downstream.control_points),
            rmse_px=float(cached_downstream.rmse_px),
            mean_score=float(cached_downstream.mean_score),
            method=cached_method,
            score_q25=float(cached_downstream.score_q25),
            min_score=float(cached_downstream.min_score),
        )
        return RegistrationState(
            reference=prepared.reference,
            moving=prepared.moving,
            geometry=prepared.geometry,
            nan_stats=prepared.nan_stats,
            eco_nan_dilation_pixels=prepared.eco_nan_dilation_pixels,
            support_loss=float(prepared.support_loss),
            overlap_pixels=prepared.overlap_pixels,
            gradient_overlap_pixels=prepared.gradient_overlap_pixels,
            robust_gradient_overlap_pixels=prepared.robust_gradient_overlap_pixels,
            choice=choice,
            global_result=global_result,
            warp_model=warp_model,
            processing_seconds=time.perf_counter() - start,
        )

    warp_reference = landsat_robust_gradient if choice.cloud_tolerant else landsat_gradient
    warp_moving = eco_robust_gradient if choice.cloud_tolerant else eco_gradient

    initial_translation = WarpModel.from_translation(
        global_result,
        f"initial translation [{choice.method}]",
    )
    first_warp_candidate = estimate_linear_warp(
        warp_reference,
        warp_moving,
        global_result,
        NATIVE_REFINE_RADIUS_PIXELS,
        logger=None,
        cloud_tolerant=choice.cloud_tolerant,
        residual_search_radius=LOCAL_WARP_RESIDUAL_RADIUS,
        local_executor=local_executor,
    )
    first_stage_model = guarded_warp_choice(
        warp_reference,
        warp_moving,
        initial_translation,
        first_warp_candidate,
        warp_reference.shape,
        "Warp 1",
        logger=None,
        confidence_multiplier=1.0,
    )

    current_affine = first_stage_model.pixel_affine()
    stage_methods = [f"initial translation [{choice.method}]"]
    accepted_warps: List[WarpModel] = []
    if first_stage_model.is_translation_only():
        stage_methods.append("warp 1 rejected or unavailable")
    else:
        stage_methods.append("guarded warp 1 accepted")
        accepted_warps.append(first_stage_model)

    if ENABLE_POST_WARP_REFINEMENT:
        legacy_current, robust_current = warp_array_continuous_pair(
            eco_gradient,
            eco_robust_gradient,
            current_affine,
        )
        residual_support_loss = support_loss_fraction(
            landsat_robust_gradient,
            robust_current,
        )
        residual_model = WarpModel.from_pixel_affine(
            Affine.identity(),
            "residual translation skipped",
        )
        try:
            residual_choice = estimate_translation_hybrid(
                landsat_gradient,
                legacy_current,
                landsat_robust_gradient,
                robust_current,
                residual_support_loss,
                POST_WARP_TRANSLATION_MAX_PIXELS,
                POST_WARP_TRANSLATION_MAX_PIXELS,
                POST_WARP_TRANSLATION_DOWNSAMPLE,
                logger=None,
                describe_shift=None,
            )
            residual_model = guarded_residual_translation(
                landsat_robust_gradient,
                robust_current,
                residual_choice.result,
                "Residual translation",
                logger=None,
            )
        except Exception:
            residual_model = WarpModel.from_pixel_affine(
                Affine.identity(),
                "residual translation unavailable",
            )

        residual_magnitude = math.hypot(
            residual_model.row_offset,
            residual_model.col_offset,
        )
        if residual_magnitude >= RESIDUAL_TRANSLATION_MIN_MAGNITUDE_PIXELS:
            current_affine = compose_pixel_affines(
                current_affine,
                residual_model.pixel_affine(),
            )
            stage_methods.append(
                f"residual translation accepted ({residual_magnitude:.2f}px)"
            )
            legacy_current, robust_current = warp_array_continuous_pair(
                eco_gradient,
                eco_robust_gradient,
                current_affine,
            )
        else:
            stage_methods.append("residual translation rejected or negligible")

        second_reference = (
            landsat_robust_gradient if choice.cloud_tolerant else landsat_gradient
        )
        second_moving = robust_current if choice.cloud_tolerant else legacy_current
        second_baseline = WarpModel.from_pixel_affine(
            Affine.identity(),
            "second-warp identity baseline",
        )
        try:
            second_zero_score, second_zero_count = zero_shift_ncc(
                second_reference,
                second_moving,
            )
            second_zero = ShiftResult(0.0, 0.0, second_zero_score, second_zero_count)
            second_warp_candidate = estimate_linear_warp(
                second_reference,
                second_moving,
                second_zero,
                SECOND_WARP_MAX_RESIDUAL_PIXELS,
                logger=None,
                cloud_tolerant=choice.cloud_tolerant,
                residual_search_radius=SECOND_WARP_MAX_RESIDUAL_PIXELS,
                local_executor=local_executor,
            )
            second_stage_model = guarded_warp_choice(
                second_reference,
                second_moving,
                second_baseline,
                second_warp_candidate,
                second_reference.shape,
                "Warp 2",
                logger=None,
                confidence_multiplier=SECOND_WARP_CONFIDENCE_MULTIPLIER,
            )
        except Exception:
            second_stage_model = second_baseline

        if second_stage_model.is_translation_only():
            stage_methods.append("warp 2 rejected or unavailable")
        else:
            current_affine = compose_pixel_affines(
                current_affine,
                second_stage_model.pixel_affine(),
            )
            stage_methods.append("guarded warp 2 accepted")
            accepted_warps.append(second_stage_model)

    warp_model = summarize_composite_model(
        current_affine,
        stage_methods,
        accepted_warps,
    )
    if preparation_cache is not None:
        preparation_cache.downstream_key = downstream_key
        preparation_cache.downstream_warp_model = warp_model
        preparation_cache.downstream_stage_tail = tuple(stage_methods[1:])
    return RegistrationState(
        reference=landsat_on_eco,
        moving=eco_data,
        geometry=geometry,
        nan_stats=nan_stats,
        eco_nan_dilation_pixels=dilation_pixels,
        support_loss=float(support_loss),
        overlap_pixels=overlap_pixels,
        gradient_overlap_pixels=gradient_overlap_pixels,
        robust_gradient_overlap_pixels=robust_gradient_overlap_pixels,
        choice=choice,
        global_result=global_result,
        warp_model=warp_model,
        processing_seconds=time.perf_counter() - start,
    )


def copy_raster_metadata(src, dst) -> None:
    dataset_tags = src.tags()
    if dataset_tags:
        dst.update_tags(**dataset_tags)
    for band_index in range(1, src.count + 1):
        band_tags = src.tags(band_index)
        if band_tags:
            dst.update_tags(band_index, **band_tags)
        description = src.descriptions[band_index - 1]
        if description:
            dst.set_band_description(band_index, description)
    try:
        dst.colorinterp = src.colorinterp
    except Exception:
        pass
    try:
        dst.scales = src.scales
        dst.offsets = src.offsets
    except Exception:
        pass
    try:
        dst.units = src.units
    except Exception:
        pass


def reproject_array_with_pixel_affine(
    source_data: np.ndarray,
    geometry: RasterGeometry,
    pixel_affine: Affine,
    zero_is_nodata: Optional[bool] = None,
) -> np.ndarray:
    if zero_is_nodata is None:
        zero_is_nodata = ZERO_IS_NODATA
    if (
        pixel_affine.a == 1.0
        and pixel_affine.b == 0.0
        and pixel_affine.c == 0.0
        and pixel_affine.d == 0.0
        and pixel_affine.e == 1.0
        and pixel_affine.f == 0.0
    ):
        destination = np.asarray(source_data, dtype=np.float32).copy()
        destination[~np.isfinite(destination)] = np.nan
        if zero_is_nodata:
            destination[destination == 0] = np.nan
        return destination
    destination = np.full(
        (geometry.height, geometry.width),
        np.nan,
        dtype=np.float32,
    )
    reproject(
        source=np.asarray(source_data, dtype=np.float32),
        destination=destination,
        src_transform=geometry.transform * pixel_affine,
        src_crs=geometry.crs,
        src_nodata=np.nan,
        dst_transform=geometry.transform,
        dst_crs=geometry.crs,
        dst_nodata=np.nan,
        init_dest_nodata=True,
        resampling=Resampling.nearest,
        num_threads=ACTIVE_GDAL_WARP_THREADS,
        warp_mem_limit=GDAL_WARP_MEMORY_MB,
    )
    destination[~np.isfinite(destination)] = np.nan
    if zero_is_nodata:
        destination[destination == 0] = np.nan
    return destination


def regional_source_coordinates(base_affine, local_warp, rows, cols):
    if local_warp is None:
        dy, dx = 0.0, 0.0
    else:
        dy, dx = local_warp.displacement(rows, cols)
    inverse = ~base_affine
    yy, xx = rows + 0.5 - dy, cols + 0.5 - dx
    return (inverse.d * xx + inverse.e * yy + inverse.f - 0.5,
            inverse.a * xx + inverse.b * yy + inverse.c - 0.5)


def resample_original_subpixel(
    source_data, base_affine, local_warp=None, zero_is_nodata: Optional[bool] = None,
):
    """Single-pass bilinear sampling; all contributing neighbors must be valid."""
    if zero_is_nodata is None:
        zero_is_nodata = ZERO_IS_NODATA
    source = np.asarray(source_data, dtype=np.float32)
    valid = np.isfinite(source)
    if zero_is_nodata:
        valid &= source != 0
    if local_warp is None:
        return warp_array_continuous(np.where(valid, source, np.nan), base_affine)
    if scipy_ndimage is None:
        raise RuntimeError("SciPy is required for local-warp output.")
    values = np.where(valid, source, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    output = np.full(source.shape, np.nan, np.float32)
    for y0 in range(0, source.shape[0], 128):
        y1 = min(source.shape[0], y0 + 128)
        for x0 in range(0, source.shape[1], 2048):
            x1 = min(source.shape[1], x0 + 2048)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            sy, sx = regional_source_coordinates(base_affine, local_warp, yy, xx)
            coords = np.asarray([sy, sx])
            sampled = scipy_ndimage.map_coordinates(values, coords, order=1,
                                                    mode="constant", cval=0.0, prefilter=False)
            support = scipy_ndimage.map_coordinates(weight, coords, order=1,
                                                    mode="constant", cval=0.0, prefilter=False)
            sampled[support < 0.9999] = np.nan
            output[y0:y1, x0:x1] = sampled
    return output


def regional_forward_shift(base_affine, local_warp, row, col):
    """Source-center displacement via contraction of the inverse warp."""
    bx, by = base_affine * (float(col), float(row))
    x, y = bx, by
    for _ in range(40):
        dy, dx = local_warp.displacement(np.asarray(y - 0.5), np.asarray(x - 0.5))
        nx, ny = bx + float(dx), by + float(dy)
        if math.hypot(nx - x, ny - y) < 1e-8:
            return ny - row, nx - col
        x, y = nx, ny
    raise RuntimeError("Regional forward coordinate did not converge.")


def regional_final_residuals(reference, after, base_affine, local_warp):
    before_values, after_values = [], []
    for bounds in local_warp.validation_bounds:
        match = local_texture_match(reference, after, bounds)
        if match is None:
            continue
        y0, y1, x0, x1 = bounds
        y, x = 0.5 * (y0 + y1 - 1), 0.5 * (x0 + x1 - 1)
        sy, sx = regional_source_coordinates(base_affine, local_warp, np.asarray(y), np.asarray(x))
        before_values.append(math.hypot(y + match.row_shift - float(sy), x + match.col_shift - float(sx)))
        after_values.append(math.hypot(match.row_shift, match.col_shift))
    if len(after_values) < 3:
        return PairedResidualResult("FAILED", math.nan, math.nan, math.nan, len(after_values),
                                    "LOCAL_WARP_NO_SUPPORT", "local_native_final",
                                    "Insufficient reliable local windows on final raster.")
    before_p90, after_p90 = float(np.percentile(before_values, 90)), float(np.percentile(after_values, 90))
    return PairedResidualResult("OK", before_p90, after_p90, before_p90 - after_p90,
                                len(after_values), "LOCAL_WARP_" + paired_residual_quality_label(len(after_values)),
                                "local_native_final", "")


def write_final_affine_raster(
    input_path: Path,
    output_path: Path,
    pixel_affine: Affine,
    first_band_data: Optional[np.ndarray] = None,
    local_warp=None,
    subpixel_resampling: bool = True,
    zero_is_nodata: Optional[bool] = None,
) -> None:
    if zero_is_nodata is None:
        zero_is_nodata = ZERO_IS_NODATA
    if output_path.exists() and not OVERWRITE_EXISTING:
        raise FileExistsError(f"Output exists: {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with rasterio.open(input_path) as src:
            require_crs(src, input_path)
            profile = src.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.update(
                driver="GTiff",
                dtype="float32",
                count=src.count,
                nodata=np.nan,
                compress="deflate",
                zlevel=TIFF_DEFLATE_LEVEL,
                NUM_THREADS=str(ACTIVE_TIFF_THREADS),
                predictor=3,
                tiled=True,
                BIGTIFF="IF_SAFER",
            )
            geometry = RasterGeometry(src.transform, src.crs, src.width, src.height)
            with rasterio.open(temporary_path, "w", **profile) as dst:
                for band_index in range(1, src.count + 1):
                    if band_index == 1 and first_band_data is not None:
                        destination = np.asarray(first_band_data, dtype=np.float32)
                        if destination.shape != (src.height, src.width):
                            raise RuntimeError(
                                "The precomputed first-band output does not match the source grid."
                            )
                    else:
                        source = src.read(band_index, masked=True).astype(np.float32)
                        source_data = source.filled(np.nan).astype(np.float32, copy=False)
                        invalid = ~np.isfinite(source_data)
                        if zero_is_nodata:
                            invalid |= source_data == 0
                        if src.nodata is not None:
                            try:
                                nodata_value = float(src.nodata)
                                if math.isnan(nodata_value):
                                    invalid |= np.isnan(source_data)
                                else:
                                    invalid |= source_data == nodata_value
                            except Exception:
                                pass
                        source_data[invalid] = np.nan
                        if local_warp is not None or subpixel_resampling:
                            destination = resample_original_subpixel(
                                source_data, pixel_affine, local_warp,
                                zero_is_nodata=zero_is_nodata)
                        else:
                            destination = reproject_array_with_pixel_affine(
                                source_data, geometry, pixel_affine,
                                zero_is_nodata=zero_is_nodata)
                    dst.write(destination, band_index)
                copy_raster_metadata(src, dst)
                if local_warp is not None:
                    dst.update_tags(TIR_RECAST_TRANSFORM="base affine plus regional inverse displacement",
                                    TIR_RECAST_LOCAL_WARP=local_warp.serialize(),
                                    TIR_RECAST_RESAMPLING="bilinear")
                elif subpixel_resampling:
                    dst.update_tags(TIR_RECAST_RESAMPLING="bilinear")
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def remove_previous_scene_outputs(output_folder: Path, input_path: Path) -> None:
    if not CLEAN_PREVIOUS_SCENE_OUTPUTS or not output_folder.exists():
        return
    scene_stem = processed_scene_stem(input_path)
    desired_prefix = f"{scene_stem}_"
    legacy_prefix = f"{input_path.stem}_"

    for path in output_folder.iterdir():
        if not path.is_file() or path.suffix.casefold() not in {".tif", ".tiff"}:
            continue
        name = path.stem
        if name.endswith("_polished" + ECO_ERROR_SUFFIX):
            name = name[:-len(ECO_ERROR_SUFFIX)]
        if not name.endswith("_polished"):
            continue

        body = name[:-len("_polished")]
        remove = False
        if body.startswith(desired_prefix):
            parts = body[len(desired_prefix):].split("_")
            remove = (
                len(parts) in {1, 2}
                and all(part.isdigit() and len(part) == 2 for part in parts)
                and all(0 <= int(part) <= 99 for part in parts)
            )
        if not remove and body.startswith(legacy_prefix):
            parts = body[len(legacy_prefix):].split("_")
            remove = (
                1 <= len(parts) <= 2
                and all(part.isdigit() for part in parts)
            )
        if remove:
            path.unlink()


def output_path_for_confidence(
    output_folder: Path,
    input_path: Path,
    confidence_percent: int,
    attempt_code: int = 0,
) -> Path:
    confidence_code = confidence_filename_code(confidence_percent)
    attempt_code = max(0, min(99, int(attempt_code)))
    if attempt_code > 0:
        filename = (
            f"{processed_scene_stem(input_path)}_{attempt_code:02d}_"
            f"{confidence_code:02d}_polished{input_path.suffix}"
        )
    else:
        filename = (
            f"{processed_scene_stem(input_path)}_{confidence_code:02d}"
            f"_polished{input_path.suffix}"
        )
    return output_folder / filename


def write_matching_error_raster(
    input_path: Path,
    output_path: Path,
    pixel_affine: Affine,
    corrections_applied: bool,
    row: Dict[str, object],
    local_warp=None,
) -> None:
    """Replay the written processed correction; isolate companion failures."""
    start = time.perf_counter()
    row.update({
        "err_input_file": "", "err_output_file": "", "err_status": "",
        "err_resampling": "", "err_error": "", "err_write_seconds": 0.0,
    })
    temporary_path = None
    try:
        if not output_path.name or row.get("status") == "OUTPUT_FAILED":
            row["err_status"] = "SKIPPED_PROCESSED_OUTPUT_FAILED"
            return
        error_path = matching_error_raster(input_path)
        if error_path is None:
            row["err_status"] = "MISSING"
            row["err_error"] = f"No matching {ECO_ERROR_SUFFIX} TIFF for {input_path.name}"
            return
        row["err_input_file"] = error_path.name
        with rasterio.open(input_path) as processed, rasterio.open(error_path) as error:
            mismatches = []
            if processed.crs is None or error.crs is None or error.crs != processed.crs:
                mismatches.append("CRS")
            if (error.height, error.width) != (processed.height, processed.width):
                mismatches.append("dimensions")
            if error.transform != processed.transform:
                mismatches.append("pixel transform")
            if mismatches:
                row["err_status"] = "GRID_MISMATCH"
                row["err_error"] = (
                    f"{error_path.name} and {input_path.name} differ in "
                    + ", ".join(mismatches)
                    + "; error output was not written."
                )
                return
        error_output = output_path.with_name(
            output_path.stem + ECO_ERROR_SUFFIX + output_path.suffix)
        if corrections_applied:
            subpixel_resampling = row.get("output_resampling") == "bilinear"
            write_final_affine_raster(
                error_path, error_output, pixel_affine,
                local_warp=local_warp, subpixel_resampling=subpixel_resampling,
                zero_is_nodata=False,
            )
            row["err_status"] = "OK"
            row["err_resampling"] = (
                "bilinear" if local_warp is not None or subpixel_resampling else "nearest")
        else:
            if error_output.exists() and not OVERWRITE_EXISTING:
                raise FileExistsError(f"Output exists: {error_output.name}")
            error_output.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = error_output.with_name(
                f".{error_output.stem}.tmp{error_output.suffix}")
            shutil.copy2(error_path, temporary_path)
            temporary_path.replace(error_output)
            row["err_status"] = "PASSTHROUGH"
            row["err_resampling"] = "unchanged"
        row["err_output_file"] = error_output.name
    except Exception as exc:
        row["err_status"] = "OUTPUT_FAILED"
        row["err_error"] = str(exc)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        row["err_write_seconds"] = time.perf_counter() - start


def affine_components(pixel_affine: Affine) -> Dict[str, float]:
    return {
        "total_warp_row_offset_px": float(pixel_affine.f),
        "total_warp_row_col_coef": float(pixel_affine.d),
        "total_warp_row_row_coef": float(pixel_affine.e - 1.0),
        "total_warp_col_offset_px": float(pixel_affine.c),
        "total_warp_col_col_coef": float(pixel_affine.a - 1.0),
        "total_warp_col_row_coef": float(pixel_affine.b),
        "total_pixel_affine_a": float(pixel_affine.a),
        "total_pixel_affine_b": float(pixel_affine.b),
        "total_pixel_affine_c": float(pixel_affine.c),
        "total_pixel_affine_d": float(pixel_affine.d),
        "total_pixel_affine_e": float(pixel_affine.e),
        "total_pixel_affine_f": float(pixel_affine.f),
    }



def finalize_correction_record(
    row: Dict[str, object],
    applied_affine: Affine,
    geometry: Optional[RasterGeometry] = None,
    local_warp=None,
) -> None:
    """Record the written transform without changing raster processing."""
    row["correction_record_error"] = ""
    final_fields = (
        "total_translation_azimuth_deg", "total_translation_distance_m",
        "total_center_row_shift_px", "total_center_col_shift_px",
    )
    if not row.get("output_file") or row.get("status") == "OUTPUT_FAILED":
        for key in (*final_fields, *affine_components(Affine.identity())):
            row[key] = ""
        row["corrections_applied"] = 0
        row["residual_after_p90_px"] = ""
        row["residual_improvement_p90_px"] = ""
        row["residual_quality"] = "OUTPUT_FAILED"
        row["transform_model"] = "OUTPUT_FAILED"
        row["local_warp_definition"] = ""
        for name in "abcdef":
            row[f"base_pixel_affine_{name}"] = ""
        return

    applied = bool(row.get("corrections_applied"))
    affine = applied_affine if applied else Affine.identity()
    row.update(affine_components(affine))
    row["transform_model"] = "AFFINE_PLUS_LOCAL_FIELD" if applied and local_warp is not None else "AFFINE" if applied else "IDENTITY"
    for name in "abcdef":
        row[f"base_pixel_affine_{name}"] = float(getattr(affine, name))
    if applied and local_warp is not None:
        row["local_warp_definition"] = local_warp.serialize()
        for key in affine_components(affine):
            row[key] = ""
    else:
        row["local_warp_definition"] = ""
    for key in final_fields:
        row[key] = "" if applied else 0.0
    try:
        if geometry is not None:
            row["source_crs"] = geometry.crs.to_string() if geometry.crs else ""
            row["source_width_px"] = int(geometry.width)
            row["source_height_px"] = int(geometry.height)
            row["translation_anchor_col_px"] = geometry.width / 2.0
            row["translation_anchor_row_px"] = geometry.height / 2.0
            basis, north, units, factor, geod = native_crs_measurement(geometry.crs.to_wkt())
            row["translation_distance_basis"] = basis
            row["translation_azimuth_reference"] = north
            row["source_xy_units"] = units
            row["source_units_to_m"] = factor if geod is None else ""
        if not applied:
            return
        if geometry is None or geometry.crs is None:
            raise ValueError("Source geometry is unavailable for the applied transform")

        col = geometry.width / 2.0
        rr = geometry.height / 2.0
        dy, dx = (regional_forward_shift(affine, local_warp, rr, col) if local_warp is not None
                  else affine_shift_at(affine, rr, col))
        row["total_center_row_shift_px"] = dy
        row["total_center_col_shift_px"] = dx
        # Use the same composed source geotransform as the final raster writer.
        x0, y0 = geometry.transform * (col, rr)
        x1, y1 = (geometry.transform * (col + dx, rr + dy) if local_warp is not None
                  else (geometry.transform * affine) * (col, rr))
        movement = native_displacement_metrics(geometry.crs, x0, y0, x1, y1)
        row["total_translation_azimuth_deg"] = movement["azimuth_deg"]
        row["total_translation_distance_m"] = movement["distance_m"]
    except Exception as exc:
        # A reporting failure must not replace a successfully written raster.
        row["correction_record_error"] = str(exc)


def candidate_affine_components(pixel_affine: Affine) -> Dict[str, float]:
    return {
        "candidate_total_pixel_affine_a": float(pixel_affine.a),
        "candidate_total_pixel_affine_b": float(pixel_affine.b),
        "candidate_total_pixel_affine_c": float(pixel_affine.c),
        "candidate_total_pixel_affine_d": float(pixel_affine.d),
        "candidate_total_pixel_affine_e": float(pixel_affine.e),
        "candidate_total_pixel_affine_f": float(pixel_affine.f),
    }


def registration_confidence_0_100(state: RegistrationState) -> float:
    score = float(state.global_result.score)
    score_component = float(np.clip((score + 0.05) / 0.35, 0.0, 1.0))
    support_component = float(np.clip(1.0 - state.support_loss / 0.75, 0.0, 1.0))
    footprint = max(1, state.nan_stats.footprint_pixels)
    overlap_ratio = state.overlap_pixels / footprint
    overlap_component = float(np.clip(overlap_ratio / 0.35, 0.0, 1.0))

    if state.warp_model.is_translation_only():
        warp_component = 1.0
    else:
        control_component = min(1.0, state.warp_model.control_points / 12.0)
        rmse_component = float(
            np.clip(1.0 - state.warp_model.rmse_px / 3.0, 0.0, 1.0)
        )
        mean_component = (
            float(np.clip((state.warp_model.mean_score - 0.03) / 0.27, 0.0, 1.0))
            if math.isfinite(state.warp_model.mean_score)
            else 0.0
        )
        warp_component = (
            0.35 * control_component
            + 0.35 * rmse_component
            + 0.30 * mean_component
        )

    value = 100.0 * (
        0.35 * score_component
        + 0.25 * support_component
        + 0.25 * overlap_component
        + 0.15 * warp_component
    )
    return float(np.clip(value, 0.0, 100.0))


def geometry_confidence(
    footprint_pixels: int,
    grid_pixels: int,
    total_shift_px: float,
) -> Dict[str, float]:
    footprint_pixels = max(0, int(footprint_pixels))
    grid_pixels = max(1, int(grid_pixels))
    hull_fraction = footprint_pixels / grid_pixels
    equivalent_radius = math.sqrt(footprint_pixels / math.pi) if footprint_pixels else 0.0
    shift_to_radius = (
        float(total_shift_px) / equivalent_radius
        if equivalent_radius > 0.0
        else math.inf
    )
    area_component = float(
        np.clip(
            math.sqrt(
                max(0.0, hull_fraction)
                / max(1e-12, CONFIDENCE_HULL_FRACTION_SATURATION)
            ),
            0.0,
            1.0,
        )
    )
    shift_component = (
        math.exp(-CONFIDENCE_SHIFT_DECAY_PER_HULL_RADIUS * shift_to_radius)
        if math.isfinite(shift_to_radius)
        else 0.0
    )
    combined = float(np.clip(area_component * shift_component, 0.0, 1.0))
    return {
        "convex_hull_area_pixels": float(footprint_pixels),
        "convex_hull_fraction": float(hull_fraction),
        "equivalent_hull_radius_px": float(equivalent_radius),
        "translation_to_hull_radius_ratio": float(shift_to_radius),
        "hull_area_component_0_1": float(area_component),
        "translation_component_0_1": float(shift_component),
        "geometry_confidence_0_100": 100.0 * combined,
        "geometry_factor_0_1": combined,
    }


def final_confidence_percent(
    assessment_confidence: float,
    registration_confidence: float,
    geometry_factor: float,
    failed: bool,
) -> Tuple[int, float]:
    if failed:
        return 0, 0.0
    evidence = np.clip(
        0.62 * (assessment_confidence / 100.0)
        + 0.38 * (registration_confidence / 100.0),
        0.0,
        1.0,
    )
    raw = 100.0 * float(evidence) * (
        float(np.clip(geometry_factor, 0.0, 1.0))
        ** CONFIDENCE_GEOMETRY_EXPONENT
    )
    return max(1, min(100, int(round(raw)))), float(np.clip(raw, 0.0, 100.0))


PIPELINE_ROW_FIELDS = (
    'input_index',
    'timestamp',
    'status',
    'failure_stage',
    'input_file_original',
    'input_file',
    'output_file',
    'reference_file',
    'processing_attempt_code',
    'high_resolution_retry_triggered',
    'corrections_applied',
    'selected_coarse_downsample_factor',
    'selected_native_search_radius_pixels',
    'selected_force_robust_matcher',
    'retry_confidence_threshold',
    'initial_confidence_percent',
    'initial_confidence_raw_0_100',
    'initial_attempt_status',
    'initial_attempt_failure_stage',
    'initial_attempt_error',
    'initial_attempt_seconds',
    'retry_attempt_seconds',
    'confidence_percent',
    'confidence_raw_0_100',
    'confidence_filename_code_00_99',
    'assessment_confidence_0_100',
    'registration_confidence_0_100',
    'geometry_confidence_0_100',
    'geometry_factor_0_1',
    'convex_hull_area_pixels',
    'convex_hull_fraction',
    'equivalent_hull_radius_px',
    'translation_to_hull_radius_ratio',
    'hull_area_component_0_1',
    'translation_component_0_1',
    'total_translation_distance_m',
    'total_translation_azimuth_deg',
    'total_center_row_shift_px',
    'total_center_col_shift_px',
    'candidate_total_translation_distance_m',
    'candidate_total_translation_azimuth_deg',
    'candidate_total_center_row_shift_px',
    'candidate_total_center_col_shift_px',
    'candidate_total_pixel_affine_a',
    'candidate_total_pixel_affine_b',
    'candidate_total_pixel_affine_c',
    'candidate_total_pixel_affine_d',
    'candidate_total_pixel_affine_e',
    'candidate_total_pixel_affine_f',
    'register_center_row_shift_px',
    'register_center_col_shift_px',
    'polish_residual_row_shift_px',
    'polish_residual_col_shift_px',
    'total_warp_row_offset_px',
    'total_warp_row_col_coef',
    'total_warp_row_row_coef',
    'total_warp_col_offset_px',
    'total_warp_col_col_coef',
    'total_warp_col_row_coef',
    'total_pixel_affine_a',
    'total_pixel_affine_b',
    'total_pixel_affine_c',
    'total_pixel_affine_d',
    'total_pixel_affine_e',
    'total_pixel_affine_f',
    'register_pixel_affine_a',
    'register_pixel_affine_b',
    'register_pixel_affine_c',
    'register_pixel_affine_d',
    'register_pixel_affine_e',
    'register_pixel_affine_f',
    'registration_method',
    'global_match_method',
    'registration_correlation_score',
    'registration_valid_pixels',
    'warp_control_points',
    'warp_rmse_px',
    'warp_mean_score',
    'warp_q25_score',
    'assessment_threshold_profile',
    'assessment_attempt_number',
    'assessment_reliable_windows',
    'assessment_evaluated_windows',
    'assessment_candidate_windows',
    'assessment_systematic_residual_px',
    'assessment_p90_residual_px',
    'assessment_local_scatter_p90_px',
    'assessment_directional_coherence_0_1',
    'assessment_median_match_score',
    'assessment_median_peak_margin',
    'assessment_median_peak_zscore',
    'assessment_median_overlap_fraction',
    'assessment_search_edge_hit_fraction',
    'assessment_unreliable_reason_counts',
    'internal_nan_percent',
    'support_loss_fraction',
    'eco_nan_dilation_pixels',
    'overlap_pixels',
    'gradient_overlap_pixels',
    'robust_gradient_overlap_pixels',
    'register_seconds',
    'assess_seconds',
    'polish_seconds',
    'total_processing_seconds',
    'error',
    'assessment_window_vectors',
    'residual_before_p90_px',
    'residual_after_p90_px',
    'residual_improvement_p90_px',
    'residual_paired_windows',
    'residual_quality',
)


def remove_previous_batch_outputs(
    output_folder: Path,
    input_paths: Sequence[Path],
) -> None:
    if not CLEAN_PREVIOUS_SCENE_OUTPUTS or not output_folder.exists():
        return
    desired_stems = {processed_scene_stem(path) for path in input_paths}
    legacy_stems = {path.stem for path in input_paths}
    for path in list(output_folder.iterdir()):
        if not path.is_file():
            continue
        if path.name == LEGACY_OUTPUT_CSV_NAME:
            path.unlink(missing_ok=True)
            continue
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        stem = path.stem
        if stem.endswith("_polished" + ECO_ERROR_SUFFIX):
            stem = stem[:-len(ECO_ERROR_SUFFIX)]
        remove = False
        if stem.endswith("_polished"):
            confidence_prefix = stem[: -len("_polished")]
            if (
                len(confidence_prefix) >= 3
                and confidence_prefix[-3] == "_"
                and len(confidence_prefix[-2:]) == 2
                and confidence_prefix[-2:].isdigit()
            ):
                base = confidence_prefix[:-3]
                remove = base in desired_stems
                if not remove and base.endswith("_01"):
                    remove = base[:-3] in desired_stems

            if not remove:
                for offset, character in enumerate(stem):
                    if character == "_" and stem[:offset] in legacy_stems:
                        remove = True
                        break

        if remove:
            path.unlink(missing_ok=True)


def blank_row(index: int, input_path: Path, reference_name: str) -> Dict[str, object]:
    row: Dict[str, object] = {field: "" for field in PIPELINE_ROW_FIELDS}
    row.update(
        {
            "input_index": index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_file_original": input_path.name,
            "input_file": input_path.name,
            "reference_file": reference_name,
            "processing_attempt_code": 0,
            "high_resolution_retry_triggered": 0,
            "corrections_applied": 0,
            "selected_coarse_downsample_factor": COARSE_GLOBAL_DOWNSAMPLE,
            "selected_native_search_radius_pixels": NATIVE_REFINE_RADIUS_PIXELS,
            "selected_force_robust_matcher": 0,
            "retry_confidence_threshold": LOW_CONFIDENCE_RETRY_THRESHOLD,
            "confidence_percent": 0,
            "confidence_raw_0_100": 0.0,
            "confidence_filename_code_00_99": 0,
            "residual_paired_windows": 0,
            "residual_quality": "NOT_RUN",
        }
    )
    return row


def update_affine_row(row: Dict[str, object], prefix: str, affine: Affine) -> None:
    row[f"{prefix}_pixel_affine_a"] = float(affine.a)
    row[f"{prefix}_pixel_affine_b"] = float(affine.b)
    row[f"{prefix}_pixel_affine_c"] = float(affine.c)
    row[f"{prefix}_pixel_affine_d"] = float(affine.d)
    row[f"{prefix}_pixel_affine_e"] = float(affine.e)
    row[f"{prefix}_pixel_affine_f"] = float(affine.f)


@dataclass
class AttemptOutcome:
    row: Dict[str, object]
    registration_state: Optional[RegistrationState]
    assessment: Optional[AssessmentResult]
    register_affine: Affine
    total_affine: Affine
    registration_confidence: float
    geometry: Dict[str, float]
    confidence_percent: int
    confidence_raw: float
    failure_stage: str
    errors: List[str]
    processing_seconds: float


def empty_geometry_metrics() -> Dict[str, float]:
    return {
        "convex_hull_area_pixels": math.nan,
        "convex_hull_fraction": math.nan,
        "equivalent_hull_radius_px": math.nan,
        "translation_to_hull_radius_ratio": math.nan,
        "hull_area_component_0_1": 0.0,
        "translation_component_0_1": 0.0,
        "geometry_confidence_0_100": 0.0,
        "geometry_factor_0_1": 0.0,
    }


def release_registration_state(state: Optional[RegistrationState]) -> None:
    if state is None:
        return
    state.reference = np.empty((0, 0), dtype=np.float32)
    state.moving = np.empty((0, 0), dtype=np.float32)


def run_pipeline_attempt(
    index: int,
    total: int,
    input_path: Path,
    reference_name: str,
    reference_cache: Optional[CachedRaster],
    setup_error: str,
    attempt_code: int,
    coarse_downsample: int,
    native_search_radius_pixels: int,
    force_robust: bool,
    logger: Optional[ConsoleLog],
    qa_reference_features: Optional[
        Dict[Tuple[int, int, int, int], QAStructuralFeatureVariants]
    ] = None,
    registration_preparation_cache: Optional[RegistrationPreparationCache] = None,
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> AttemptOutcome:
    attempt_start = time.perf_counter()
    row = blank_row(index, input_path, reference_name)
    row.update(
        {
            "processing_attempt_code": int(attempt_code),
            "selected_coarse_downsample_factor": int(coarse_downsample),
            "selected_native_search_radius_pixels": int(native_search_radius_pixels),
            "selected_force_robust_matcher": int(bool(force_robust)),
            "retry_confidence_threshold": LOW_CONFIDENCE_RETRY_THRESHOLD,
        }
    )
    errors: List[str] = []
    failure_stage = ""
    registration_state: Optional[RegistrationState] = None
    assessment: Optional[AssessmentResult] = None
    register_affine = Affine.identity()
    total_affine = Affine.identity()
    registration_confidence = 0.0
    geometry = empty_geometry_metrics()

    attempt_label = (
        "standard 16x pass"
        if attempt_code == 0
        else "retry 01: no 16x catch; full-range native + forced robust"
    )
    if logger is not None:
        logger.info(f"    Attempt {attempt_code:02d} | {attempt_label}")

    if setup_error:
        failure_stage = "setup"
        errors.append(setup_error)
        if logger is not None:
            logger.info(f"    Register  FAILED | setup: {setup_error}")
    else:
        register_start = time.perf_counter()
        try:
            if reference_cache is None:
                raise RuntimeError("Reference cache is unavailable.")
            registration_state = register_scene(
                reference_cache,
                input_path,
                coarse_downsample=coarse_downsample,
                native_refine_radius_pixels=native_search_radius_pixels,
                force_robust=force_robust,
                method_prefix=("high-resolution retry 01" if attempt_code else ""),
                preparation_cache=registration_preparation_cache,
                local_executor=local_executor,
            )
            row["register_seconds"] = registration_state.processing_seconds
            register_affine = registration_state.warp_model.pixel_affine()
            total_affine = register_affine
            registration_confidence = registration_confidence_0_100(
                registration_state
            )
            center_row = registration_state.geometry.height / 2.0
            center_col = registration_state.geometry.width / 2.0
            register_row, register_col = affine_shift_at(
                register_affine,
                center_row,
                center_col,
            )
            row["register_center_row_shift_px"] = register_row
            row["register_center_col_shift_px"] = register_col
            update_affine_row(row, "register", register_affine)
            row.update(
                {
                    "registration_confidence_0_100": registration_confidence,
                    "registration_method": registration_state.warp_model.method,
                    "global_match_method": registration_state.choice.method,
                    "registration_correlation_score": registration_state.global_result.score,
                    "registration_valid_pixels": registration_state.global_result.valid_pixels,
                    "warp_control_points": registration_state.warp_model.control_points,
                    "warp_rmse_px": registration_state.warp_model.rmse_px,
                    "warp_mean_score": registration_state.warp_model.mean_score,
                    "warp_q25_score": registration_state.warp_model.score_q25,
                    "internal_nan_percent": registration_state.nan_stats.internal_nan_percent,
                    "support_loss_fraction": registration_state.support_loss,
                    "eco_nan_dilation_pixels": registration_state.eco_nan_dilation_pixels,
                    "overlap_pixels": registration_state.overlap_pixels,
                    "gradient_overlap_pixels": registration_state.gradient_overlap_pixels,
                    "robust_gradient_overlap_pixels": registration_state.robust_gradient_overlap_pixels,
                }
            )
            if logger is not None:
                geo = translation_metrics_from_geometry(
                    registration_state.geometry,
                    register_row,
                    register_col,
                )
                logger.info(
                    f"    Register  OK | {format_elapsed(registration_state.processing_seconds)} | "
                    f"{geo['distance_m']:.1f} m @ {geo['azimuth_deg']:.2f} deg"
                )
        except Exception as exc:
            row["register_seconds"] = time.perf_counter() - register_start
            failure_stage = "register"
            errors.append(str(exc))
            if logger is not None:
                logger.info(
                    f"    Register  FAILED | {format_elapsed(row['register_seconds'])} | {exc}"
                )

    if registration_state is not None:
        assess_start = time.perf_counter()
        registered_for_assessment: Optional[np.ndarray] = None
        try:
            assessment_key = (
                float(register_affine.a),
                float(register_affine.b),
                float(register_affine.c),
                float(register_affine.d),
                float(register_affine.e),
                float(register_affine.f),
            )
            cached_assessment = (
                registration_preparation_cache.assessment_result
                if registration_preparation_cache is not None
                and registration_preparation_cache.assessment_affine_key
                == assessment_key
                else None
            )
            if cached_assessment is not None:
                assessment = AssessmentResult(
                    **{
                        **cached_assessment.__dict__,
                        "processing_seconds": 0.0,
                    }
                )
            else:
                registered_for_assessment = warp_array_continuous(
                    registration_state.moving,
                    register_affine,
                )
                assessment = qa_assess_registered(
                    registration_state.reference,
                    registered_for_assessment,
                    reference_feature_cache=qa_reference_features,
                    local_executor=local_executor,
                )
                if registration_preparation_cache is not None:
                    registration_preparation_cache.assessment_affine_key = (
                        assessment_key
                    )
                    registration_preparation_cache.assessment_result = assessment
            row["assess_seconds"] = assessment.processing_seconds
            if assessment.status != "OK":
                raise RuntimeError(assessment.error or "Residual assessment unavailable.")

            residual_affine = Affine.translation(
                assessment.systematic_col_px,
                assessment.systematic_row_px,
            )
            total_affine = compose_pixel_affines(register_affine, residual_affine)
            row.update(
                {
                    "assessment_confidence_0_100": assessment.confidence_0_100,
                    "polish_residual_row_shift_px": assessment.systematic_row_px,
                    "polish_residual_col_shift_px": assessment.systematic_col_px,
                    "assessment_threshold_profile": assessment.threshold_profile,
                    "assessment_attempt_number": assessment.attempt_number,
                    "assessment_reliable_windows": assessment.reliable_windows,
                    "assessment_evaluated_windows": assessment.evaluated_windows,
                    "assessment_candidate_windows": assessment.candidate_windows,
                    "assessment_systematic_residual_px": assessment.systematic_magnitude_px,
                    "assessment_p90_residual_px": assessment.p90_residual_px,
                    "assessment_local_scatter_p90_px": assessment.local_scatter_p90_px,
                    "assessment_directional_coherence_0_1": assessment.directional_coherence,
                    "assessment_median_match_score": assessment.median_match_score,
                    "assessment_median_peak_margin": assessment.median_peak_margin,
                    "assessment_median_peak_zscore": assessment.median_peak_zscore,
                    "assessment_median_overlap_fraction": assessment.median_overlap_fraction,
                    "assessment_search_edge_hit_fraction": assessment.search_edge_hit_fraction,
                    "assessment_unreliable_reason_counts": assessment.unreliable_reason_counts,
                    "assessment_window_vectors": assessment.window_vectors,
                }
            )
            if logger is not None:
                logger.info(
                    f"    Assess    OK | {format_elapsed(assessment.processing_seconds)} | "
                    f"residual={assessment.systematic_magnitude_px:.2f} px | "
                    f"QA={assessment.confidence_0_100:.1f}%"
                )
        except Exception as exc:
            if assessment is None:
                row["assess_seconds"] = time.perf_counter() - assess_start
            failure_stage = failure_stage or "assess"
            errors.append(str(exc))
            total_affine = register_affine
            if logger is not None:
                logger.info(
                    f"    Assess    FAILED | {format_elapsed(row.get('assess_seconds') or 0.0)} | {exc}"
                )
        finally:
            registered_for_assessment = None

    row["fine_texture_status"] = (
        "DISABLED" if not FINE_TEXTURE_ENABLED else
        "SCIPY_UNAVAILABLE" if scipy_ndimage is None else "STRONG_BASELINE_RETAINED"
    )
    if fine_texture_should_try(registration_state, assessment):
        texture_start = time.perf_counter()
        prepared = registration_preparation_cache.prepared if registration_preparation_cache is not None else None
        row["fine_texture_status"] = "NO_PREPARED_DATA" if prepared is None else "BASELINE_RETAINED"
        baseline_state = (total_affine, register_affine, registration_state, assessment,
                          registration_confidence, failure_stage, list(errors), dict(row))
        try:
            evidence = refine_fine_texture(prepared, total_affine) if prepared is not None else None
            if evidence is not None:
                old_failure = " | ".join(errors)
                total_affine = Affine.translation(evidence.col_shift, evidence.row_shift)
                register_affine = total_affine
                result = ShiftResult(evidence.row_shift, evidence.col_shift,
                                     float(np.median([m.score for m in evidence.matches])),
                                     sum(m.pixels for m in evidence.matches), "fine native texture")
                choice = TranslationChoice(result, False, "held-out native fine-texture translation", math.nan, math.nan)
                model = WarpModel.from_translation(result, choice.method)
                registration_state = RegistrationState(
                    prepared.reference, prepared.moving, prepared.geometry, prepared.nan_stats,
                    prepared.eco_nan_dilation_pixels, prepared.support_loss, prepared.overlap_pixels,
                    prepared.gradient_overlap_pixels, prepared.robust_gradient_overlap_pixels,
                    choice, result, model, time.perf_counter() - attempt_start, evidence)
                assessment = fine_texture_assessment(evidence)
                registration_confidence = evidence.confidence
                failure_stage = ""
                errors.clear()
                update_affine_row(row, "register", register_affine)
                row.update({
                    "fine_texture_status": "ACCEPTED",
                    "fine_texture_fit_windows": evidence.fit_windows,
                    "fine_texture_validation_windows": evidence.validation_windows,
                    "fine_texture_scatter_p90_px": evidence.scatter_p90_px,
                    "fine_texture_median_score_gain": evidence.median_gain,
                    "fine_texture_previous_failure": old_failure,
                    "register_center_row_shift_px": evidence.row_shift,
                    "register_center_col_shift_px": evidence.col_shift,
                    "registration_confidence_0_100": registration_confidence,
                    "registration_method": model.method,
                    "global_match_method": choice.method,
                    "registration_correlation_score": result.score,
                    "registration_valid_pixels": result.valid_pixels,
                    "warp_control_points": 0, "warp_rmse_px": 0.0,
                    "warp_mean_score": result.score, "warp_q25_score": math.nan,
                    "assessment_confidence_0_100": evidence.confidence,
                    "polish_residual_row_shift_px": 0.0, "polish_residual_col_shift_px": 0.0,
                    "assessment_threshold_profile": assessment.threshold_profile,
                    "assessment_attempt_number": 1,
                    "assessment_reliable_windows": assessment.reliable_windows,
                    "assessment_evaluated_windows": assessment.evaluated_windows,
                    "assessment_candidate_windows": assessment.candidate_windows,
                    "assessment_systematic_residual_px": assessment.systematic_magnitude_px,
                    "assessment_p90_residual_px": assessment.p90_residual_px,
                    "assessment_local_scatter_p90_px": assessment.local_scatter_p90_px,
                    "assessment_directional_coherence_0_1": assessment.directional_coherence,
                    "assessment_median_match_score": assessment.median_match_score,
                    "assessment_median_peak_margin": assessment.median_peak_margin,
                    "assessment_median_peak_zscore": assessment.median_peak_zscore,
                    "assessment_median_overlap_fraction": math.nan,
                    "assessment_search_edge_hit_fraction": 0.0,
                    "assessment_unreliable_reason_counts": "",
                    "assessment_window_vectors": "",
                })
                if logger is not None:
                    logger.info(f"    Fine texture ACCEPTED | row={evidence.row_shift:+.3f}, col={evidence.col_shift:+.3f} px | "
                                f"fit={evidence.fit_windows}, held-out={evidence.validation_windows} | "
                                f"native gain={evidence.median_gain:+.4f}")
        except Exception as exc:
            (total_affine, register_affine, registration_state, assessment,
             registration_confidence, failure_stage, errors, baseline_row) = baseline_state
            row.clear()
            row.update(baseline_row)
            row["fine_texture_status"] = "ERROR_BASELINE_RETAINED"
            row["fine_texture_error"] = str(exc)
            if logger is not None:
                logger.info(f"    Fine texture unavailable; existing result retained: {exc}")
        row["fine_texture_seconds"] = time.perf_counter() - texture_start

    if registration_state is not None:
        center_row = registration_state.geometry.height / 2.0
        center_col = registration_state.geometry.width / 2.0
        candidate_row_shift, candidate_col_shift = affine_shift_at(
            total_affine,
            center_row,
            center_col,
        )
        candidate_geo = translation_metrics_from_geometry(
            registration_state.geometry,
            candidate_row_shift,
            candidate_col_shift,
        )
        candidate_shift_px = math.hypot(candidate_row_shift, candidate_col_shift)
        geometry = geometry_confidence(
            registration_state.nan_stats.footprint_pixels,
            registration_state.moving.size,
            candidate_shift_px,
        )
        row.update(geometry)
        row.update(
            {
                "total_translation_distance_m": candidate_geo["distance_m"],
                "total_translation_azimuth_deg": candidate_geo["azimuth_deg"],
                "total_center_row_shift_px": candidate_row_shift,
                "total_center_col_shift_px": candidate_col_shift,
                "candidate_total_translation_distance_m": candidate_geo["distance_m"],
                "candidate_total_translation_azimuth_deg": candidate_geo["azimuth_deg"],
                "candidate_total_center_row_shift_px": candidate_row_shift,
                "candidate_total_center_col_shift_px": candidate_col_shift,
            }
        )
    else:
        row.update(
            {
                "total_translation_distance_m": 0.0,
                "total_translation_azimuth_deg": 0.0,
                "total_center_row_shift_px": 0.0,
                "total_center_col_shift_px": 0.0,
                "candidate_total_translation_distance_m": 0.0,
                "candidate_total_translation_azimuth_deg": 0.0,
                "candidate_total_center_row_shift_px": 0.0,
                "candidate_total_center_col_shift_px": 0.0,
            }
        )

    row.update(affine_components(total_affine))
    row.update(candidate_affine_components(total_affine))
    assessment_confidence = (
        assessment.confidence_0_100
        if assessment is not None and assessment.status == "OK"
        else 0.0
    )
    confidence_percent, confidence_raw = final_confidence_percent(
        assessment_confidence,
        registration_confidence,
        float(geometry["geometry_factor_0_1"]),
        bool(failure_stage),
    )
    row.update(
        {
            "confidence_percent": confidence_percent,
            "confidence_raw_0_100": confidence_raw,
            "confidence_filename_code_00_99": confidence_filename_code(confidence_percent),
            "geometry_confidence_0_100": geometry["geometry_confidence_0_100"],
        }
    )
    return AttemptOutcome(
        row=row,
        registration_state=registration_state,
        assessment=assessment,
        register_affine=register_affine,
        total_affine=total_affine,
        registration_confidence=registration_confidence,
        geometry=geometry,
        confidence_percent=confidence_percent,
        confidence_raw=confidence_raw,
        failure_stage=failure_stage,
        errors=errors,
        processing_seconds=time.perf_counter() - attempt_start,
    )


# BEGIN REGIONAL TEXTURE ADDON
# Native residual deformation; integrated after the affine pipeline.
LOCAL_TEXTURE_SEARCH_RADIUS = 16
LOCAL_TEXTURE_CORE = 160
LOCAL_TEXTURE_SPACING = 72 if PRECISION_ENABLED else 96
LOCAL_TEXTURE_SUPPORT_RADIUS = 2.0 if PRECISION_ENABLED else 2.5
LOCAL_TEXTURE_MAX_GRID = 16
LOCAL_TEXTURE_MIN_CORE = 48
LOCAL_TEXTURE_MIN_SCORE = 0.12
LOCAL_TEXTURE_MIN_MARGIN = 0.006
LOCAL_TEXTURE_MIN_GAIN = 0.003
LOCAL_TEXTURE_MAX_GRADIENT = 0.45


@dataclass
class LocalTextureWarp:
    baseline_affine: Affine
    shape: Tuple[int, int]
    row_knots: np.ndarray
    col_knots: np.ndarray
    row_field: np.ndarray
    col_field: np.ndarray
    controls: np.ndarray
    shifts: np.ndarray
    fit_windows: int = 0
    validation_windows: int = 0
    median_gain: float = math.nan
    scatter_p90_px: float = math.nan
    confidence: float = 0.0
    processing_seconds: float = 0.0
    gradient_bound: float = 0.0
    validation_matches: list = field(default_factory=list)
    validation_bounds: list = field(default_factory=list)
    median_score: float = math.nan
    before_p90_px: float = math.nan
    precision_mode: bool = False

    def displacement(self, rows, cols):
        """Destination pixel-index coordinates -> (dy, dx)."""
        rows, cols = np.broadcast_arrays(np.asarray(rows, dtype=np.float64),
                                        np.asarray(cols, dtype=np.float64))
        ri = np.interp(rows.ravel(), self.row_knots, np.arange(len(self.row_knots)))
        ci = np.interp(cols.ravel(), self.col_knots, np.arange(len(self.col_knots)))
        coordinates = np.asarray([ri, ci])
        dy = scipy_ndimage.map_coordinates(self.row_field, coordinates, order=1,
                                           mode="nearest", prefilter=False).reshape(rows.shape)
        dx = scipy_ndimage.map_coordinates(self.col_field, coordinates, order=1,
                                           mode="nearest", prefilter=False).reshape(rows.shape)
        inside = (rows >= 0) & (rows <= self.shape[0] - 1) & (cols >= 0) & (cols <= self.shape[1] - 1)
        return np.where(inside, dy, 0.0), np.where(inside, dx, 0.0)

    def serialize(self):
        """Exact float32 bilinear field; no companion files."""
        import base64
        import zlib
        header = [*self.shape, len(self.row_knots), len(self.col_knots),
                  self.baseline_affine.a, self.baseline_affine.b, self.baseline_affine.c,
                  self.baseline_affine.d, self.baseline_affine.e, self.baseline_affine.f]
        values = np.concatenate([self.row_knots, self.col_knots,
                                 self.row_field.ravel(), self.col_field.ravel()]).astype("<f4")
        payload = base64.b64encode(zlib.compress(values.tobytes(), 6)).decode("ascii")
        return "LTW1|" + ",".join(format(v, ".17g") for v in header) + "|" + payload

    @classmethod
    def deserialize(cls, definition):
        import base64
        import zlib
        version, header, payload = definition.split("|", 2)
        if version != "LTW1":
            raise ValueError("Unsupported local-warp definition.")
        h, w, ny, nx, *coefficients = [float(v) for v in header.split(",")]
        h, w, ny, nx = int(h), int(w), int(ny), int(nx)
        values = np.frombuffer(zlib.decompress(base64.b64decode(payload)), dtype="<f4")
        if len(values) != ny + nx + 2 * ny * nx:
            raise ValueError("Incomplete local-warp definition.")
        k = ny + nx
        return cls(Affine(*coefficients), (h, w), values[:ny].copy(), values[ny:k].copy(),
                   values[k:k + ny * nx].reshape(ny, nx).copy(),
                   values[k + ny * nx:].reshape(ny, nx).copy(),
                   np.empty((0, 2)), np.empty((0, 2)))


def local_texture_windows(shape):
    """Disjoint native cores; alternating fit and withheld validation."""
    height, width = shape
    counts = [min(LOCAL_TEXTURE_MAX_GRID, max(1, int(v // (LOCAL_TEXTURE_MIN_CORE + 16))),
                  max(1, min(4, int(v // 64)), int(math.ceil(v / LOCAL_TEXTURE_SPACING))))
              for v in shape]
    edges = [np.linspace(0, size, count + 1, dtype=int)
             for size, count in zip(shape, counts)]
    windows = []
    for iy in range(counts[0]):
        for ix in range(counts[1]):
            hh = min(LOCAL_TEXTURE_CORE, int(edges[0][iy + 1] - edges[0][iy]) - 16)
            ww = min(LOCAL_TEXTURE_CORE, int(edges[1][ix + 1] - edges[1][ix]) - 16)
            if min(hh, ww) < LOCAL_TEXTURE_MIN_CORE:
                continue
            cy = int((edges[0][iy] + edges[0][iy + 1]) // 2)
            cx = int((edges[1][ix] + edges[1][ix + 1]) // 2)
            windows.append(((cy - hh // 2, cy - hh // 2 + hh,
                             cx - ww // 2, cx - ww // 2 + ww), (iy + ix) % 2 == 0))
    spacing = float(max(height / counts[0], width / counts[1]))
    return windows, spacing


def _local_texture_center(bounds):
    return ((bounds[0] + bounds[1] - 1) / 2.0, (bounds[2] + bounds[3] - 1) / 2.0)


def _local_texture_sample(data, sy, sx, origin=(0, 0)):
    coordinates = np.asarray([sy - origin[0], sx - origin[1]])
    valid = np.isfinite(data)
    support = scipy_ndimage.map_coordinates(valid.astype(np.float32), coordinates,
                                            order=1, mode="constant", cval=0.0, prefilter=False)
    result = scipy_ndimage.map_coordinates(np.where(valid, data, 0.0), coordinates,
                                           order=1, mode="constant", cval=0.0, prefilter=False)
    result[support < 0.9999] = np.nan
    return result


def _local_texture_aligned_patch(moving, bounds, baseline):
    y0, y1, x0, x1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    inverse = ~baseline
    sy = inverse.d * (xx + 0.5) + inverse.e * (yy + 0.5) + inverse.f - 0.5
    sx = inverse.a * (xx + 0.5) + inverse.b * (yy + 0.5) + inverse.c - 0.5
    sy0 = max(0, int(math.floor(float(sy.min()))) - 1)
    sy1 = min(moving.shape[0], int(math.ceil(float(sy.max()))) + 2)
    sx0 = max(0, int(math.floor(float(sx.min()))) - 1)
    sx1 = min(moving.shape[1], int(math.ceil(float(sx.max()))) + 2)
    if min(sy1 - sy0, sx1 - sx0) < 2:
        return np.full(yy.shape, np.nan, dtype=np.float32)
    return _local_texture_sample(moving[sy0:sy1, sx0:sx1], sy, sx, (sy0, sx0))


def local_texture_match(reference, moving, bounds, baseline=Affine.identity()):
    """Residual shift after baseline; no global directional consensus."""
    y0, y1, x0, x1 = bounds
    halo = 40
    py0, py1 = max(0, y0 - halo), min(reference.shape[0], y1 + halo)
    px0, px1 = max(0, x0 - halo), min(reference.shape[1], x1 + halo)
    ref_features = fine_texture_features(reference[py0:py1, px0:px1])
    mov_features = fine_texture_features(_local_texture_aligned_patch(
        moving, (py0, py1, px0, px1), baseline))
    radius = LOCAL_TEXTURE_SEARCH_RADIUS
    thresholds = QAThresholdProfile("local_native", 0.35, 384, 0.60,
                                    LOCAL_TEXTURE_MIN_SCORE, LOCAL_TEXTURE_MIN_MARGIN, 2.0)
    surfaces, counts_list = [], []
    for reference_feature, moving_feature in zip(ref_features, mov_features):
        ref = reference_feature[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        mov = moving_feature[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        try:
            scores, counts, _, _, _ = qa_masked_zncc_surface(ref, mov, radius, thresholds)
        except RuntimeError:
            return None
        if not np.isfinite(scores).any():
            return None
        surfaces.append(scores)
        counts_list.append(counts)
    peaks = [np.asarray(np.unravel_index(np.nanargmax(s), s.shape)) for s in surfaces]
    if np.linalg.norm(peaks[0] - peaks[1]) > 2.0:
        return None
    scores = 0.5 * (surfaces[0] + surfaces[1])
    if not np.isfinite(scores).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(scores), scores.shape)
    if iy in (0, 2 * radius) or ix in (0, 2 * radius):
        return None
    score = float(scores[iy, ix])
    second, zscore = qa_peak_background_metrics(scores, iy, ix, score)
    margin = score - second
    if (not math.isfinite(margin) or not math.isfinite(zscore)
            or score < LOCAL_TEXTURE_MIN_SCORE or margin < LOCAL_TEXTURE_MIN_MARGIN or zscore < 2.0):
        return None
    dr, dc, _ = qa_quadratic_subpixel(scores, iy, ix)
    row_shift, col_shift = float(iy - radius + dr), float(ix - radius + dc)
    if PRECISION_ENABLED and score >= 0.25 and math.hypot(row_shift, col_shift) >= 0.02:
        refined = refine_local_texture_shift(reference, moving, bounds, baseline, row_shift, col_shift)
        if refined is not None:
            row_shift, col_shift = refined
    return FineTextureMatch(bounds, row_shift, col_shift,
                            score, margin, zscore, int(min(c[iy, ix] for c in counts_list)))


def _local_texture_gradient_bound(row_field, col_field, row_knots, col_knots):
    """Cellwise Frobenius bound covers every bilinear derivative."""
    terms = []
    for values in (row_field.astype(np.float64), col_field.astype(np.float64)):
        ddy = np.diff(values, axis=0) / np.diff(row_knots)[:, None]
        ddx = np.diff(values, axis=1) / np.diff(col_knots)[None, :]
        terms.append(np.maximum(np.abs(ddy[:, :-1]), np.abs(ddy[:, 1:])) ** 2)
        terms.append(np.maximum(np.abs(ddx[:-1, :]), np.abs(ddx[1:, :])) ** 2)
    return float(np.sqrt(sum(terms)).max())


def _local_texture_build_model(matches, shape, baseline, spacing, accepted=None, rejected=None):
    controls = np.asarray([_local_texture_center(m.bounds) for m in matches], dtype=np.float64)
    shifts = np.asarray([(m.row_shift, m.col_shift) for m in matches], dtype=np.float64)
    weights = np.asarray([max(0.05, m.score) for m in matches], dtype=np.float64)
    row_knots = np.linspace(0, shape[0] - 1, max(2, int(math.ceil((shape[0] - 1) / 48)) + 1), dtype=np.float32)
    col_knots = np.linspace(0, shape[1] - 1, max(2, int(math.ceil((shape[1] - 1) / 48)) + 1), dtype=np.float32)
    gy, gx = np.meshgrid(row_knots, col_knots, indexing="ij")
    locations = np.column_stack([gy.ravel(), gx.ravel()])
    radius = LOCAL_TEXTURE_SUPPORT_RADIUS * spacing
    output = np.zeros((len(locations), 2), dtype=np.float64)
    for start in range(0, len(locations), 1024):
        points = locations[start:start + 1024]
        delta = (controls[None, :, :] - points[:, None, :]) / radius
        distance = np.linalg.norm(delta, axis=2)
        q = np.maximum(0.0, 1.0 - distance)
        kernel = q ** 4 * (4.0 * distance + 1.0) * weights[None, :]
        total = kernel.sum(axis=1)
        design = np.concatenate([np.ones((*delta.shape[:2], 1)), delta], axis=2)
        matrix = np.einsum("pni,pn,pnj->pij", design, kernel, design)
        matrix[:, 0, 0] += 1e-8
        matrix[:, 1, 1] += 0.001 * total + 1e-8
        matrix[:, 2, 2] += 0.001 * total + 1e-8
        target = np.einsum("pni,pn,nj->pij", design, kernel, shifts)
        value = np.linalg.solve(matrix, target)[:, 0, :]
        fade = np.clip((2.0 - distance.min(axis=1) * LOCAL_TEXTURE_SUPPORT_RADIUS) / 1.1, 0.0, 1.0)
        fade = fade * fade * (3.0 - 2.0 * fade)
        if accepted is not None:
            accepted_distance = np.linalg.norm(points[:, None, :] - accepted[None, :, :], axis=2).min(axis=1)
            gate = np.clip((1.8 - accepted_distance / spacing) / 1.1, 0.0, 1.0)
            gate = gate * gate * (3.0 - 2.0 * gate)
            if rejected is not None and len(rejected):
                rejected_distance = np.linalg.norm(points[:, None, :] - rejected[None, :, :], axis=2).min(axis=1)
                protect = np.clip((rejected_distance / spacing - 0.55) / 0.45, 0.0, 1.0)
                gate *= protect * protect * (3.0 - 2.0 * protect)
            fade *= gate
        value *= fade[:, None]
        magnitude = np.linalg.norm(value, axis=1)
        value *= np.minimum(1.0, (LOCAL_TEXTURE_SEARCH_RADIUS - 0.5) / np.maximum(magnitude, 1e-12))[:, None]
        output[start:start + len(points)] = value
    row_field = output[:, 0].reshape(gy.shape).astype(np.float32)
    col_field = output[:, 1].reshape(gy.shape).astype(np.float32)
    gradient = _local_texture_gradient_bound(row_field, col_field, row_knots, col_knots)
    if gradient > LOCAL_TEXTURE_MAX_GRADIENT:
        factor = LOCAL_TEXTURE_MAX_GRADIENT / gradient * 0.999
        row_field *= factor
        col_field *= factor
        gradient = _local_texture_gradient_bound(row_field, col_field, row_knots, col_knots)
    return LocalTextureWarp(baseline, shape, row_knots, col_knots, row_field, col_field,
                            controls, shifts, len(matches), gradient_bound=gradient)


def _local_texture_validate(prepared, bounds, model):
    """Compare on common support using one original-feature interpolation."""
    y0, y1, x0, x1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dy, dx = model.displacement(yy, xx)
    inverse = ~model.baseline_affine
    before_y = inverse.d * (xx + 0.5) + inverse.e * (yy + 0.5) + inverse.f - 0.5
    before_x = inverse.a * (xx + 0.5) + inverse.b * (yy + 0.5) + inverse.c - 0.5
    after_y = before_y - inverse.d * dx - inverse.e * dy
    after_x = before_x - inverse.a * dx - inverse.b * dy
    sy0 = max(0, int(math.floor(min(before_y.min(), after_y.min()))) - 40)
    sy1 = min(prepared.moving.shape[0], int(math.ceil(max(before_y.max(), after_y.max()))) + 41)
    sx0 = max(0, int(math.floor(min(before_x.min(), after_x.min()))) - 40)
    sx1 = min(prepared.moving.shape[1], int(math.ceil(max(before_x.max(), after_x.max()))) + 41)
    if min(sy1 - sy0, sx1 - sx0) < LOCAL_TEXTURE_MIN_CORE or max(sy1 - sy0, sx1 - sx0) > 1024:
        return None
    reference, origin = fine_texture_patch(prepared.reference, bounds, extra=8)
    moving = fine_texture_features(prepared.fine_texture_moving[sy0:sy1, sx0:sx1])
    pairs = []
    for rf, mf in zip(reference, moving):
        ref = rf[y0 - origin[0]:y1 - origin[0], x0 - origin[1]:x1 - origin[1]]
        pair = fine_texture_correlations(ref,
            _local_texture_sample(mf, before_y, before_x, (sy0, sx0)),
            _local_texture_sample(mf, after_y, after_x, (sy0, sx0)))
        if pair is None:
            return None
        pairs.append(pair)
    broad = fine_texture_correlations(prepared.legacy_reference[y0:y1, x0:x1],
        _local_texture_sample(prepared.legacy_moving[sy0:sy1, sx0:sx1], before_y, before_x, (sy0, sx0)),
        _local_texture_sample(prepared.legacy_moving[sy0:sy1, sx0:sx1], after_y, after_x, (sy0, sx0)))
    gain = float(np.mean([b - a for a, b in pairs]))
    score = float(np.mean([b for a, b in pairs]))
    protected = (broad is not None and broad[0] > 0.30 and broad[1] - broad[0] < -0.02)
    return gain, score, protected


def estimate_local_texture_warp(prepared, baseline_affine):
    """Fit residual areas independently; keep unsupported areas at baseline."""
    started = time.perf_counter()
    if not LOCAL_TEXTURE_ENABLED or scipy_ndimage is None or prepared is None:
        return None
    coefficients = np.asarray(tuple(baseline_affine)[:6], dtype=np.float64)
    if not np.isfinite(coefficients).all() or baseline_affine.determinant <= 0.0:
        return None
    windows, spacing = local_texture_windows(prepared.reference.shape)
    fit = [m for bounds, is_fit in windows if is_fit
           if (m := local_texture_match(prepared.reference, prepared.fine_texture_moving,
                                        bounds, baseline_affine)) is not None]
    if len(fit) < 3:
        return None
    centers = np.asarray([_local_texture_center(m.bounds) for m in fit])
    shifts = np.asarray([(m.row_shift, m.col_shift) for m in fit])
    separation = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    disagreement = np.linalg.norm(shifts[:, None] - shifts[None, :], axis=2)
    consistent = ((separation > 0) & (separation <= 2.5 * spacing)
                  & (disagreement <= np.maximum(2.5, 0.04 * separation)))
    fit = [m for m, supported in zip(fit, consistent.any(axis=1)) if supported]
    if len(fit) < 3:
        return None
    centers = np.asarray([_local_texture_center(m.bounds) for m in fit])
    if np.linalg.matrix_rank((centers - centers.mean(axis=0)) / spacing, tol=1e-5) < 2:
        return None
    precision_mode = bool(PRECISION_ENABLED and len(fit) >= 6
                          and np.median([m.score for m in fit]) >= 0.65
                          and np.percentile([math.hypot(m.row_shift, m.col_shift) for m in fit], 90) <= 0.75)
    minimum_motion = 0.05 if precision_mode else 0.20
    minimum_gain = 0.0002 if precision_mode else LOCAL_TEXTURE_MIN_GAIN
    minimum_improvement = 0.02 if precision_mode else 0.10
    minimum_validation = 4 if precision_mode else 2
    damage_gain = -0.0005 if precision_mode else -0.01
    rejection_gain = -0.0005 if precision_mode else -0.005
    model = _local_texture_build_model(fit, prepared.reference.shape, baseline_affine, spacing)
    if float(np.max(np.hypot(model.row_field, model.col_field))) < minimum_motion:
        return None
    validation = []
    rejected_centers = []
    for bounds, is_fit in windows:
        if is_fit:
            continue
        match = local_texture_match(prepared.reference, prepared.fine_texture_moving,
                                    bounds, baseline_affine)
        pair = _local_texture_validate(prepared, bounds, model)
        cy, cx = _local_texture_center(bounds)
        if pair is None:
            rejected_centers.append((cy, cx))
            continue
        dy, dx = model.displacement(cy, cx)
        strong_match = (not precision_mode or (match is not None and match.score >= 0.65
                                               and match.pixels >= 512 and pair[1] >= 0.65))
        if match is None or not strong_match or pair[2] or pair[0] < rejection_gain:
            rejected_centers.append((cy, cx))
        if match is None or not strong_match or pair[2] or pair[0] < minimum_gain:
            continue
        residual = math.hypot(match.row_shift - float(dy), match.col_shift - float(dx))
        before = math.hypot(match.row_shift, match.col_shift)
        residual_limit = max(0.08, 0.35 * before) if precision_mode else max(2.5, 0.30 * before)
        if residual <= residual_limit and residual < before - minimum_improvement:
            validation.append(match)
    if len(validation) < minimum_validation:
        return None
    accepted = np.asarray([_local_texture_center(m.bounds) for m in validation])
    for repair_pass in range(3):
        model = _local_texture_build_model(fit, prepared.reference.shape, baseline_affine,
                                          spacing, accepted, np.asarray(rejected_centers))
        final_matches, gains, scores, residuals, before_residuals = [], [], [], [], []
        for match in validation:
            pair = _local_texture_validate(prepared, match.bounds, model)
            if pair is None or pair[2] or pair[0] < minimum_gain:
                continue
            cy, cx = _local_texture_center(match.bounds)
            dy, dx = model.displacement(cy, cx)
            residual = math.hypot(match.row_shift - float(dy), match.col_shift - float(dx))
            before = math.hypot(match.row_shift, match.col_shift)
            residual_limit = max(0.08, 0.35 * before) if precision_mode else max(2.5, 0.30 * before)
            if residual > residual_limit or residual >= before - minimum_improvement:
                continue
            final_matches.append(match)
            gains.append(pair[0])
            scores.append(pair[1])
            residuals.append(residual)
            before_residuals.append(before)
        if len(final_matches) < minimum_validation:
            return None
        # Suppress only the measured damaged regions, then revalidate.
        damaged = []
        final_bounds = {m.bounds for m in final_matches}
        for bounds, is_fit in windows:
            if is_fit or bounds in final_bounds:
                continue
            cy, cx = _local_texture_center(bounds)
            dy, dx = model.displacement(cy, cx)
            if math.hypot(float(dy), float(dx)) < minimum_motion:
                continue
            pair = _local_texture_validate(prepared, bounds, model)
            if pair is not None and (pair[2] or pair[0] < damage_gain):
                damaged.append((cy, cx))
        if not damaged:
            break
        if repair_pass == 2:
            return None
        rejected_centers.extend(damaged)
    model.validation_matches = final_matches
    model.precision_mode = precision_mode
    model.validation_bounds = [m.bounds for m in final_matches]
    model.validation_windows = len(final_matches)
    model.median_gain = float(np.median(gains))
    model.median_score = float(np.median(scores))
    model.scatter_p90_px = float(np.percentile(residuals, 90))
    model.before_p90_px = float(np.percentile(before_residuals, 90))
    model.confidence = min(95.0, 100.0 * (
        0.25 * min(1.0, len(final_matches) / 8.0)
        + 0.35 * min(1.0, max(0.0, model.median_score) / 0.65)
        + 0.40 * max(0.0, 1.0 - model.scatter_p90_px / 3.0)))
    model.processing_seconds = time.perf_counter() - started
    return model
# END REGIONAL TEXTURE ADDON


# BEGIN CONTINUOUS PRECISION REFINEMENT
PRECISION_MAX_SHIFT_PX = 2.0
PRECISION_MAX_DEFORMATION_PX = 0.45
PRECISION_MIN_GAIN = 0.00005


@dataclass
class PrecisionAffineEvidence:
    candidate_affine: Affine
    model: str
    fit_windows: int
    validation_windows: int
    median_gain: float
    baseline_score: float
    candidate_score: float
    maximum_displacement_px: float
    processing_seconds: float


def _precision_windows(shape):
    height, width = shape
    grid = 4 if min(shape) >= 240 else 3
    ye = np.linspace(0, height, grid + 1, dtype=int)
    xe = np.linspace(0, width, grid + 1, dtype=int)
    windows = []
    for iy in range(grid):
        for ix in range(grid):
            h = min(128, int(ye[iy + 1] - ye[iy]) - 12)
            w = min(128, int(xe[ix + 1] - xe[ix]) - 12)
            if min(h, w) < 32:
                continue
            cy = (int(ye[iy]) + int(ye[iy + 1])) // 2
            cx = (int(xe[ix]) + int(xe[ix + 1])) // 2
            windows.append(((cy - h // 2, cy - h // 2 + h,
                             cx - w // 2, cx - w // 2 + w), (iy + ix) % 2 == 0))
    return windows


def _precision_delta(parameters, shape):
    height, width = shape
    p = np.zeros(6, dtype=np.float64)
    p[:len(parameters)] = parameters
    cy, cx = height / 2.0, width / 2.0
    sy, sx = max(1.0, cy), max(1.0, cx)
    return Affine(1.0 + p[4] / sx, p[5] / sy,
                  p[1] - p[4] * cx / sx - p[5] * cy / sy,
                  p[2] / sx, 1.0 + p[3] / sy,
                  p[0] - p[2] * cx / sx - p[3] * cy / sy)


def _precision_sample(values, rows, cols, affine, origin):
    inverse = ~affine
    sy = inverse.d * (cols + 0.5) + inverse.e * (rows + 0.5) + inverse.f - 0.5 - origin[0]
    sx = inverse.a * (cols + 0.5) + inverse.b * (rows + 0.5) + inverse.c - 0.5 - origin[1]
    return scipy_ndimage.map_coordinates(values, np.asarray([sy, sx]), order=1,
                                         mode="constant", cval=0.0, prefilter=False)


def _precision_patch(prepared, bounds, baseline):
    y0, y1, x0, x1 = bounds
    reference, ro = fine_texture_patch(prepared.reference, bounds)
    corners = [(x0 + 0.5, y0 + 0.5), (x1 - 0.5, y0 + 0.5),
               (x0 + 0.5, y1 - 0.5), (x1 - 0.5, y1 - 0.5)]
    points = [(~baseline) * p for p in corners]
    pad = FINE_TEXTURE_HALO + 5
    h, w = prepared.moving.shape
    sy0 = max(0, int(math.floor(min(p[1] for p in points))) - pad)
    sy1 = min(h, int(math.ceil(max(p[1] for p in points))) + pad + 1)
    sx0 = max(0, int(math.floor(min(p[0] for p in points))) - pad)
    sx1 = min(w, int(math.ceil(max(p[0] for p in points))) + pad + 1)
    if min(sy1 - sy0, sx1 - sx0) < 32 or max(sy1 - sy0, sx1 - sx0) > 512:
        return None
    moving = fine_texture_features(prepared.fine_texture_moving[sy0:sy1, sx0:sx1])
    # Fixed support covers every permitted trial displacement.
    step = 2 if min(y1 - y0, x1 - x0) >= 56 else 1
    yy, xx = np.mgrid[y0:y1:step, x0:x1:step].astype(np.float64)
    valid = np.ones(yy.shape, dtype=bool)
    references, sources = [], []
    for r, m in zip(reference, moving):
        rc = r[y0 - ro[0]:y1 - ro[0]:step, x0 - ro[1]:x1 - ro[1]:step]
        safe = scipy_ndimage.minimum_filter(np.isfinite(m).astype(np.float32), size=9,
                                             mode="constant", cval=0.0)
        support = _precision_sample(safe, yy, xx, baseline, (sy0, sx0))
        valid &= np.isfinite(rc) & (support >= 0.9999)
        references.append(rc)
        sources.append(np.where(np.isfinite(m), m, 0.0).astype(np.float64))
    count = int(valid.sum())
    if count < max(256, int(0.50 * yy.size)):
        return None
    rows, cols = yy[valid], xx[valid]
    normalized, baseline_scores = [], []
    for r, m in zip(references, sources):
        rv = r[valid].astype(np.float64)
        rv -= rv.mean()
        norm = float(np.linalg.norm(rv))
        if norm / math.sqrt(count) < 0.035:
            return None
        rv /= norm
        mv = _precision_sample(m, rows, cols, baseline, (sy0, sx0))
        mv -= mv.mean()
        mn = float(np.linalg.norm(mv))
        if mn / math.sqrt(count) < 0.035:
            return None
        baseline_scores.append(float(rv @ mv / mn))
        normalized.append(rv)
    if min(baseline_scores) < 0.30 or float(np.mean(baseline_scores)) < 0.40:
        return None
    return {"bounds": bounds, "rows": rows, "cols": cols, "reference": normalized,
            "moving": sources, "origin": (sy0, sx0), "baseline": baseline_scores}


def _precision_scores(patches, affine):
    scores = []
    for patch in patches:
        channels = []
        for ref, moving in zip(patch["reference"], patch["moving"]):
            sample = _precision_sample(moving, patch["rows"], patch["cols"], affine, patch["origin"])
            sample -= sample.mean()
            norm = float(np.linalg.norm(sample))
            channels.append(float(ref @ sample / max(norm, 1e-12)))
        scores.append(channels)
    return np.asarray(scores)


def _precision_spread(patches, shape):
    if len(patches) < 3:
        return False
    centers = np.asarray([((p["bounds"][0] + p["bounds"][1]) / 2.0,
                           (p["bounds"][2] + p["bounds"][3]) / 2.0) for p in patches])
    normalized = (centers - np.asarray(shape) / 2.0) / np.asarray(shape)
    return bool(np.all(np.ptp(normalized, axis=0) >= 0.35)
                and np.linalg.matrix_rank(normalized - normalized.mean(axis=0), tol=0.03) == 2)


def refine_precision_affine(prepared, baseline_affine) -> Optional[PrecisionAffineEvidence]:
    """Continuous native-coordinate fit with disjoint, fixed-support validation."""
    started = time.perf_counter()
    if not PRECISION_ENABLED or scipy_ndimage is None or prepared is None:
        return None
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    coefficients = np.asarray(tuple(baseline_affine)[:6], dtype=np.float64)
    if (not np.isfinite(coefficients).all() or baseline_affine.determinant <= 0.0
            or not 0.85 <= baseline_affine.determinant <= 1.18):
        return None
    singular_values = np.linalg.svd(np.asarray([[baseline_affine.a, baseline_affine.b],
                                               [baseline_affine.d, baseline_affine.e]]), compute_uv=False)
    if singular_values.min() < 0.85 or singular_values.max() > 1.18:
        return None
    shape = prepared.reference.shape
    windows = _precision_windows(shape)
    fit, validation = [], []
    for bounds, is_fit in windows:
        patch = _precision_patch(prepared, bounds, baseline_affine)
        if patch is not None:
            (fit if is_fit else validation).append(patch)
    if not _precision_spread(fit, shape) or not _precision_spread(validation, shape):
        return None
    # Stable per-window weights reduce influence from changed/cloudy texture.
    fit_weights = np.clip(np.mean([p["baseline"] for p in fit], axis=1), 0.25, 0.95) ** 2
    fit_weights /= fit_weights.sum()

    def objective(p):
        candidate = _precision_delta(p, shape) * baseline_affine
        correlations = _precision_scores(fit, candidate)
        loss = float(fit_weights @ (1.0 - correlations.mean(axis=1)))
        if len(p) == 6:
            loss += 0.00001 * float(np.dot(p[2:], p[2:]))
        return loss

    options = {"maxiter": 40, "ftol": 1e-11, "gtol": 1e-6, "eps": 0.001, "maxls": 15}
    shift_bound = 1.5
    initial_loss = objective(np.zeros(2))
    translation = minimize(objective, np.zeros(2), method="L-BFGS-B",
                           bounds=[(-shift_bound, shift_bound)] * 2, options=options)
    if not np.isfinite(translation.x).all() or not math.isfinite(translation.fun):
        return None
    parameters = translation.x
    model = "translation"
    if len(fit) >= 6:
        initial = np.r_[parameters, np.zeros(4)]
        affine = minimize(objective, initial, method="L-BFGS-B",
                          bounds=[(-shift_bound, shift_bound)] * 2
                          + [(-PRECISION_MAX_DEFORMATION_PX, PRECISION_MAX_DEFORMATION_PX)] * 4,
                          options=options)
        # Choose complexity using fitting data only; validation remains untouched.
        gain = float(translation.fun - affine.fun)
        if (np.isfinite(affine.x).all() and math.isfinite(affine.fun)
                and gain > max(0.00005, 0.02 * max(initial_loss - translation.fun, 0.0))):
            parameters, model = affine.x, "affine"
    bounds = np.r_[[shift_bound] * 2, [PRECISION_MAX_DEFORMATION_PX] * max(0, len(parameters) - 2)]
    if np.any(np.abs(parameters) >= 0.98 * bounds):
        return None
    delta = _precision_delta(parameters, shape)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    corners = [(0.5, 0.5), (shape[1] - 0.5, 0.5), (0.5, shape[0] - 0.5),
               (shape[1] - 0.5, shape[0] - 0.5), (cx, cy)]
    displacement = max(math.hypot(*(np.asarray(delta * p) - p)) for p in corners)
    if not 0.01 <= displacement <= PRECISION_MAX_SHIFT_PX:
        return None
    candidate = delta * baseline_affine
    before = np.asarray([p["baseline"] for p in validation])
    after = _precision_scores(validation, candidate)
    gains = np.mean(after - before, axis=1)
    if (float(np.median(gains)) < PRECISION_MIN_GAIN
            or float(np.mean(gains > 0.0)) < 0.75
            or float(np.percentile(gains, 10)) < -0.0002
            or np.any(np.mean(after - before, axis=0) < 0.0)
            or float(np.median(after)) < 0.45):
        return None
    # Existing independent broad-edge test protects large thermal boundaries.
    broad_gains = []
    for patch in validation:
        pair = fine_texture_validate_window(prepared, patch["bounds"], baseline_affine, candidate)
        if pair is None:
            return None
        if pair[1] is not None:
            broad_gains.append(pair[1][1] - pair[1][0])
    if broad_gains and (float(np.median(broad_gains)) < -0.002
                        or float(np.percentile(broad_gains, 10)) < -0.010):
        return None
    return PrecisionAffineEvidence(candidate, model, len(fit), len(validation),
                                   float(np.median(gains)), float(np.median(before)),
                                   float(np.median(after)), float(displacement),
                                   time.perf_counter() - started)
# END CONTINUOUS PRECISION REFINEMENT


def refine_local_texture_shift(reference, moving, bounds, baseline, seed_row, seed_col):
    """Refine a local residual against native features on fixed support."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    y0, y1, x0, x1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    inverse = ~baseline
    seed = np.asarray([seed_row, seed_col], dtype=np.float64)
    if not np.isfinite(seed).all():
        return None
    limit = 0.70
    base_y = inverse.d * (xx + 0.5) + inverse.e * (yy + 0.5) + inverse.f - 0.5
    base_x = inverse.a * (xx + 0.5) + inverse.b * (yy + 0.5) + inverse.c - 0.5
    sy = base_y - inverse.e * seed[0] - inverse.d * seed[1]
    sx = base_x - inverse.b * seed[0] - inverse.a * seed[1]
    max_y = limit * (abs(inverse.d) + abs(inverse.e))
    max_x = limit * (abs(inverse.a) + abs(inverse.b))
    sy0 = max(0, int(math.floor(float(sy.min() - max_y))) - 40)
    sy1 = min(moving.shape[0], int(math.ceil(float(sy.max() + max_y))) + 41)
    sx0 = max(0, int(math.floor(float(sx.min() - max_x))) - 40)
    sx1 = min(moving.shape[1], int(math.ceil(float(sx.max() + max_x))) + 41)
    if min(sy1 - sy0, sx1 - sx0) < LOCAL_TEXTURE_MIN_CORE or max(sy1 - sy0, sx1 - sx0) > 1024:
        return None
    rfeatures, origin = fine_texture_patch(reference, bounds, extra=8)
    source = moving[sy0:sy1, sx0:sx1]
    mfeatures = fine_texture_features(source)
    ref_raw = reference[y0:y1, x0:x1]
    common = np.isfinite(ref_raw)
    margin_y, margin_x = int(math.ceil(max_y)) + 2, int(math.ceil(max_x)) + 2
    for rf, mf in zip(rfeatures, mfeatures):
        ref = rf[y0 - origin[0]:y1 - origin[0], x0 - origin[1]:x1 - origin[1]]
        valid = scipy_ndimage.minimum_filter(np.isfinite(mf).astype(np.uint8),
            size=(2 * margin_y + 1, 2 * margin_x + 1), mode="constant", cval=0)
        supported = _local_texture_sample(valid.astype(np.float32), sy, sx, (sy0, sx0))
        common &= np.isfinite(ref) & (supported >= 0.9999)
    count = int(np.count_nonzero(common))
    if count < max(384, int(0.60 * common.size)):
        return None
    if count > 4096:
        indices = np.flatnonzero(common)
        selected = indices[np.linspace(0, count - 1, 4096, dtype=int)]
        common = np.zeros(common.shape, dtype=bool)
        common.ravel()[selected] = True
    rvalues = []
    for rf in rfeatures:
        ref = rf[y0 - origin[0]:y1 - origin[0], x0 - origin[1]:x1 - origin[1]][common].astype(np.float64)
        ref -= ref.mean()
        norm = math.sqrt(float(ref @ ref))
        if norm <= 1e-9:
            return None
        rvalues.append(ref / norm)
    by, bx = base_y[common] - sy0, base_x[common] - sx0
    filled = [np.where(np.isfinite(f), f, 0.0) for f in mfeatures]
    def evaluate(shift):
        coords = np.asarray([by - inverse.e * shift[0] - inverse.d * shift[1],
                             bx - inverse.b * shift[0] - inverse.a * shift[1]])
        scores = []
        for ref, mf in zip(rvalues, filled):
            values = scipy_ndimage.map_coordinates(mf, coords, order=1,
                mode="constant", cval=0.0, prefilter=False).astype(np.float64)
            values -= values.mean()
            norm = math.sqrt(float(values @ values))
            if norm <= 1e-9:
                return np.asarray([-1.0, -1.0])
            scores.append(float(ref @ values) / norm)
        return np.asarray(scores)
    initial = evaluate(seed)
    result = minimize(lambda shift: -float(evaluate(shift).mean()), seed, method="Powell",
        bounds=[(float(v - limit), float(v + limit)) for v in seed],
        options={"maxiter": 7, "maxfev": 65, "xtol": 0.0015, "ftol": 1e-8})
    candidate = np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(candidate).all() or np.any(np.abs(candidate - seed) > limit * 0.98):
        return None
    final = evaluate(candidate)
    if float(final.mean() - initial.mean()) < 2e-7 or np.any(final - initial < -0.001):
        return None
    r = ref_raw[common].astype(np.float64)
    r -= r.mean()
    rnorm = math.sqrt(float(r @ r))
    raw = np.where(np.isfinite(source), source, 0.0)
    broad_scores = []
    for shift in (seed, candidate):
        coords = np.asarray([by - inverse.e * shift[0] - inverse.d * shift[1],
                             bx - inverse.b * shift[0] - inverse.a * shift[1]])
        values = scipy_ndimage.map_coordinates(raw, coords, order=1,
            mode="constant", cval=0.0, prefilter=False).astype(np.float64)
        values -= values.mean()
        denom = rnorm * math.sqrt(float(values @ values))
        broad_scores.append(float(r @ values) / denom if denom > 1e-12 else math.nan)
    if broad_scores[0] > 0.5 and broad_scores[1] < broad_scores[0] - 0.002:
        return None
    return float(candidate[0]), float(candidate[1])


def try_regional_refinement(selected, preparation_cache, logger=None):
    """Retain the selected attempt unless regional validation succeeds."""
    selected.row["local_warp_status"] = "DISABLED" if not LOCAL_TEXTURE_ENABLED else "SCIPY_UNAVAILABLE" if scipy_ndimage is None else "BASELINE_RETAINED"
    if not LOCAL_TEXTURE_ENABLED or scipy_ndimage is None or preparation_cache.prepared is None:
        return selected
    start = time.perf_counter()
    prepared = preparation_cache.prepared
    usable_baseline = not selected.failure_stage and selected.confidence_percent >= LOW_CONFIDENCE_RETRY_THRESHOLD
    baseline = selected.total_affine if usable_baseline else Affine.identity()
    try:
        local = estimate_local_texture_warp(prepared, baseline)
        if local is None:
            selected.row["local_warp_seconds"] = time.perf_counter() - start
            return selected
        center_row, center_col = prepared.geometry.height / 2.0, prepared.geometry.width / 2.0
        dy, dx = regional_forward_shift(baseline, local, center_row, center_col)
        geometry = geometry_confidence(prepared.nan_stats.footprint_pixels, prepared.moving.size, math.hypot(dy, dx))
        confidence, raw = final_confidence_percent(local.confidence, local.confidence, geometry["geometry_factor_0_1"], False)
        if confidence < LOW_CONFIDENCE_RETRY_THRESHOLD:
            selected.row["local_warp_status"] = "INSUFFICIENT_CONFIDENCE"
            selected.row["local_warp_seconds"] = time.perf_counter() - start
            return selected
        if selected.registration_state is not None and usable_baseline:
            state = RegistrationState(**{**selected.registration_state.__dict__, "local_texture": local})
        else:
            shift = ShiftResult(float(baseline.f), float(baseline.c), math.nan, prepared.overlap_pixels, "regional texture recovery")
            choice = TranslationChoice(shift, False, "regional texture recovery", math.nan, math.nan)
            state = RegistrationState(
                prepared.reference, prepared.moving, prepared.geometry, prepared.nan_stats,
                prepared.eco_nan_dilation_pixels, prepared.support_loss, prepared.overlap_pixels,
                prepared.gradient_overlap_pixels, prepared.robust_gradient_overlap_pixels,
                choice, shift, WarpModel.from_pixel_affine(baseline, choice.method),
                time.perf_counter() - start, local_texture=local)
        row = dict(selected.row)
        row.update(geometry)
        row.update({
            "local_warp_status": "ACCEPTED",
            "local_warp_precision_mode": int(local.precision_mode),
            "local_warp_fit_windows": local.fit_windows,
            "local_warp_validation_windows": local.validation_windows,
            "local_warp_scatter_p90_px": local.scatter_p90_px,
            "local_warp_median_score_gain": local.median_gain,
            "local_warp_confidence_0_100": local.confidence,
            "local_warp_max_gradient": local.gradient_bound,
            "local_warp_seconds": time.perf_counter() - start,
            "local_warp_previous_failure": " | ".join(selected.errors),
            "local_warp_definition": local.serialize(),
            "confidence_percent": confidence, "confidence_raw_0_100": raw,
            "confidence_filename_code_00_99": confidence_filename_code(confidence),
            "registration_method": "regional fine-texture displacement after base affine",
            "total_center_row_shift_px": dy, "total_center_col_shift_px": dx,
        })
        output = AttemptOutcome(**{
            **selected.__dict__, "row": row, "registration_state": state, "total_affine": baseline,
            "registration_confidence": local.confidence, "geometry": geometry,
            "confidence_percent": confidence, "confidence_raw": raw, "failure_stage": "", "errors": [],
        })
        if logger is not None:
            logger.info(f"    Regional warp ACCEPTED | controls={local.fit_windows}, validation={local.validation_windows} | "
                        f"gain={local.median_gain:+.4f} | held-out residual P90={local.scatter_p90_px:.3f}px")
        return output
    except Exception as exc:
        selected.row["local_warp_status"] = "ERROR_BASELINE_RETAINED"
        selected.row["local_warp_error"] = str(exc)
        selected.row["local_warp_seconds"] = time.perf_counter() - start
        if logger is not None:
            logger.info(f"    Regional warp unavailable; existing result retained: {exc}")
        return selected


def try_precision_refinement(selected, preparation_cache, logger=None):
    """Refine a usable affine; preserve confidence and rejection behavior."""
    row = selected.row
    row.update({"precision_status": "BASELINE_RETAINED", "precision_model": "",
                "precision_fit_windows": 0, "precision_validation_windows": 0,
                "precision_median_score_gain": "", "precision_baseline_score": "",
                "precision_candidate_score": "", "precision_max_adjustment_px": "",
                "precision_seconds": 0.0, "precision_error": ""})
    if not PRECISION_ENABLED or scipy_ndimage is None:
        row["precision_status"] = "DISABLED" if not PRECISION_ENABLED else "SCIPY_UNAVAILABLE"
        return selected
    prepared = preparation_cache.prepared
    if (prepared is None or selected.registration_state is None or selected.failure_stage
            or selected.confidence_percent < LOW_CONFIDENCE_RETRY_THRESHOLD):
        row["precision_status"] = "NO_USABLE_BASELINE"
        return selected
    started = time.perf_counter()
    report_row = row
    try:
        evidence = refine_precision_affine(prepared, selected.total_affine)
        if evidence is None:
            if logger is not None:
                logger.info("    Precision BASELINE_RETAINED | no independently supported improvement")
            return selected
        candidate = evidence.candidate_affine
        dy, dx = affine_shift_at(candidate, prepared.geometry.height / 2.0,
                                prepared.geometry.width / 2.0)
        movement = translation_metrics_from_geometry(prepared.geometry, dy, dx)
        geometry = geometry_confidence(prepared.nan_stats.footprint_pixels,
                                       prepared.moving.size, math.hypot(dy, dx))
        assessment_confidence = (selected.assessment.confidence_0_100
                                 if selected.assessment is not None else 0.0)
        confidence, raw = final_confidence_percent(assessment_confidence,
            selected.registration_confidence, geometry["geometry_factor_0_1"], False)
        confidence, raw = min(selected.confidence_percent, confidence), min(selected.confidence_raw, raw)
        if confidence < LOW_CONFIDENCE_RETRY_THRESHOLD:
            row["precision_status"] = "CONFIDENCE_GUARD_BASELINE_RETAINED"
            return selected
        report_row = dict(row)
        report_row.update(geometry)
        report_row.update(affine_components(candidate))
        report_row.update(candidate_affine_components(candidate))
        report_row.update({"precision_status": "ACCEPTED", "precision_model": evidence.model,
                    "precision_fit_windows": evidence.fit_windows,
                    "precision_validation_windows": evidence.validation_windows,
                    "precision_median_score_gain": evidence.median_gain,
                    "precision_baseline_score": evidence.baseline_score,
                    "precision_candidate_score": evidence.candidate_score,
                    "precision_max_adjustment_px": evidence.maximum_displacement_px,
                    "total_center_row_shift_px": dy, "total_center_col_shift_px": dx,
                    "total_translation_distance_m": movement["distance_m"],
                    "total_translation_azimuth_deg": movement["azimuth_deg"],
                    "candidate_total_center_row_shift_px": dy,
                    "candidate_total_center_col_shift_px": dx,
                    "candidate_total_translation_distance_m": movement["distance_m"],
                    "candidate_total_translation_azimuth_deg": movement["azimuth_deg"],
                    "confidence_percent": confidence, "confidence_raw_0_100": raw,
                    "confidence_filename_code_00_99": confidence_filename_code(confidence),
                    "registration_method": str(row.get("registration_method", ""))
                                           + "; continuous precision " + evidence.model})
        if logger is not None:
            logger.info(f"    Precision ACCEPTED | {evidence.model} | "
                        f"held-out gain={evidence.median_gain:+.5f} | "
                        f"max adjustment={evidence.maximum_displacement_px:.3f}px | "
                        f"fit={evidence.fit_windows}, validation={evidence.validation_windows}")
        return AttemptOutcome(**{**selected.__dict__, "row": report_row, "total_affine": candidate,
                                 "geometry": geometry, "confidence_percent": confidence,
                                 "confidence_raw": raw})
    except Exception as exc:
        row["precision_status"] = "ERROR_BASELINE_RETAINED"
        row["precision_error"] = str(exc)
        if logger is not None:
            logger.info(f"    Precision unavailable; existing result retained: {exc}")
        return selected
    finally:
        row["precision_seconds"] = time.perf_counter() - started
        report_row["precision_seconds"] = row["precision_seconds"]


def _process_pipeline_scene(
    index: int,
    total: int,
    input_path: Path,
    reference_path: Optional[Path],
    reference_cache: Optional[CachedRaster],
    output_folder: Path,
    logger: Optional[ConsoleLog],
    setup_error: str = "",
    local_executor: Optional[ThreadPoolExecutor] = None,
) -> PipelineResult:
    total_start = time.perf_counter()
    reference_name = reference_path.name if reference_path else LANDSAT_INPUT_NAME
    if not BATCH_OUTPUTS_PRE_CLEANED:
        remove_previous_scene_outputs(output_folder, input_path)

    if logger is not None:
        logger.info(f"[{index}/{total}] {input_path.name}")

    qa_reference_features: Dict[
        Tuple[int, int, int, int], QAStructuralFeatureVariants
    ] = QAReferenceFeatureStore()
    registration_preparation_cache = RegistrationPreparationCache()
    initial = run_pipeline_attempt(
        index=index,
        total=total,
        input_path=input_path,
        reference_name=reference_name,
        reference_cache=reference_cache,
        setup_error=setup_error,
        attempt_code=0,
        coarse_downsample=COARSE_GLOBAL_DOWNSAMPLE,
        native_search_radius_pixels=NATIVE_REFINE_RADIUS_PIXELS,
        force_robust=False,
        logger=logger,
        qa_reference_features=qa_reference_features,
        registration_preparation_cache=registration_preparation_cache,
        local_executor=local_executor,
    )

    retry_triggered = bool(
        not setup_error
        and initial.confidence_percent < LOW_CONFIDENCE_RETRY_THRESHOLD
    )
    selected = initial
    retry: Optional[AttemptOutcome] = None
    if retry_triggered:
        if logger is not None:
            logger.info(
                f"    Retry 01 triggered | initial confidence "
                f"{initial.confidence_percent:02d}% < "
                f"{LOW_CONFIDENCE_RETRY_THRESHOLD}% | discarding attempt 00 transform"
            )
        release_registration_state(initial.registration_state)
        retry = run_pipeline_attempt(
            index=index,
            total=total,
            input_path=input_path,
            reference_name=reference_name,
            reference_cache=reference_cache,
            setup_error="",
            attempt_code=1,
            coarse_downsample=RETRY_COARSE_GLOBAL_DOWNSAMPLE,
            native_search_radius_pixels=RETRY_NATIVE_SEARCH_RADIUS_PIXELS,
            force_robust=RETRY_FORCE_ROBUST_MATCHER,
            logger=logger,
            qa_reference_features=qa_reference_features,
            registration_preparation_cache=registration_preparation_cache,
            local_executor=local_executor,
        )
        selected = retry

    selected = try_precision_refinement(selected, registration_preparation_cache, logger)
    pre_regional_selected = selected
    selected = try_regional_refinement(selected, registration_preparation_cache, logger)
    registration_preparation_cache.prepared = None
    registration_preparation_cache.native_phase_results.clear()
    registration_preparation_cache.robust_global_result = None
    registration_preparation_cache.robust_global_feature_score = math.nan
    registration_preparation_cache.robust_global_feature_count = 0
    registration_preparation_cache.downstream_key = None
    registration_preparation_cache.downstream_warp_model = None
    registration_preparation_cache.downstream_stage_tail = ()
    registration_preparation_cache.assessment_affine_key = None
    registration_preparation_cache.assessment_result = None

    row = selected.row
    row.update(
        {
            "high_resolution_retry_triggered": int(retry_triggered),
            "initial_confidence_percent": initial.confidence_percent,
            "initial_confidence_raw_0_100": initial.confidence_raw,
            "initial_attempt_status": (
                "RETRY_TRIGGERED_LOW_CONFIDENCE"
                if retry_triggered
                else ("CANDIDATE_OK" if not initial.failure_stage else "CANDIDATE_FAILED")
            ),
            "initial_attempt_failure_stage": initial.failure_stage,
            "initial_attempt_error": " | ".join(dict.fromkeys(initial.errors)),
            "initial_attempt_seconds": initial.processing_seconds,
            "retry_attempt_seconds": retry.processing_seconds if retry is not None else 0.0,
        }
    )

    candidate_succeeded = bool(
        selected.registration_state is not None
        and not selected.failure_stage
    )
    corrections_applied = bool(
        candidate_succeeded
        and selected.confidence_percent >= LOW_CONFIDENCE_RETRY_THRESHOLD
    )
    row["corrections_applied"] = int(corrections_applied)
    applied_affine = selected.total_affine if corrections_applied else Affine.identity()
    applied_local_warp = (selected.registration_state.local_texture
                          if corrections_applied and selected.registration_state is not None else None)
    subpixel_output = bool(corrections_applied)
    row["output_resampling"] = "bilinear" if subpixel_output else "nearest" if corrections_applied else "unchanged"

    if not corrections_applied:
        row.update(affine_components(applied_affine))
        row.update(
            {
                "total_translation_distance_m": 0.0,
                "total_translation_azimuth_deg": 0.0,
                "total_center_row_shift_px": 0.0,
                "total_center_col_shift_px": 0.0,
            }
        )

    final_resampled_band: Optional[np.ndarray] = None
    if selected.registration_state is not None and corrections_applied:
        try:
            if subpixel_output:
                final_resampled_band = resample_original_subpixel(
                    selected.registration_state.moving, applied_affine, applied_local_warp)
            else:
                final_resampled_band = reproject_array_with_pixel_affine(
                    selected.registration_state.moving, selected.registration_state.geometry, applied_affine)
            if applied_local_warp is not None:
                paired_residual = regional_final_residuals(
                    selected.registration_state.reference, final_resampled_band, applied_affine, applied_local_warp)
            elif selected.registration_state.fine_texture is not None:
                paired_residual = fine_texture_final_residuals(
                    selected.registration_state.reference, final_resampled_band, applied_affine)
            else:
                paired_residual = qa_assess_before_after_residuals(
                    selected.registration_state.reference,
                    selected.registration_state.moving,
                    final_resampled_band,
                    applied_affine,
                    reference_feature_cache=qa_reference_features,
                    local_executor=local_executor,
                )
            row["residual_paired_windows"] = paired_residual.paired_windows
            row["residual_quality"] = paired_residual.residual_quality
            if paired_residual.status == "OK":
                row["residual_before_p90_px"] = paired_residual.before_p90_px
                row["residual_after_p90_px"] = paired_residual.after_p90_px
                row["residual_improvement_p90_px"] = paired_residual.improvement_p90_px
                if logger is not None:
                    logger.info(
                        f"    Residual  OK | P90 "
                        f"{paired_residual.before_p90_px:.2f} -> "
                        f"{paired_residual.after_p90_px:.2f} px "
                        f"({paired_residual.improvement_p90_px:+.2f}) | "
                        f"paired={paired_residual.paired_windows} | "
                        f"quality={paired_residual.residual_quality}"
                    )
            elif logger is not None:
                logger.info(
                    f"    Residual  unavailable | paired={paired_residual.paired_windows} | "
                    f"quality={paired_residual.residual_quality} | {paired_residual.error}"
                )
        except Exception as exc:
            final_resampled_band = None
            row["residual_quality"] = "ESTIMATION_FAILED"
            if logger is not None:
                logger.info(f"    Residual  FAILED | {exc}")
    elif selected.registration_state is not None:
        # The output is an unchanged copy of the input. Do not spend another full
        # correlation pass measuring an output that is intentionally identical to
        # the input; no final correction improvement was applied.
        row["residual_paired_windows"] = 0
        row["residual_quality"] = "UNCHANGED_PASSTHROUGH"
        row["residual_improvement_p90_px"] = 0.0
        if logger is not None:
            logger.info(
                "    Residual  skipped | unchanged passthrough; no correction applied"
            )

    release_registration_state(selected.registration_state)
    qa_reference_features.fft_cache.clear()
    qa_reference_features.clear()

    attempt_code = int(row.get("processing_attempt_code") or 0)
    confidence_percent = selected.confidence_percent
    confidence_raw = selected.confidence_raw
    confidence_code = confidence_filename_code(confidence_percent)
    output_path = output_path_for_confidence(
        output_folder,
        input_path,
        confidence_percent,
        attempt_code=attempt_code,
    )

    if corrections_applied:
        status = "OK_LOCAL_WARP" if applied_local_warp is not None else "OK_HIGH_RES_RETRY" if attempt_code == 1 else "OK"
    elif attempt_code == 1:
        status = (
            "RETRY_FAILED_PASSTHROUGH"
            if selected.failure_stage
            else "RETRY_LOW_CONFIDENCE_PASSTHROUGH"
        )
    else:
        status = "PASSTHROUGH_WITH_FAILURE"

    errors = list(selected.errors)
    failure_stage = selected.failure_stage
    polish_start = time.perf_counter()
    try:
        if corrections_applied:
            try:
                write_final_affine_raster(
                    input_path, output_path, applied_affine,
                    first_band_data=final_resampled_band,
                    local_warp=applied_local_warp, subpixel_resampling=subpixel_output)
            except Exception as local_write_error:
                can_restore_affine = (applied_local_warp is not None
                                      and not pre_regional_selected.failure_stage
                                      and pre_regional_selected.confidence_percent >= LOW_CONFIDENCE_RETRY_THRESHOLD)
                if not can_restore_affine:
                    raise
                fallback_affine = pre_regional_selected.total_affine
                fallback_path = output_path_for_confidence(
                    output_folder, input_path, pre_regional_selected.confidence_percent, attempt_code=attempt_code)
                write_final_affine_raster(input_path, fallback_path, fallback_affine,
                                          subpixel_resampling=True)
                if output_path != fallback_path and output_path.exists():
                    output_path.unlink()
                output_path, applied_affine, applied_local_warp = fallback_path, fallback_affine, None
                confidence_percent = pre_regional_selected.confidence_percent
                confidence_raw = pre_regional_selected.confidence_raw
                confidence_code = confidence_filename_code(confidence_percent)
                status = "OK_AFFINE_FALLBACK"
                row.update({
                    "local_warp_status": "WRITE_FAILED_AFFINE_RESTORED", "local_warp_error": str(local_write_error),
                    "output_resampling": "bilinear", "confidence_percent": confidence_percent,
                    "confidence_raw_0_100": confidence_raw, "confidence_filename_code_00_99": confidence_code,
                    "residual_before_p90_px": "", "residual_after_p90_px": "", "residual_improvement_p90_px": "",
                    "residual_paired_windows": 0, "residual_quality": "AFFINE_FALLBACK_NOT_REMEASURED",
                    "registration_method": pre_regional_selected.row.get("registration_method", ""),
                })
                if logger is not None:
                    logger.info(f"    Regional write failed; validated affine restored: {local_write_error}")
        else:
            shutil.copy2(input_path, output_path)
        row["polish_seconds"] = time.perf_counter() - polish_start
    except Exception as exc:
        errors.append(f"final write: {exc}")
        failure_stage = failure_stage or "polish"
        status = (
            "RETRY_POLISH_FAILED_PASSTHROUGH"
            if attempt_code == 1
            else "POLISH_FAILED_PASSTHROUGH"
        )
        confidence_percent = 0
        confidence_raw = 0.0
        confidence_code = 0
        corrections_applied = False
        applied_local_warp = None
        row["output_resampling"] = "unchanged"
        row.update(
            {
                "confidence_percent": 0,
                "confidence_raw_0_100": 0.0,
                "confidence_filename_code_00_99": 0,
                "corrections_applied": 0,
            }
        )
        before_residual_value = row.get("residual_before_p90_px")
        if isinstance(before_residual_value, (int, float)) and math.isfinite(
            float(before_residual_value)
        ):
            row["residual_after_p90_px"] = float(before_residual_value)
            row["residual_improvement_p90_px"] = 0.0
        try:
            if output_path.exists():
                output_path.unlink()
            output_path = output_path_for_confidence(
                output_folder,
                input_path,
                0,
                attempt_code=attempt_code,
            )
            shutil.copy2(input_path, output_path)
            applied_affine = Affine.identity()
            row.update(affine_components(applied_affine))
            row.update(
                {
                    "total_translation_distance_m": 0.0,
                    "total_translation_azimuth_deg": 0.0,
                    "total_center_row_shift_px": 0.0,
                    "total_center_col_shift_px": 0.0,
                }
            )
            row["polish_seconds"] = time.perf_counter() - polish_start
        except Exception as copy_exc:
            errors.append(f"passthrough copy: {copy_exc}")
            status = "OUTPUT_FAILED"
            output_path = Path("")
            row["polish_seconds"] = time.perf_counter() - polish_start

    final_resampled_band = None
    row["input_file"] = input_path.name
    row["status"] = status
    row["failure_stage"] = failure_stage
    row["output_file"] = output_path.name if output_path.name else ""
    row["error"] = " | ".join(dict.fromkeys(errors))
    write_matching_error_raster(
        input_path, output_path, applied_affine, corrections_applied, row,
        local_warp=applied_local_warp,
    )
    row["polish_seconds"] = time.perf_counter() - polish_start
    finalize_correction_record(
        row, applied_affine,
        selected.registration_state.geometry if selected.registration_state is not None else None,
        local_warp=applied_local_warp,
    )
    row["total_processing_seconds"] = time.perf_counter() - total_start

    if logger is not None:
        if output_path.name:
            distance = float(row.get("total_translation_distance_m") or 0.0)
            azimuth = float(row.get("total_translation_azimuth_deg") or 0.0)
            logger.info(
                f"    Polish    {status} | {format_elapsed(row['polish_seconds'])} | "
                f"attempt={attempt_code:02d} | confidence={confidence_percent:02d}% "
                f"(file={confidence_code:02d}) | applied={int(corrections_applied)} | "
                f"total={distance:.1f} m @ {azimuth:.2f} deg"
            )
            logger.info(f"    Wrote     {output_path.name}")
            if row["err_output_file"]:
                logger.info(f"    Wrote ERR {row['err_output_file']} | {row['err_resampling']}")
        else:
            logger.info(f"    Polish    OUTPUT FAILED | {' | '.join(errors)}")

    return PipelineResult(
        index=index,
        input_file=input_path.name,
        output_file=output_path.name if output_path.name else "",
        status=status,
        confidence_percent=confidence_percent,
        row=row,
    )


def process_pipeline_scene(
    index: int,
    total: int,
    input_path: Path,
    reference_path: Optional[Path],
    reference_cache: Optional[CachedRaster],
    output_folder: Path,
    logger: Optional[ConsoleLog],
    setup_error: str = "",
) -> PipelineResult:
    if ACTIVE_LOCAL_WINDOW_WORKERS <= 1:
        return _process_pipeline_scene(
            index,
            total,
            input_path,
            reference_path,
            reference_cache,
            output_folder,
            logger,
            setup_error,
            local_executor=None,
        )
    with ThreadPoolExecutor(
        max_workers=ACTIVE_LOCAL_WINDOW_WORKERS,
        thread_name_prefix="tir_bridge_local",
        initializer=configure_local_cpu_worker,
    ) as local_executor:
        return _process_pipeline_scene(
            index,
            total,
            input_path,
            reference_path,
            reference_cache,
            output_folder,
            logger,
            setup_error,
            local_executor=local_executor,
        )


def _safe_process_pipeline_scene_core(
    index: int,
    total: int,
    input_path: Path,
    reference_path: Optional[Path],
    reference_cache: Optional[CachedRaster],
    output_folder: Path,
    logger: Optional[ConsoleLog],
    setup_error: str = "",
) -> PipelineResult:
    start = time.perf_counter()
    try:
        return process_pipeline_scene(
            index,
            total,
            input_path,
            reference_path,
            reference_cache,
            output_folder,
            logger,
            setup_error,
        )
    except Exception as exc:
        reference_name = reference_path.name if reference_path else LANDSAT_INPUT_NAME
        row = blank_row(index, input_path, reference_name)
        row.update(affine_components(Affine.identity()))
        update_affine_row(row, "register", Affine.identity())
        output_path = output_path_for_confidence(output_folder, input_path, 0)
        error = f"unexpected pipeline failure: {exc}"
        status = "UNEXPECTED_FAILURE_PASSTHROUGH"
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
            remove_previous_scene_outputs(output_folder, input_path)
            shutil.copy2(input_path, output_path)
        except Exception as copy_exc:
            error += f" | passthrough copy: {copy_exc}"
            status = "OUTPUT_FAILED"
            output_path = Path("")

        processed_path = input_path
        row.update(
            {
                "status": status,
                "failure_stage": "unexpected",
                "input_file": processed_path.name,
                "output_file": output_path.name if output_path.name else "",
                "output_resampling": "unchanged" if output_path.name else "",
                "confidence_percent": 0,
                "confidence_raw_0_100": 0.0,
                "confidence_filename_code_00_99": 0,
                "assessment_confidence_0_100": 0.0,
                "registration_confidence_0_100": 0.0,
                "geometry_confidence_0_100": 0.0,
                "geometry_factor_0_1": 0.0,
                "total_translation_distance_m": 0.0,
                "total_translation_azimuth_deg": 0.0,
                "total_center_row_shift_px": 0.0,
                "total_center_col_shift_px": 0.0,
                "total_processing_seconds": time.perf_counter() - start,
                "error": error,
            }
        )
        write_matching_error_raster(
            input_path, output_path, Affine.identity(), False, row)
        finalize_correction_record(row, Affine.identity())
        row["total_processing_seconds"] = time.perf_counter() - start
        if logger is not None:
            logger.info(f"[{index}/{total}] {input_path.name}")
            logger.info(f"    Pipeline  {status} | {error}")
            if output_path.name:
                logger.info(f"    Wrote     {output_path.name}")
            if row["err_output_file"]:
                logger.info(f"    Wrote ERR {row['err_output_file']} | unchanged")
        return PipelineResult(
            index=index,
            input_file=processed_path.name,
            output_file=output_path.name if output_path.name else "",
            status=status,
            confidence_percent=0,
            row=row,
        )


def safe_process_pipeline_scene(*args, **kwargs) -> PipelineResult:
    previous = getattr(CPU_THREAD_STATE, "performance", None)
    metrics = {} if PERFORMANCE_ENABLED else None
    CPU_THREAD_STATE.performance = metrics
    try:
        result = _safe_process_pipeline_scene_core(*args, **kwargs)
        if metrics is not None:
            metrics["write_seconds"] = float(result.row.get("polish_seconds") or 0.0)
        result.performance = metrics or {}
        return result
    finally:
        CPU_THREAD_STATE.performance = previous

CSV_BASE_FIELDS = (
    "input_index", "timestamp", "input_file", "source_folder", "source_directory",
    "reference_file", "reference_path",
    "residual_before_p90_px", "residual_after_p90_px", "residual_improvement_p90_px",
    "residual_paired_windows", "residual_quality",
    "output_file", "output_directory",
    "total_translation_azimuth_deg", "total_translation_distance_m",
    "total_center_row_shift_px", "total_center_col_shift_px",
    "total_pixel_affine_a", "total_pixel_affine_b", "total_pixel_affine_c",
    "total_pixel_affine_d", "total_pixel_affine_e", "total_pixel_affine_f",
    "total_warp_row_offset_px", "total_warp_row_col_coef", "total_warp_row_row_coef",
    "total_warp_col_offset_px", "total_warp_col_col_coef", "total_warp_col_row_coef",
    "translation_anchor_col_px", "translation_anchor_row_px",
    "source_crs", "source_width_px", "source_height_px",
    "translation_distance_basis", "translation_azimuth_reference",
    "source_xy_units", "source_units_to_m",
    "status", "corrections_applied", "processing_attempt_code",
    "high_resolution_retry_triggered", "confidence_percent",
    "fine_texture_status", "fine_texture_fit_windows", "fine_texture_validation_windows",
    "fine_texture_scatter_p90_px", "fine_texture_median_score_gain", "fine_texture_seconds",
    "fine_texture_previous_failure", "fine_texture_error",
    "precision_status", "precision_model", "precision_fit_windows", "precision_validation_windows",
    "precision_median_score_gain", "precision_baseline_score", "precision_candidate_score",
    "precision_max_adjustment_px", "precision_seconds", "precision_error",
    "transform_model", "output_resampling",
    "base_pixel_affine_a", "base_pixel_affine_b", "base_pixel_affine_c",
    "base_pixel_affine_d", "base_pixel_affine_e", "base_pixel_affine_f",
    "local_warp_status", "local_warp_fit_windows", "local_warp_validation_windows",
    "local_warp_scatter_p90_px", "local_warp_median_score_gain", "local_warp_seconds",
    "local_warp_confidence_0_100", "local_warp_max_gradient",
    "local_warp_precision_mode",
    "local_warp_previous_failure", "local_warp_error", "local_warp_definition",
    "register_seconds", "assess_seconds", "polish_seconds", "total_processing_seconds",
    "failure_stage", "error", "correction_record_error",
    "err_input_file", "err_output_file", "err_status", "err_resampling",
    "err_write_seconds", "err_error",
)
CSV_PROFILE_METRICS = (
    "preparation_seconds", "ncc_seconds", "local_warp_seconds",
    "validation_seconds", "assessment_seconds", "final_residual_seconds",
    "resample_seconds", "write_seconds", "gpu_queue_seconds",
    "gpu_job_seconds", "gpu_jobs",
)
CSV_FIELDS = CSV_BASE_FIELDS + (
    tuple("profile_" + name for name in CSV_PROFILE_METRICS)
    if PERFORMANCE_ENABLED else ()
)

CSV_FIELD_SET = frozenset(CSV_FIELDS)


class BufferedCorrectionCSV:
    """Queue row references; serialize once per completed dataset."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = None
        self.writer = None
        self.rows_written = 0
        self.serialization_write_seconds = 0.0
        self._pending = []
        self._write_failed = False

    def __enter__(self):
        start = time.perf_counter()
        try:
            self.file = self.path.open(
                "w", newline="", encoding="utf-8", buffering=CSV_BUFFER_BYTES
            )
            self.writer = csv.writer(self.file)
            self.writer.writerow(CSV_FIELDS)
        except BaseException:
            if self.file is not None:
                self.file.close()
            self.file = None
            self.writer = None
            raise
        finally:
            self.serialization_write_seconds += time.perf_counter() - start
        return self

    def append(self, row: Dict[str, object]) -> None:
        if self.writer is None or self._write_failed:
            raise RuntimeError("The correction CSV is not writable")
        self._pending.append(row)

    def flush(self) -> None:
        if self.file is None:
            return
        if self._write_failed:
            raise RuntimeError("A previous CSV write failed; partial rows will not be replayed")
        start = time.perf_counter()
        try:
            if self._pending:
                fields = CSV_FIELDS
                getter = itemgetter(*fields)
                self.writer.writerows(
                    getter(row) if row.keys() >= CSV_FIELD_SET
                    else tuple(row.get(name, "") for name in fields)
                    for row in self._pending
                )
                self.rows_written += len(self._pending)
                self._pending.clear()
            self.file.flush()
        except BaseException:
            self._write_failed = True
            raise
        finally:
            self.serialization_write_seconds += time.perf_counter() - start

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if not self._write_failed:
                self.flush()
        finally:
            start = time.perf_counter()
            try:
                if self.file is not None:
                    self.file.close()
            finally:
                self.serialization_write_seconds += time.perf_counter() - start
                self.writer = None
                self.file = None
        return False


# Optional inclusive timers share the correction CSV; no separate report.
PROCESS_WORKERS_REQUESTED = max(0, int(os.environ.get("TIR_BRIDGE_PROCESSES", "0")))
_PROCESS_LANE = os.environ.get("TIR_BRIDGE_PROCESS_LANE", "0") == "1"
_PROCESS_POOL = None
_PROCESS_TASK_QUEUE = None
_PROCESS_RESULT_QUEUE = None
_PROCESS_CANCEL_EVENT = None
_LANE_TASK_QUEUE = None
_LANE_RESULT_QUEUE = None
_LANE_CANCEL_EVENT = None
_LANE_SCENES = 1
ROLLING_LANES_MODE = os.environ.get("TIR_BRIDGE_ROLLING_LANES", "auto").strip().casefold()
ROLLING_LANES_ENABLED = (
    ROLLING_LANES_MODE not in {"0", "off", "false", "no"}
    and (ROLLING_LANES_MODE != "auto" or gpu_is_active())
)
_LANE_SCENE_POOL = None
_LANE_REFERENCE = None
_LANE_REFERENCE_KEY = None
_LANE_RUNTIME_KEY = None
_LANE_REFERENCE_HITS = 0
_LANE_REFERENCE_MISSES = 0
_PARENT_BLAS_SETTINGS = ()
_LANE_BLAS_CONTROLLERS = ()


def _parent_blas_settings():
    if ThreadpoolController is None:
        return ()
    return tuple({key: info.get(key) for key in
                  ("filepath", "internal_api", "version", "architecture", "num_threads")}
                 for info in ThreadpoolController().info() if info.get("user_api") == "blas")


def _restore_parent_blas_settings() -> None:
    global _LANE_BLAS_CONTROLLERS
    if not _PARENT_BLAS_SETTINGS:
        return
    if ThreadpoolController is None:
        raise RuntimeError("threadpoolctl is required to preserve the coordinator BLAS configuration")
    controller = ThreadpoolController()
    selected = []
    for expected in _PARENT_BLAS_SETTINGS:
        path = os.path.normcase(os.path.realpath(expected["filepath"]))
        matches = [lib for lib in controller.lib_controllers
                   if os.path.normcase(os.path.realpath(lib.filepath)) == path]
        if len(matches) != 1:
            raise RuntimeError(f"Could not match the coordinator BLAS library: {path}")
        lib = matches[0]
        desired = int(expected["num_threads"])
        if lib.num_threads != desired:
            lib.set_num_threads(desired)
        if lib.num_threads != desired:
            raise RuntimeError(f"Could not preserve {path} BLAS threads={desired}")
        selected.append((lib, desired))
    _LANE_BLAS_CONTROLLERS = tuple(selected)


def _configure_lane_scene_thread() -> None:
    # Also covers libraries whose thread limits are thread-local.
    for lib, desired in _LANE_BLAS_CONTROLLERS:
        if lib.num_threads != desired:
            lib.set_num_threads(desired)



def _timed_stage(name, function):
    @wraps(function)
    def timed(*args, **kwargs):
        metrics = getattr(CPU_THREAD_STATE, "performance", None)
        if metrics is None:
            return function(*args, **kwargs)
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            metrics[name] = metrics.get(name, 0.0) + time.perf_counter() - start
    return timed


def install_performance_timers() -> None:
    if not PERFORMANCE_ENABLED:
        return
    stages = {
        "register_scene": "register_seconds",
        "prepare_registration_scene": "preparation_seconds",
        "normalized_correlations_at_shifts": "ncc_seconds",
        "estimate_linear_warp": "local_warp_seconds",
        "validate_alignment_candidate": "validation_seconds",
        "qa_assess_registered": "assessment_seconds",
        "qa_assess_before_after_residuals": "final_residual_seconds",
        "reproject_array_with_pixel_affine": "resample_seconds",
        "write_final_affine_raster": "write_seconds",
    }
    for function_name, label in stages.items():
        function = globals()[function_name]
        if not getattr(function, "_tir_timed", False):
            wrapped = _timed_stage(label, function)
            wrapped._tir_timed = True
            globals()[function_name] = wrapped


def runtime_snapshot() -> dict:
    names = (
        "ACTIVE_SCENE_WORKERS", "ACTIVE_CPU_FFT_WORKERS",
        "ACTIVE_GDAL_WARP_THREADS", "ACTIVE_TIFF_THREADS",
        "ACTIVE_LOCAL_WINDOW_WORKERS", "ACTIVE_LOCAL_FFT_WORKERS",
        "GDAL_WARP_MEMORY_MB", "TIFF_DEFLATE_LEVEL",
        "BATCH_OUTPUTS_PRE_CLEANED", "QA_REFERENCE_FFT_CACHE_MAX_BYTES",
    )
    snapshot = {name: globals()[name] for name in names}
    snapshot["_PARENT_BLAS_SETTINGS"] = _parent_blas_settings()
    return snapshot


def choose_process_lanes(workers: int, paths: Sequence[Path], largest_pixels: Optional[int] = None) -> int:
    if _PROCESS_LANE or workers < 2:
        return 1
    if ThreadpoolController is None and _NUMPY_PRELOADED:
        return 1
    cores = max(1, int(os.cpu_count() or 1))
    requested = PROCESS_WORKERS_REQUESTED
    if requested == 1:
        return 1
    # Small batches cannot amortize interpreter/CUDA initialization.
    if requested == 0 and (cores < 8 or len(paths) < 16):
        return 1
    target = requested or (4 if cores >= 24 else 2)
    if requested == 0 and ThreadpoolController is not None:
        try:
            blas_threads = max((int(item["num_threads"]) for item in _parent_blas_settings()), default=1)
            target = min(target, max(1, cores // max(1, blas_threads)))
        except Exception:
            return 1
    target = max(1, min(target, workers, 8))
    # Equal lane widths retain the existing maximum scene concurrency.
    while target > 1 and workers % target:
        target -= 1
    if gpu_is_active():
        pixels = largest_input_pixel_count(paths) if largest_pixels is None else largest_pixels
        qa_pixels = min(QA_WINDOW_SIZE_PIXELS, int(math.sqrt(pixels))) ** 2
        qa_estimate = int(300 * QA_GRID_ROWS * QA_GRID_COLS * qa_pixels)
        job_estimate = max(384 * pixels, qa_estimate, 512 * 1024 ** 2)
        # Budget separate pools, FFT plans, and CUDA contexts before spawning.
        safe_lanes = max(1, int(GPU_TOTAL_MEMORY_BYTES * 0.72) // max(1, int(1.35 * job_estimate)))
        target = min(target, safe_lanes)
        while target > 1 and workers % target:
            target -= 1
    available = available_system_memory_bytes()
    if available > 0 and _PROCESS_POOL is None:
        target = min(target, max(1, available // (2 * 1024 ** 3)))
        while target > 1 and workers % target:
            target -= 1
    return max(1, int(target))


def _initialize_process_lane(lanes: int, scenes_per_lane: int, gpu_enabled: bool,
                             task_queue=None, result_queue=None, cancel_event=None) -> None:
    global _PROCESS_LANE, GPU_JOB_WORKERS_REQUESTED, GPU_FFT_PLAN_CACHE_BYTES
    global _LANE_SCENE_POOL, QA_REFERENCE_FFT_CACHE_MAX_BYTES
    global _LANE_TASK_QUEUE, _LANE_RESULT_QUEUE, _LANE_CANCEL_EVENT, _LANE_SCENES
    _PROCESS_LANE = True
    _LANE_TASK_QUEUE, _LANE_RESULT_QUEUE = task_queue, result_queue
    _LANE_CANCEL_EVENT, _LANE_SCENES = cancel_event, scenes_per_lane
    GPU_JOB_WORKERS_REQUESTED = 1
    GPU_FFT_PLAN_CACHE_BYTES = max(256 * 1024 ** 2, GPU_FFT_PLAN_CACHE_BYTES // max(1, lanes))
    QA_REFERENCE_FFT_CACHE_MAX_BYTES = max(0, QA_REFERENCE_FFT_CACHE_MAX_BYTES)
    if gpu_enabled:
        initialize_gpu_runtime()
        if not gpu_is_active():
            raise RuntimeError(f"A worker could not initialize the requested GPU: {CUPY_IMPORT_ERROR}")
        # Cached free blocks are released when another lane needs VRAM.
        with cp.cuda.Device(GPU_DEVICE_ID):
            cp.get_default_memory_pool().set_limit(size=int(GPU_TOTAL_MEMORY_BYTES * 0.78 / lanes))
    initialize_native_kernels()
    install_performance_timers()
    _LANE_SCENE_POOL = ThreadPoolExecutor(max_workers=scenes_per_lane, thread_name_prefix="tir_scene",
                                          initializer=_configure_lane_scene_thread)
    atexit.register(shutdown_lane)


def shutdown_lane() -> None:
    global _LANE_SCENE_POOL
    if _LANE_SCENE_POOL is not None:
        _LANE_SCENE_POOL.shutdown(wait=True)
        _LANE_SCENE_POOL = None
    if _LANE_CANCEL_EVENT is not None and _LANE_CANCEL_EVENT.is_set() and _LANE_RESULT_QUEUE is not None:
        _LANE_RESULT_QUEUE.cancel_join_thread()
    shutdown_gpu_runtime()


def _lane_chunk(tasks, reference_path, output_folder, setup_error, configuration):
    global _LANE_REFERENCE, _LANE_REFERENCE_KEY, _LANE_RUNTIME_KEY
    global _LANE_REFERENCE_HITS, _LANE_REFERENCE_MISSES
    for name, value in configuration.items():
        globals()[name] = value
    reference_key = (str(reference_path), str(output_folder))
    runtime_key = tuple(sorted(configuration.items()))
    if reference_key != _LANE_REFERENCE_KEY or runtime_key != _LANE_RUNTIME_KEY:
        _restore_parent_blas_settings()
        clear_reference_grid_cache()
        _LANE_REFERENCE = None
        if reference_path is not None and not setup_error:
            _LANE_REFERENCE = load_cached_raster(Path(reference_path))
        _LANE_REFERENCE_KEY = reference_key
        _LANE_RUNTIME_KEY = runtime_key
    hits_before = REFERENCE_GRID_CACHE_HITS
    misses_before = REFERENCE_GRID_CACHE_MISSES
    futures = [
        _LANE_SCENE_POOL.submit(
            safe_process_pipeline_scene, index, total, Path(path),
            Path(reference_path) if reference_path else None,
            _LANE_REFERENCE, Path(output_folder), None, setup_error,
        ) for index, total, path in tasks
    ]
    results = [future.result() for future in as_completed(futures)]
    return (results, REFERENCE_GRID_CACHE_HITS - hits_before,
            REFERENCE_GRID_CACHE_MISSES - misses_before)


def _run_process_lanes_chunked(dataset, index_offset, total_scenes, reference_path,
                               setup_error, workers, lanes, record_completion) -> None:
    global _PROCESS_POOL, REFERENCE_GRID_CACHE_HITS, REFERENCE_GRID_CACHE_MISSES
    global GPU_FFT_PLAN_CACHE_BYTES
    scenes_per_lane = max(1, workers // lanes)
    key = (lanes, scenes_per_lane, gpu_is_active())
    if _PROCESS_POOL is not None and _PROCESS_POOL[0] != key:
        shutdown_process_lanes()
    if _PROCESS_POOL is None:
        # Spawn only: no inherited GDAL/CUDA state or forked BLAS pools.
        os.environ["TIR_BRIDGE_PROCESS_LANE"] = "1"
        try:
            executor = ProcessPoolExecutor(
                max_workers=lanes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process_lane,
                initargs=(lanes, scenes_per_lane, gpu_is_active()),
            )
            _PROCESS_POOL = (key, executor)
            # Workers start lazily; retain the marker through submission.
        except Exception:
            os.environ.pop("TIR_BRIDGE_PROCESS_LANE", None)
            raise
    executor = _PROCESS_POOL[1]
    configuration = runtime_snapshot()
    tasks = [
        (index_offset + local_index, total_scenes, str(path))
        for local_index, path in enumerate(dataset.eco_paths, start=1)
    ]
    try:
        futures = [executor.submit(
            _lane_chunk, tasks[start:start + scenes_per_lane],
            str(reference_path) if reference_path else None,
            str(dataset.output_folder), setup_error, configuration,
        ) for start in range(0, len(tasks), scenes_per_lane)]
    finally:
        os.environ.pop("TIR_BRIDGE_PROCESS_LANE", None)
    for future in as_completed(futures):
        lane_results, hits, misses = future.result()
        REFERENCE_GRID_CACHE_HITS += hits
        REFERENCE_GRID_CACHE_MISSES += misses
        for result in lane_results:
            record_completion(result.index - index_offset, result.index, result)


def _lane_stream(reference_path, output_folder, setup_error, configuration):
    """Refill each scene slot independently from a shared work queue."""
    global _LANE_REFERENCE, _LANE_REFERENCE_KEY, _LANE_RUNTIME_KEY
    for name, value in configuration.items():
        globals()[name] = value
    reference_key = (str(reference_path), str(output_folder))
    runtime_key = tuple(sorted(configuration.items()))
    if reference_key != _LANE_REFERENCE_KEY or runtime_key != _LANE_RUNTIME_KEY:
        _restore_parent_blas_settings()
        clear_reference_grid_cache()
        _LANE_REFERENCE = None
        if reference_path is not None and not setup_error:
            _LANE_REFERENCE = load_cached_raster(Path(reference_path))
        _LANE_REFERENCE_KEY = reference_key
        _LANE_RUNTIME_KEY = runtime_key
    hits_before, misses_before = REFERENCE_GRID_CACHE_HITS, REFERENCE_GRID_CACHE_MISSES
    reference = Path(reference_path) if reference_path else None
    output = Path(output_folder)

    def consume():
        count = 0
        try:
            while not _LANE_CANCEL_EVENT.is_set():
                task = _LANE_TASK_QUEUE.get()
                if task is None:
                    break
                index, total, path = task
                result = safe_process_pipeline_scene(
                    index, total, Path(path), reference, _LANE_REFERENCE,
                    output, None, setup_error,
                )
                # Serialize here so failures cannot be lost in Queue's feeder.
                payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                _LANE_RESULT_QUEUE.put(payload)
                count += 1
        except BaseException:
            _LANE_CANCEL_EVENT.set()
            raise
        return count

    futures = [_LANE_SCENE_POOL.submit(consume) for _ in range(_LANE_SCENES)]
    completed = 0
    try:
        for future in as_completed(futures):
            completed += future.result()
    finally:
        # Never reconfigure a reference while another scene still uses it.
        for future in futures:
            try:
                future.result()
            except BaseException:
                _LANE_CANCEL_EVENT.set()
    return completed, REFERENCE_GRID_CACHE_HITS - hits_before, REFERENCE_GRID_CACHE_MISSES - misses_before


def _close_process_queues() -> None:
    global _PROCESS_TASK_QUEUE, _PROCESS_RESULT_QUEUE, _PROCESS_CANCEL_EVENT
    for work_queue in (_PROCESS_TASK_QUEUE, _PROCESS_RESULT_QUEUE):
        if work_queue is not None:
            # Also safe after cancellation with unconsumed queued tasks.
            work_queue.cancel_join_thread()
            work_queue.close()
    _PROCESS_TASK_QUEUE = _PROCESS_RESULT_QUEUE = _PROCESS_CANCEL_EVENT = None


def run_process_lanes(dataset, index_offset, total_scenes, reference_path,
                      setup_error, workers, lanes, record_completion) -> None:
    if not ROLLING_LANES_ENABLED:
        return _run_process_lanes_chunked(
            dataset, index_offset, total_scenes, reference_path,
            setup_error, workers, lanes, record_completion,
        )
    global _PROCESS_POOL, _PROCESS_TASK_QUEUE, _PROCESS_RESULT_QUEUE, _PROCESS_CANCEL_EVENT
    global REFERENCE_GRID_CACHE_HITS, REFERENCE_GRID_CACHE_MISSES
    scenes_per_lane = max(1, workers // lanes)
    key = (lanes, scenes_per_lane, gpu_is_active(), "rolling")
    if _PROCESS_POOL is not None and _PROCESS_POOL[0] != key:
        shutdown_process_lanes()
    prior_marker = os.environ.get("TIR_BRIDGE_PROCESS_LANE")
    if _PROCESS_POOL is None:
        context = multiprocessing.get_context("spawn")
        _PROCESS_TASK_QUEUE = context.Queue()
        _PROCESS_RESULT_QUEUE = context.Queue()
        _PROCESS_CANCEL_EVENT = context.Event()
        os.environ["TIR_BRIDGE_PROCESS_LANE"] = "1"
        try:
            executor = ProcessPoolExecutor(
                max_workers=lanes, mp_context=context,
                initializer=_initialize_process_lane,
                initargs=(lanes, scenes_per_lane, gpu_is_active(),
                          _PROCESS_TASK_QUEUE, _PROCESS_RESULT_QUEUE, _PROCESS_CANCEL_EVENT),
            )
            _PROCESS_POOL = (key, executor)
        except BaseException:
            _close_process_queues()
            if prior_marker is None:
                os.environ.pop("TIR_BRIDGE_PROCESS_LANE", None)
            else:
                os.environ["TIR_BRIDGE_PROCESS_LANE"] = prior_marker
            raise
    executor = _PROCESS_POOL[1]
    _PROCESS_CANCEL_EVENT.clear()
    futures = []
    try:
        configuration = runtime_snapshot()
        for local_index, path in enumerate(dataset.eco_paths, start=1):
            _PROCESS_TASK_QUEUE.put((index_offset + local_index, total_scenes, str(path)))
        for _ in range(lanes * scenes_per_lane):
            _PROCESS_TASK_QUEUE.put(None)
        try:
            for _ in range(lanes):
                futures.append(executor.submit(
                    _lane_stream, str(reference_path) if reference_path else None,
                    str(dataset.output_folder), setup_error, configuration,
                ))
        finally:
            if prior_marker is None:
                os.environ.pop("TIR_BRIDGE_PROCESS_LANE", None)
            else:
                os.environ["TIR_BRIDGE_PROCESS_LANE"] = prior_marker
        remaining = len(dataset.eco_paths)
        seen = set()
        finished_without_messages = 0
        while remaining:
            try:
                payload = _PROCESS_RESULT_QUEUE.get(timeout=0.1)
            except Empty:
                for future in futures:
                    if future.done():
                        future.result()
                if all(future.done() for future in futures):
                    finished_without_messages += 1
                    if finished_without_messages >= 50:
                        raise RuntimeError(f"Missing {remaining} process result(s)")
                continue
            result = pickle.loads(payload)
            local_index = result.index - index_offset
            if not 1 <= local_index <= len(dataset.eco_paths) or result.index in seen:
                raise RuntimeError("Unexpected or duplicate process result")
            seen.add(result.index)
            record_completion(local_index, result.index, result)
            remaining -= 1
        completed = 0
        for future in futures:
            count, hits, misses = future.result()
            completed += count
            REFERENCE_GRID_CACHE_HITS += hits
            REFERENCE_GRID_CACHE_MISSES += misses
        if completed != len(dataset.eco_paths):
            raise RuntimeError("Process completion accounting mismatch")
    except BaseException:
        _PROCESS_CANCEL_EVENT.set()
        # Drain transfers while in-flight scenes finish; never resubmit a TIFF.
        while any(not future.done() for future in futures):
            try:
                _PROCESS_RESULT_QUEUE.get(timeout=0.1)
            except Empty:
                pass
        shutdown_process_lanes()
        raise
    finally:
        if prior_marker is None:
            os.environ.pop("TIR_BRIDGE_PROCESS_LANE", None)
        else:
            os.environ["TIR_BRIDGE_PROCESS_LANE"] = prior_marker


def shutdown_process_lanes() -> None:
    global _PROCESS_POOL
    try:
        if _PROCESS_POOL is not None:
            _PROCESS_POOL[1].shutdown(wait=True, cancel_futures=False)
    finally:
        _PROCESS_POOL = None
        _close_process_queues()


def log_performance_summary(results, logger: ConsoleLog) -> None:
    if not PERFORMANCE_ENABLED:
        return
    measured = [result for result in results if result is not None]
    totals = {}
    for result in measured:
        for name, value in result.performance.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    count = max(1, len(measured))
    logger.info("Stage means/file: " + " | ".join(
        f"{label}={totals.get(name, 0.0) / count:.3f}s" for label, name in (
            ("register", "register_seconds"), ("assess", "assessment_seconds"),
            ("final residual", "final_residual_seconds"), ("write", "write_seconds"),
            ("GPU queue", "gpu_queue_seconds"),
        )
    ))



def main(root_folder: Optional[Path] = None) -> None:
    global BATCH_OUTPUTS_PRE_CLEANED
    BATCH_OUTPUTS_PRE_CLEANED = False
    logger = ConsoleLog()
    native_start = time.perf_counter()
    initialize_native_kernels()
    native_startup_seconds = time.perf_counter() - native_start
    install_performance_timers()
    logger.banner("TIR-ReCAST v2.26 - REGISTER -> ASSESS -> PRECISION -> REGIONAL WARP -> POLISH + ERR")
    folder = (
        Path(root_folder).resolve()
        if root_folder is not None
        else Path(__file__).resolve().parent
    )
    datasets = discover_dataset_batches(folder)
    total_scenes = sum(len(dataset.eco_paths) for dataset in datasets)
    available_memory = available_system_memory_bytes()

    logger.info(f"Workspace: {folder}")
    logger.info(f"Datasets: {len(datasets)} folder(s)")
    logger.info(f"Inputs: {total_scenes} {ECO_INPUT_SUFFIX}.tif scene(s) total")
    logger.info(f"Outputs: one {OUTPUT_FOLDER_NAME} folder per dataset")
    logger.info("CRS: original scene CRS and grid retained; reference sampled directly to each scene")
    logger.info("Projected shifts: native grid distance in metres; azimuth clockwise from grid north")
    logger.info("Subpixel sampling: consistent bilinear refinement, assessment, and final output")
    logger.info(f"Continuous precision refinement: {'enabled' if PRECISION_ENABLED and scipy_ndimage is not None else 'disabled'} | independent validation")
    csv_path = folder / OUTPUT_CSV_NAME
    logger.info(f"Corrections CSV: {csv_path}")
    logger.info("CSV: RAM-queued rows; one buffered write per completed dataset")
    logger.info("Disk output: no JSON report or Python bytecode files")
    if NATIVE_DISK_CACHE_ENABLED:
        logger.info(f"Compiled-kernel cache: {NATIVE_CACHE_DIRECTORY}")
    else:
        logger.info("Compiled-kernel cache: disabled; compilation retained in process memory")
    if NATIVE_CACHE_ERROR:
        logger.info(f"Compiled-kernel cache warning: {NATIVE_CACHE_ERROR}")
    logger.info(f"Coordinator native warm-up: {native_startup_seconds:.3f} s (included in throughput)")
    if gpu_is_active():
        logger.info(
            f"Compute: CUDA GPU {GPU_DEVICE_ID} | {GPU_DEVICE_NAME} | "
            f"VRAM={format_bytes(GPU_TOTAL_MEMORY_BYTES)} | "
            f"CUDA streams={GPU_JOB_WORKERS_ACTIVE} | CPU fallback enabled"
        )
    else:
        reason = CUPY_IMPORT_ERROR or "CuPy/CUDA not requested or unavailable"
        logger.info(f"Compute: optimized CPU | GPU inactive ({reason})")
    logger.info(
        f"CPU FFT backend: {'SciPy pocketfft' if scipy_fft is not None else 'NumPy FFT'}; "
        f"available RAM={format_bytes(available_memory) if available_memory else 'unknown'}"
    )
    logger.info("Parallelism: adapted to each dataset folder; folders run sequentially")
    logger.info(f"Native CPU preparation: {'enabled' if NATIVE_KERNELS_ACTIVE else 'NumPy fallback'} | extra exact kernels={EXACT_CPU_KERNELS_ACTIVE}")
    if not NATIVE_KERNELS_ACTIVE:
        logger.info("Optional acceleration: install NumPy-compatible Numba; see install_recast_speed_support.py")
    logger.info(
        f"Low-confidence fallback: retry when confidence < "
        f"{LOW_CONFIDENCE_RETRY_THRESHOLD}; retry=01; coarse catch skipped; "
        f"native radius=+/-{RETRY_NATIVE_SEARCH_RADIUS_PIXELS}px; robust forced"
    )
    logger.info(
        f"Native fine-texture refinement: {'enabled' if FINE_TEXTURE_ENABLED and scipy_ndimage is not None else 'disabled'} | "
        f"+/-{FINE_TEXTURE_SEARCH_RADIUS}px | independent fit/validation windows"
    )
    if FINE_TEXTURE_ENABLED and scipy_ndimage is None:
        logger.info("Install scipy to enable fine-texture refinement; existing registration remains available.")
    logger.info(
        f"Regional fine-texture warp: {'enabled' if LOCAL_TEXTURE_ENABLED and scipy_ndimage is not None else 'disabled'} | "
        "independent regional shifts | bilinear single-pass output for accepted refinements"
    )

    results: List[Optional[PipelineResult]] = [None] * total_scenes
    completed_global = 0
    index_offset = 0
    BATCH_OUTPUTS_PRE_CLEANED = True

    with BufferedCorrectionCSV(csv_path) as correction_csv:
        for dataset_number, dataset in enumerate(datasets, start=1):
            dataset_label = (
                "."
                if dataset.folder == folder
                else dataset.folder.relative_to(folder).as_posix()
            )
            dataset.output_folder.mkdir(parents=True, exist_ok=True)
            remove_previous_batch_outputs(dataset.output_folder, dataset.eco_paths)
            clear_reference_grid_cache()
            largest_pixels = largest_input_pixel_count(dataset.eco_paths)
            workers = configure_runtime(dataset.eco_paths, largest_pixels)
            lanes = choose_process_lanes(workers, dataset.eco_paths, largest_pixels)
            if lanes == 1:
                shutdown_process_lanes()

            reference_path: Optional[Path] = None
            reference_cache: Optional[CachedRaster] = None
            setup_error = ""
            try:
                reference_path = find_case_insensitive(dataset.folder, LANDSAT_INPUT_NAME)
                reference_cache = load_cached_raster(reference_path)
            except Exception as exc:
                setup_error = str(exc)

            logger.section(
                f"DATASET {dataset_number}/{len(datasets)} | {dataset_label} | "
                f"{len(dataset.eco_paths)} scene(s)"
            )
            logger.info(f"Folder: {dataset.folder}")
            logger.info(
                f"Reference: {reference_path.name if reference_path else LANDSAT_INPUT_NAME}"
            )
            logger.info(f"Output: {dataset.output_folder}")
            logger.info(
                f"Parallelism: scenes={workers}; "
                f"local windows/scene={ACTIVE_LOCAL_WINDOW_WORKERS}; "
                f"FFT workers/scene={ACTIVE_CPU_FFT_WORKERS}; "
                f"FFT workers/local={ACTIVE_LOCAL_FFT_WORKERS}; "
                f"GDAL warp threads/scene={ACTIVE_GDAL_WARP_THREADS}; "
                f"TIFF threads/scene={ACTIVE_TIFF_THREADS}"
            )
            logger.info(f"Execution: {lanes} process(es); {workers} total scene slots; "
                        f"scheduler={'rolling' if lanes > 1 and ROLLING_LANES_ENABLED else 'standard'}; "
                        "existing per-scene FFT/GDAL settings retained")
            if lanes > 1 and gpu_is_active():
                logger.info(f"CUDA execution: {lanes} worker contexts, one stream each; coordinator GPU executor idle")
            if setup_error:
                logger.info(
                    "Reference setup failed for this dataset; all of its scenes will "
                    f"be copied with confidence 00: {setup_error}"
                )

            folder_results: List[Optional[PipelineResult]] = [
                None
            ] * len(dataset.eco_paths)
            record_context = {
                "source_folder": dataset.folder.name,
                "source_directory": str(dataset.folder),
                "reference_file": reference_path.name if reference_path is not None else "",
                "reference_path": str(reference_path) if reference_path is not None else "",
            }
            output_directory = str(dataset.output_folder)

            def record_completion(
                local_index: int,
                global_index: int,
                result: PipelineResult,
            ) -> None:
                nonlocal completed_global
                folder_results[local_index - 1] = result
                results[global_index - 1] = result
                result.row.update(record_context)
                result.row["output_directory"] = output_directory if result.output_file else ""
                if PERFORMANCE_ENABLED:
                    result.row.update(
                        ("profile_" + name, result.performance.get(name, 0.0))
                        for name in CSV_PROFILE_METRICS
                    )
                correction_csv.append(result.row)
                if result.row.get("correction_record_error"):
                    logger.info(
                        f"Correction record warning: {result.input_file}: "
                        f"{result.row['correction_record_error']}"
                    )
                if result.row.get("err_error"):
                    logger.info(
                        f"Error-raster warning: {result.input_file}: "
                        f"{result.row['err_error']}"
                    )
                completed_global += 1
                attempt_code = int(result.row.get("processing_attempt_code") or 0)
                logger.progress(
                    completed_global,
                    total_scenes,
                    f"{result.status} | A{attempt_code:02d} "
                    f"C{confidence_filename_code(result.confidence_percent):02d} | "
                    f"ERR={result.row.get('err_status', '')} | "
                    f"{dataset_label}/{result.input_file}",
                )

            if lanes > 1 and not setup_error:
                run_process_lanes(dataset, index_offset, total_scenes, reference_path,
                                  setup_error, workers, lanes, record_completion)
            elif workers == 1:
                for local_index, eco_path in enumerate(dataset.eco_paths, start=1):
                    global_index = index_offset + local_index
                    result = safe_process_pipeline_scene(
                        global_index,
                        total_scenes,
                        eco_path,
                        reference_path,
                        reference_cache,
                        dataset.output_folder,
                        logger,
                        setup_error,
                    )
                    record_completion(local_index, global_index, result)
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="tir_bridge",
                ) as executor:
                    futures = {
                        executor.submit(
                            safe_process_pipeline_scene,
                            index_offset + local_index,
                            total_scenes,
                            eco_path,
                            reference_path,
                            reference_cache,
                            dataset.output_folder,
                            None,
                            setup_error,
                        ): (local_index, index_offset + local_index)
                        for local_index, eco_path in enumerate(
                            dataset.eco_paths,
                            start=1,
                        )
                    }
                    for future in as_completed(futures):
                        local_index, global_index = futures[future]
                        record_completion(local_index, global_index, future.result())

            complete_folder_results = [
                result for result in folder_results if result is not None
            ]
            correction_csv.flush()
            logger.info(
                f"Dataset complete: {dataset_label} | "
                f"{len(complete_folder_results)} scene(s) | CSV rows={correction_csv.rows_written}"
            )
            index_offset += len(dataset.eco_paths)
            reference_cache = None
            clear_reference_grid_cache()

    complete_results = [result for result in results if result is not None]
    ok = sum(
        result.status in {"OK", "OK_HIGH_RES_RETRY"}
        for result in complete_results
    )
    retries = sum(
        int(result.row.get("high_resolution_retry_triggered") or 0) > 0
        for result in complete_results
    )
    passthrough = sum(
        result.status in {
            "PASSTHROUGH_WITH_FAILURE",
            "POLISH_FAILED_PASSTHROUGH",
            "RETRY_LOW_CONFIDENCE_PASSTHROUGH",
            "RETRY_FAILED_PASSTHROUGH",
            "RETRY_POLISH_FAILED_PASSTHROUGH",
            "UNEXPECTED_FAILURE_PASSTHROUGH",
        }
        for result in complete_results
    )
    output_failed = sum(result.status == "OUTPUT_FAILED" for result in complete_results)

    shutdown_process_lanes()
    batch_wall_seconds = time.perf_counter() - logger.start
    average_throughput_seconds = (
        batch_wall_seconds / len(complete_results)
        if complete_results
        else 0.0
    )
    computation_times = [
        float(result.row.get("total_processing_seconds") or 0.0)
        for result in complete_results
        if math.isfinite(float(result.row.get("total_processing_seconds") or 0.0))
    ]
    average_computation_seconds = (
        sum(computation_times) / len(computation_times)
        if computation_times
        else 0.0
    )

    logger.line("-")
    logger.info(f"Corrections CSV: {csv_path} | rows={len(complete_results)}")
    logger.info(
        f"Finished: datasets={len(datasets)}, scenes={len(complete_results)}, "
        f"OK={ok}, retries={retries}, "
        f"passthrough={passthrough}, output failures={output_failed}, "
        f"elapsed={logger.elapsed()}"
    )
    logger.info(
        "Error rasters: "
        f"written={sum(bool(result.row.get('err_output_file')) for result in complete_results)}, "
        f"missing={sum(result.row.get('err_status') == 'MISSING' for result in complete_results)}, "
        f"failed={sum(result.row.get('err_status') in {'GRID_MISMATCH', 'OUTPUT_FAILED'} for result in complete_results)}, "
        f"skipped={sum(result.row.get('err_status') == 'SKIPPED_PROCESSED_OUTPUT_FAILED' for result in complete_results)}"
    )
    logger.info(
        f"Average throughput time/file: {average_throughput_seconds:.2f} s | "
        f"Average computation time/file: {average_computation_seconds:.2f} s"
    )
    logger.info(
        f"CSV serialization/write: {correction_csv.serialization_write_seconds:.6f} s total | "
        f"{1000.0 * correction_csv.serialization_write_seconds / max(1, len(complete_results)):.3f} ms/file "
        "(included in throughput)"
    )
    logger.info(
        f"Reference-grid cache: hits={REFERENCE_GRID_CACHE_HITS}; "
        f"misses={REFERENCE_GRID_CACHE_MISSES}; "
        f"resident={format_bytes(REFERENCE_GRID_CACHE_BYTES)}"
    )
    log_performance_summary(complete_results, logger)
    logger.line("=")
    shutdown_process_lanes()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        shutdown_process_lanes()
        shutdown_gpu_runtime()
