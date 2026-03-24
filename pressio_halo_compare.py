#!/usr/bin/env python3
"""
Compress a 3D HDF5 volume with libpressio using multiple compressors, and run
all user-specified error bounds directly (no nearest-CR pairing).

This script is designed for environments like Anvil where `pressio` is
available. It avoids direct HDF5 parsing by using libpressio's HDF5 input and
NumPy output path. Rendering uses PyVista when available; otherwise it still
saves decompressed volumes and a JSON summary for later visualization.

Typical usage:
    python pressio_halo_compare.py \
      --input /path/to/NVB_C009_l10n512_S12345T692_z5.hdf5 \
      --dataset /DataCT/data \
      --workdir ./halo_compare_z5 \
      --compressors sz3 sperr mgard zfp \
      --bounds 1e-3 5e-4 1e-4 5e-5 \
      --key rel \
      --render

    # Legacy aliases still work: --sz3-bounds / --sperr-bounds, --sz3-key / --sperr-key
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("This script requires numpy.") from exc

try:
    import pyvista as pv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pv = None


SIZE_COMPRESSED_RE = re.compile(r"size:compressed_size<[^>]+>\s*=\s*([0-9]+)")
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_compressor_tag(name: str) -> str:
    """Filesystem-safe fragment derived from a pressio compressor id."""
    return SAFE_NAME_RE.sub("_", name.strip()).strip("_") or "cmp"


@dataclass
class RunResult:
    compressor: str
    error_key: str
    error_bound: str
    compressed_bytes: int
    compression_ratio: float
    decompressed_path: str
    wasserstein_distance: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pressio on a 3D HDF5 dataset and compare multiple compressors."
    )
    parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="HDF5 dataset path to read, passed to pressio via -I.",
    )
    parser.add_argument(
        "--pressio-bin", default="pressio", help="Path to the pressio executable."
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Directory to store exported arrays, decompressed outputs, and figures.",
    )
    parser.add_argument(
        "--compressor-a",
        default="sz3",
        help="First pressio compressor id (e.g. sz3, zfp, sperr).",
    )
    parser.add_argument(
        "--compressor-b",
        default="sperr",
        help="Second pressio compressor id (e.g. sz3, zfp, sperr).",
    )
    parser.add_argument(
        "--compressors",
        nargs="+",
        default=None,
        help="Run all listed compressors (e.g. sz3 sperr mgard zfp). Overrides --compressor-a/--compressor-b.",
    )
    parser.add_argument(
        "--bounds",
        nargs="+",
        default=None,
        help="Error bounds to run directly for all compressors (e.g. 1e-2 5e-3).",
    )
    parser.add_argument(
        "--comp-bound",
        action="append",
        default=None,
        metavar="COMP=EB1,EB2",
        help=(
            "Per-compressor bounds, repeatable. Example: "
            "--comp-bound sz3=1e-2 --comp-bound zfp=1e-1,5e-2. "
            "When provided, overrides shared --bounds for listed compressors."
        ),
    )
    parser.add_argument(
        "--key",
        default=None,
        choices=("rel", "abs", "pw_rel"),
        help="Shared error-bound key for --compressors/--bounds mode. Defaults to rel.",
    )
    parser.add_argument(
        "--bounds-a",
        "--sz3-bounds",
        dest="bounds_a",
        nargs="+",
        default=["1e-3", "5e-4", "1e-4", "5e-5"],
        help="Candidate error bounds for compressor A (--compressor-a). Alias: --sz3-bounds.",
    )
    parser.add_argument(
        "--bounds-b",
        "--sperr-bounds",
        dest="bounds_b",
        nargs="+",
        default=["1e-3", "5e-4", "1e-4", "5e-5"],
        help="Candidate error bounds for compressor B (--compressor-b). Alias: --sperr-bounds.",
    )
    parser.add_argument(
        "--key-a",
        "--sz3-key",
        dest="key_a",
        default="rel",
        choices=("rel", "abs", "pw_rel"),
        help="Error-bound key for compressor A. Alias: --sz3-key.",
    )
    parser.add_argument(
        "--key-b",
        "--sperr-key",
        dest="key_b",
        default="rel",
        choices=("rel", "abs", "pw_rel"),
        help="Error-bound key for compressor B. Alias: --sperr-key.",
    )
    parser.add_argument(
        "--target-cr",
        type=float,
        default=None,
        help="Optional target compression ratio. When set, prefer pairs close to this CR.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional hint for the input dtype. Only used for documentation/output.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a 3-panel comparison with PyVista if available.",
    )
    parser.add_argument(
        "--render-log",
        action="store_true",
        help="Apply log1p before rendering to emphasize low-intensity structures.",
    )
    parser.add_argument(
        "--render-downsample",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Stride downsample each axis before rendering only (1=full resolution). "
            "Default 4 targets ~few-minute volume renders for 512^3-class data."
        ),
    )
    parser.add_argument(
        "--surface-quantile",
        type=float,
        default=0.7,
        help="Quantile on the original volume used to choose the isosurface level.",
    )
    parser.add_argument(
        "--render-style",
        choices=("isosurface", "volume"),
        default="volume",
        help="Rendering style: translucent volume or solid isosurface.",
    )
    parser.add_argument(
        "--volume-qmin",
        type=float,
        default=0.65,
        help="Lower quantile for volume rendering intensity range.",
    )
    parser.add_argument(
        "--volume-qmax",
        type=float,
        default=0.999,
        help="Upper quantile for volume rendering intensity range.",
    )
    parser.add_argument(
        "--volume-cmap",
        default="Blues",
        help="Colormap for volume rendering.",
    )
    parser.add_argument(
        "--clip-normal",
        nargs=3,
        type=float,
        default=(1.0, -1.0, 0.0),
        metavar=("NX", "NY", "NZ"),
        help="Clip-plane normal; only used when --render-clip is set.",
    )
    parser.add_argument(
        "--render-clip",
        action="store_true",
        help=(
            "Cut away half the domain with a plane through the center (--clip-normal). "
            "Default is off so the full cube is shown."
        ),
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=(1800, 650),
        metavar=("W", "H"),
        help="Window size for rendering.",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all decompressed candidate outputs instead of only the selected pair.",
    )
    parser.add_argument(
        "--halo-metrics-dir",
        default=None,
        help=(
            "Optional directory containing <compressor>_halo.csv files. "
            "When set, selected rel bounds are matched to wasserstein_distance."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: Sequence[str], cwd: Optional[Path] = None) -> Tuple[str, str]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def parse_compressed_size(stdout_text: str, stderr_text: str) -> int:
    combined = (stdout_text or "") + "\n" + (stderr_text or "")
    match = SIZE_COMPRESSED_RE.search(combined)
    if not match:
        raise RuntimeError(
            "Failed to parse size:compressed_size from pressio output.\n"
            f"Output tail:\n{combined[-1000:]}"
        )
    return int(match.group(1))


def export_original_numpy(
    pressio_bin: str, input_path: str, dataset: str, out_path: Path
) -> None:
    cmd = [
        pressio_bin,
        "-i",
        input_path,
        "-I",
        dataset,
        "-T",
        "hdf5",
        "-b",
        "compressor=noop",
        "-W",
        str(out_path),
        "-F",
        "numpy",
    ]
    run_command(cmd)


def compress_candidate(
    pressio_bin: str,
    input_path: str,
    dataset: str,
    compressor: str,
    error_key: str,
    error_bound: str,
    out_path: Path,
    original_bytes: int,
) -> RunResult:
    cmd = [
        pressio_bin,
        "-i",
        input_path,
        "-I",
        dataset,
        "-T",
        "hdf5",
        "-W",
        str(out_path),
        "-F",
        "numpy",
        "-b",
        f"compressor={compressor}",
        "-o",
        f"{error_key}={error_bound}",
        "-m",
        "size",
        "-M",
        "all",
    ]
    stdout_text, stderr_text = run_command(cmd)
    compressed_bytes = parse_compressed_size(stdout_text, stderr_text)
    ratio = float(original_bytes) / float(compressed_bytes)
    return RunResult(
        compressor=compressor,
        error_key=error_key,
        error_bound=error_bound,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        decompressed_path=str(out_path),
    )


def normalize_error_bound(value: str) -> str:
    text = str(value).strip()
    try:
        return f"{float(text):.12g}"
    except (TypeError, ValueError):
        return text


def load_wasserstein_map(csv_path: Path, input_basename: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if not csv_path.exists():
        return mapping
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("input", "")).strip() != input_basename:
                continue
            eb = normalize_error_bound(str(row.get("error_bound", "")))
            wd = row.get("wasserstein_distance", "")
            if not eb or wd in (None, ""):
                continue
            try:
                mapping[eb] = float(wd)
            except ValueError:
                continue
    return mapping


def fill_selected_wasserstein(
    selected: RunResult,
    metrics_dir: Optional[str],
    input_basename: str,
) -> Optional[float]:
    if not metrics_dir or selected.error_key != "rel":
        return None
    csv_name = f"{safe_compressor_tag(selected.compressor)}_halo.csv"
    csv_path = Path(metrics_dir).expanduser().resolve() / csv_name
    mapping = load_wasserstein_map(csv_path, input_basename)
    return mapping.get(normalize_error_bound(selected.error_bound))


def cleanup_unselected(
    runs: Iterable[RunResult], selected_paths: set[str], keep_all: bool
) -> None:
    if keep_all:
        return
    for run in runs:
        if run.decompressed_path not in selected_paths and os.path.exists(run.decompressed_path):
            os.remove(run.decompressed_path)


def prepare_volume(arr: np.ndarray, apply_log: bool) -> np.ndarray:
    vol = np.asarray(arr, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {vol.shape}")
    if apply_log:
        shifted = vol - np.min(vol)
        vol = np.log1p(shifted)
    return vol


def downsample_for_render(vol: np.ndarray, factor: int) -> np.ndarray:
    """Cheap stride subsample for VTK only; does not affect compression metrics."""
    if factor <= 1:
        return vol
    v = np.asarray(vol, dtype=np.float32)
    d0, d1, d2 = v.shape
    s0 = (d0 // factor) * factor
    s1 = (d1 // factor) * factor
    s2 = (d2 // factor) * factor
    if s0 == 0 or s1 == 0 or s2 == 0:
        return v
    cropped = v[:s0, :s1, :s2]
    return np.ascontiguousarray(cropped[::factor, ::factor, ::factor])


def build_uniform_grid(volume: np.ndarray) -> "pv.ImageData":
    grid = pv.ImageData()  # type: ignore[union-attr]
    grid.dimensions = np.array(volume.shape) + 1
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.cell_data["values"] = volume.flatten(order="F")
    return grid.cell_data_to_point_data()


def compute_clim(values: np.ndarray, qmin: float, qmax: float) -> Tuple[float, float]:
    qmin = float(np.clip(qmin, 0.0, 0.99999))
    qmax = float(np.clip(qmax, qmin + 1e-6, 1.0))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.quantile(finite, qmin))
    vmax = float(np.quantile(finite, qmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    return vmin, vmax


def render_one_row(
    original: np.ndarray,
    compare_volumes: Sequence[np.ndarray],
    out_path: Path,
    compare_runs: Sequence[RunResult],
    surface_quantile: float,
    render_style: str,
    volume_qmin: float,
    volume_qmax: float,
    volume_cmap: str,
    clip_normal: Tuple[float, float, float],
    use_clip: bool,
    window_size: Tuple[int, int],
) -> None:
    if pv is None:  # pragma: no cover - dependency guard
        raise RuntimeError("PyVista is not available; cannot render.")

    original_grid = build_uniform_grid(original)
    compare_grids = [build_uniform_grid(v) for v in compare_volumes]

    base_values = original_grid["values"]
    level = float(np.quantile(base_values[base_values > np.min(base_values)], surface_quantile))
    opacity_tf = [0.0, 0.0, 0.002, 0.008, 0.02, 0.05, 0.12, 0.22, 0.35]

    panel_count = 1 + len(compare_grids)
    plotter = pv.Plotter(shape=(1, panel_count), off_screen=True, window_size=window_size)  # type: ignore[union-attr]
    overlays = ["Original"]
    for run in compare_runs:
        w = f"{run.wasserstein_distance:.6g}" if run.wasserstein_distance is not None else "N/A"
        overlays.append(
            f"{run.compressor.upper()}\n"
            f"{run.error_key}={run.error_bound}\n"
            f"CR={run.compression_ratio:.2f}\n"
            f"W={w}"
        )
    panels = [original_grid, *compare_grids]
    for idx, grid in enumerate(panels):
        plotter.subplot(0, idx)
        center = tuple((np.array(grid.bounds)[::2] + np.array(grid.bounds)[1::2]) / 2.0)
        panel_values = grid["values"]
        panel_clim = compute_clim(panel_values, volume_qmin, volume_qmax)
        if render_style == "volume":
            vol_mesh = (
                grid.clip(normal=clip_normal, origin=center, invert=False)
                if use_clip
                else grid
            )
            plotter.add_volume(
                vol_mesh,
                scalars="values",
                cmap=volume_cmap,
                clim=panel_clim,
                opacity=opacity_tf,
                blending="composite",
                shade=False,
                diffuse=0.8,
                ambient=0.25,
                specular=0.05,
                show_scalar_bar=False,
            )
            plotter.show_bounds(
                color="#5c6aa5",
                font_size=8,
                location="outer",
                ticks="outside",
                grid=False,
            )
        else:
            contour = grid.contour(isosurfaces=[level], scalars="values")
            surf_mesh = (
                contour.clip(normal=clip_normal, origin=center) if use_clip else contour
            )
            plotter.add_mesh(
                surf_mesh,
                color="#cfc7b5",
                smooth_shading=True,
                specular=0.12,
                diffuse=0.9,
                ambient=0.2,
                show_scalar_bar=False,
            )
        plotter.add_text(
            overlays[idx],
            position="upper_left",
            font_size=12,
            color="black",
            shadow=True,
        )
        plotter.set_background("white")
        plotter.camera_position = "iso"
    plotter.link_views()
    plotter.screenshot(str(out_path))
    plotter.close()


def save_summary(
    out_path: Path,
    input_path: str,
    dataset: str,
    original_npy: str,
    original_shape: Sequence[int],
    original_dtype: str,
    compressors: Sequence[str],
    all_runs: dict[str, Sequence[RunResult]],
) -> None:
    summary = {
        "input_hdf5": input_path,
        "dataset": dataset,
        "original_numpy": original_npy,
        "original_shape": list(original_shape),
        "original_dtype": original_dtype,
        "compressors": list(compressors),
        "all_runs": {
            comp: [asdict(run) for run in runs] for comp, runs in all_runs.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_comp_bound_specs(specs: Optional[Sequence[str]]) -> dict[str, List[str]]:
    mapping: dict[str, List[str]] = {}
    if not specs:
        return mapping
    for raw in specs:
        text = str(raw).strip()
        if not text or "=" not in text:
            raise SystemExit(
                f"Invalid --comp-bound '{raw}'. Expected format COMP=EB1,EB2."
            )
        comp, bounds = text.split("=", 1)
        comp = comp.strip()
        if not comp:
            raise SystemExit(f"Invalid --comp-bound '{raw}': compressor is empty.")
        items = [b.strip() for b in bounds.split(",") if b.strip()]
        if not items:
            raise SystemExit(
                f"Invalid --comp-bound '{raw}': at least one bound is required."
            )
        mapping[comp] = items
    return mapping


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    ensure_dir(workdir)

    input_path = str(Path(args.input).resolve())
    original_npy = workdir / "original.npy"

    print("Exporting original volume via pressio noop...", flush=True)
    export_original_numpy(args.pressio_bin, input_path, args.dataset, original_npy)
    original = np.load(original_npy)
    if original.ndim != 3:
        raise SystemExit(
            f"Expected a 3D dataset, but {args.dataset} produced shape {original.shape}."
        )
    original_bytes = int(original.nbytes)

    print(f"Original shape={original.shape}, dtype={original.dtype}, bytes={original_bytes}", flush=True)

    if args.compressors:
        compressors = [c.strip() for c in args.compressors if c.strip()]
        if len(compressors) < 2:
            raise SystemExit("--compressors requires at least 2 compressor names.")
        shared_bounds = args.bounds if args.bounds else args.bounds_a
        shared_key = args.key if args.key else "rel"
        bounds_by_comp = {c: list(shared_bounds) for c in compressors}
        custom_bounds = parse_comp_bound_specs(args.comp_bound)
        for comp in custom_bounds:
            if comp not in compressors:
                raise SystemExit(
                    f"--comp-bound compressor '{comp}' is not in --compressors list."
                )
        for comp, vals in custom_bounds.items():
            bounds_by_comp[comp] = vals
        key_by_comp = {c: shared_key for c in compressors}
    else:
        compressors = [args.compressor_a, args.compressor_b]
        bounds_by_comp = {
            args.compressor_a: list(args.bounds_a),
            args.compressor_b: list(args.bounds_b),
        }
        key_by_comp = {
            args.compressor_a: args.key_a,
            args.compressor_b: args.key_b,
        }

    all_runs: dict[str, List[RunResult]] = {c: [] for c in compressors}
    for comp in compressors:
        tag = safe_compressor_tag(comp)
        key = key_by_comp[comp]
        for bound in bounds_by_comp[comp]:
            out_path = workdir / f"{tag}_{key}_{bound}.npy"
            print(f"Running {comp} at {key}={bound}...", flush=True)
            all_runs[comp].append(
                compress_candidate(
                    args.pressio_bin,
                    input_path,
                    args.dataset,
                    comp,
                    key,
                    bound,
                    out_path,
                    original_bytes,
                )
            )

    input_basename = Path(input_path).name
    for comp in compressors:
        for run in all_runs[comp]:
            run.wasserstein_distance = fill_selected_wasserstein(
                run, args.halo_metrics_dir, input_basename
            )
    print("Completed runs:", flush=True)
    for comp in compressors:
        for run in all_runs[comp]:
            print(
                f"  {comp} {run.error_key}={run.error_bound} "
                f"CR={run.compression_ratio:.3f} "
                f"W={run.wasserstein_distance if run.wasserstein_distance is not None else 'N/A'}",
                flush=True,
            )

    if not args.keep_all:
        print("Keeping all decompressed outputs by default in multi-compressor mode.", flush=True)

    summary_path = workdir / "selected_pair.json"
    save_summary(
        summary_path,
        input_path,
        args.dataset,
        str(original_npy),
        original.shape,
        str(original.dtype),
        compressors,
        all_runs,
    )
    print(f"Saved summary to {summary_path}", flush=True)

    if args.render:
        if len(compressors) < 2:
            raise SystemExit("--render requires at least 2 compressors.")
        if pv is None:
            print(
                "PyVista is not available in this environment. "
                "Skipping rendering but keeping selected outputs.",
                flush=True,
            )
        else:
            figure_path = workdir / "halo_compare.png"
            compare_runs: List[RunResult] = []
            compare_volumes: List[np.ndarray] = []
            for comp in compressors:
                runs = all_runs.get(comp, [])
                if not runs:
                    continue
                first_run = runs[0]
                compare_runs.append(first_run)
                compare_volumes.append(np.load(first_run.decompressed_path))
            if not compare_runs:
                raise SystemExit("No compressor outputs available for rendering.")
            rd = max(1, int(args.render_downsample))
            vo = prepare_volume(original, args.render_log)
            vo_ds = downsample_for_render(vo, rd)
            compare_ds = [
                downsample_for_render(prepare_volume(v, args.render_log), rd)
                for v in compare_volumes
            ]
            print(
                f"Rendering with --render-downsample={rd} "
                f"(shape {tuple(vo.shape)} -> {tuple(vo_ds.shape)})",
                flush=True,
            )
            panel_count = 1 + len(compare_ds)
            w, h = tuple(args.window_size)
            scaled_window = (max(w, 520 * panel_count), h)
            render_one_row(
                vo_ds,
                compare_ds,
                figure_path,
                compare_runs,
                args.surface_quantile,
                args.render_style,
                args.volume_qmin,
                args.volume_qmax,
                args.volume_cmap,
                tuple(args.clip_normal),
                args.render_clip,
                scaled_window,
            )
            print(f"Saved render to {figure_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
