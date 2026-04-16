#!/usr/bin/env python3
"""
Plot HALO catalogs as (x, y, z, mass): mass-weighted voxelization to a density grid,
optional Gaussian smoothing (scipy), log1p, then PyVista volume rendering (Blues).

Outputs are written as separate PNG files (one per catalog). CDF/CCDF composites
are no longer produced.
"""

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None  # type: ignore[misc, assignment]

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required.") from exc

try:
    import pyvista as pv  # type: ignore
except ImportError:  # pragma: no cover
    pv = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot HALO (x,y,z,mass) as separate density-volume PNGs.")
    p.add_argument("--original", required=True, help="Path to original halo CSV.")
    p.add_argument(
        "--decompressed",
        action="append",
        default=[],
        metavar="CSV",
        help="Decompressed halo CSV; repeat to emit one PNG per decompressed catalog.",
    )
    p.add_argument(
        "--decompressed-label",
        action="append",
        default=[],
        metavar="LABEL",
        help="Legend/subplot title for each --decompressed, same order. "
        "If omitted, uses '<compressor> <eb>' parsed from halo_decompressed_*.csv.",
    )
    p.add_argument(
        "--halo-metrics-dir",
        default=None,
        help=(
            "Optional dir containing <compressor>_halo.csv. If set and --decompressed-label "
            "is omitted, append W to labels: '<compressor> <eb> W=<wasserstein_distance>'."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output .png path (used as filename prefix) or output directory.",
    )
    p.add_argument(
        "--top-frac",
        type=float,
        default=1.0,
        help="Keep top mass fraction (0,1]. Use 1.0 to draw all halos.",
    )
    p.add_argument(
        "--grid-dims",
        type=int,
        default=160,
        help="Resolution of the mass-weighted histogram (per axis).",
    )
    p.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.4,
        help="Gaussian smoothing sigma in voxels; 0 disables. No effect if scipy is missing.",
    )
    p.add_argument("--elev", type=float, default=20.0, help="Camera elevation (deg).")
    p.add_argument("--azim", type=float, default=-55.0, help="Camera azimuth (deg).")
    p.add_argument("--title", default="HALO (x,y,z,mass)", help="Figure title.")
    p.add_argument(
        "--layout",
        choices=("overlay", "split"),
        default="overlay",
        help="overlay: one 3D scene (original + all decompresseds); split: one panel per catalog.",
    )
    p.add_argument(
        "--mass-dist",
        dest="mass_dist",
        action="store_true",
        default=True,
        help="Add log10(mass) empirical CDF and CCDF panels (default: on).",
    )
    p.add_argument(
        "--no-mass-dist",
        dest="mass_dist",
        action="store_false",
        help="Do not add CDF/CCDF panels (3D only).",
    )
    p.add_argument(
        "--ccdf-log-y",
        action="store_true",
        help="Use log scale on CCDF y-axis (helps heavy tails).",
    )
    p.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=(1800, 700),
        metavar=("W", "H"),
        help="PyVista off-screen window size for the 3D part.",
    )
    return p.parse_args()


def load_catalog(path: str) -> Dict[str, np.ndarray]:
    xs, ys, zs, ms = [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        required = {"x", "y", "z", "mass"}
        fields = set(r.fieldnames or [])
        if not required.issubset(fields):
            missing = sorted(required - fields)
            raise ValueError(f"{path} missing columns: {missing}")
        for row in r:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            ms.append(float(row["mass"]))
    if not xs:
        raise ValueError(f"{path} is empty.")
    return {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "z": np.asarray(zs, dtype=np.float64),
        "mass": np.asarray(ms, dtype=np.float64),
    }


def filter_top_mass(cat: Dict[str, np.ndarray], top_frac: float) -> Dict[str, np.ndarray]:
    top_frac = float(np.clip(top_frac, 1e-6, 1.0))
    if top_frac >= 0.999999:
        return cat
    q = np.quantile(cat["mass"], 1.0 - top_frac)
    m = cat["mass"] >= q
    return {k: v[m] for k, v in cat.items()}


def log10_mass(cat: Dict[str, np.ndarray]) -> np.ndarray:
    return np.log10(np.maximum(cat["mass"], 1e-300))


_DEC_BASENAME_RE = re.compile(
    r"^halo_decompressed_([^_]+)_([^_]+)_.+\.csv$", re.IGNORECASE
)
_DEC_META_RE = re.compile(
    r"^halo_decompressed_([^_]+)_([^_]+)_(.+)\.csv$", re.IGNORECASE
)
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def short_dec_label(path: str) -> str:
    name = Path(path).name
    m = _DEC_BASENAME_RE.match(name)
    if m:
        comp, eb = m.group(1), m.group(2)
        return f"{comp} {eb}"
    return Path(path).stem


def _norm_eb(text: str) -> str:
    try:
        return f"{float(str(text).strip()):.12g}"
    except Exception:
        return str(text).strip()


def safe_name(text: str) -> str:
    return SAFE_NAME_RE.sub("_", str(text).strip()).strip("_") or "na"


def _parse_dec_meta(path: str) -> Optional[Tuple[str, str, str]]:
    m = _DEC_META_RE.match(Path(path).name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _load_w_lookup(csv_path: Path) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            inp = str(row.get("input", "")).strip()
            eb = _norm_eb(str(row.get("error_bound", "")))
            w = row.get("wasserstein_distance", "")
            if not inp or not eb or w in ("", None):
                continue
            try:
                out[(inp, eb)] = float(w)
            except ValueError:
                continue
    return out


def label_with_w(
    dec_path: str,
    halo_metrics_dir: Optional[str],
    input_ext: str = ".hdf5",
) -> str:
    base = short_dec_label(dec_path)
    if not halo_metrics_dir:
        return base
    meta = _parse_dec_meta(dec_path)
    if not meta:
        return base
    comp, eb, file_tag = meta
    csv_path = Path(halo_metrics_dir).expanduser().resolve() / f"{comp}_halo.csv"
    w_map = _load_w_lookup(csv_path)
    inp = f"{file_tag}{input_ext}"
    w = w_map.get((inp, _norm_eb(eb)))
    if w is None:
        return f"{base} W=N/A"
    return f"{base} W={w:.6g}"


def wasserstein_for_dec(
    dec_path: str,
    halo_metrics_dir: Optional[str],
    input_ext: str = ".hdf5",
) -> Optional[float]:
    if not halo_metrics_dir:
        return None
    meta = _parse_dec_meta(dec_path)
    if not meta:
        return None
    comp, eb, file_tag = meta
    csv_path = Path(halo_metrics_dir).expanduser().resolve() / f"{comp}_halo.csv"
    w_map = _load_w_lookup(csv_path)
    inp = f"{file_tag}{input_ext}"
    return w_map.get((inp, _norm_eb(eb)))


def output_prefix_and_dir(out: Path) -> Tuple[Path, str]:
    out = out.expanduser().resolve()
    if out.suffix.lower() == ".png":
        return out.parent, safe_name(out.stem)
    return out, "halo_mass"


def empirical_cdf_ccdf(logm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(logm, dtype=np.float64))
    n = x.size
    if n == 0:
        return x, np.array([]), np.array([])
    cdf = np.arange(1, n + 1, dtype=np.float64) / n
    ccdf = (n - np.arange(1, n + 1, dtype=np.float64)) / n
    return x, cdf, ccdf


def plot_mass_cdf_ccdf(
    ax_cdf,
    ax_ccdf,
    series: List[Tuple[str, np.ndarray]],
    ccdf_log_y: bool,
) -> None:
    lab_to_color = {lab: f"C{i % 10}" for i, (lab, _) in enumerate(series)}
    plot_order = [s for s in series if s[0] != "original"] + [
        s for s in series if s[0] == "original"
    ]
    for lab, lm in plot_order:
        x, cdf, ccdf = empirical_cdf_ccdf(lm)
        if x.size == 0:
            continue
        color = lab_to_color[lab]
        is_orig = lab.strip().lower() == "original"
        lw = 2.8 if is_orig else 1.8
        zorder = 5 if is_orig else 2
        ax_cdf.plot(
            x, cdf, label=lab, color=color, lw=lw, zorder=zorder, alpha=0.95 if is_orig else 1.0
        )
        if ccdf_log_y:
            m = ccdf > 0
            if np.any(m):
                ax_ccdf.plot(
                    x[m],
                    ccdf[m],
                    label=lab,
                    color=color,
                    lw=lw,
                    zorder=zorder,
                    alpha=0.95 if is_orig else 1.0,
                )
        else:
            ax_ccdf.plot(
                x,
                ccdf,
                label=lab,
                color=color,
                lw=lw,
                zorder=zorder,
                alpha=0.95 if is_orig else 1.0,
            )
    ax_cdf.set_xlabel(r"$\log_{10}$(mass)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.grid(True, alpha=0.35)
    h1, l1 = ax_cdf.get_legend_handles_labels()
    by_l = dict(zip(l1, h1))
    leg_order = [lab for lab, _ in series if lab in by_l]
    ax_cdf.legend([by_l[lab] for lab in leg_order], leg_order, fontsize=8, loc="lower right")
    ax_ccdf.set_xlabel(r"$\log_{10}$(mass)")
    ax_ccdf.set_ylabel("CCDF")
    ax_ccdf.grid(True, alpha=0.35)
    h2, l2 = ax_ccdf.get_legend_handles_labels()
    by_l2 = dict(zip(l2, h2))
    ax_ccdf.legend(
        [by_l2[lab] for lab in leg_order if lab in by_l2],
        [lab for lab in leg_order if lab in by_l2],
        fontsize=8,
        loc="upper right",
    )
    if ccdf_log_y:
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_ylim(1e-6, 1.05)


def axis_limits(*cats: Dict[str, np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    x = np.concatenate([c["x"] for c in cats])
    y = np.concatenate([c["y"] for c in cats])
    z = np.concatenate([c["z"] for c in cats])
    return (float(x.min()), float(x.max())), (float(y.min()), float(y.max())), (float(z.min()), float(z.max()))


def _camera_bounds(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
) -> List[float]:
    return [xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]]


def halo_to_density_grid(
    cat: Dict[str, np.ndarray],
    dims: int = 160,
    sigma: float = 1.4,
    *,
    mins: Optional[np.ndarray] = None,
    maxs: Optional[np.ndarray] = None,
) -> "pv.ImageData":
    x, y, z, m = cat["x"], cat["y"], cat["z"], cat["mass"]
    if mins is None:
        mins = np.array([float(x.min()), float(y.min()), float(z.min())], dtype=np.float64)
    else:
        mins = np.asarray(mins, dtype=np.float64).reshape(3)
    if maxs is None:
        maxs = np.array([float(x.max()), float(y.max()), float(z.max())], dtype=np.float64)
    else:
        maxs = np.asarray(maxs, dtype=np.float64).reshape(3)

    H, _edges = np.histogramdd(
        np.column_stack([x, y, z]),
        bins=(dims, dims, dims),
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]]],
        weights=m,
    )

    if sigma > 0:
        if gaussian_filter is not None:
            H = gaussian_filter(H, sigma=sigma)
        # 没有 scipy 就先不平滑

    H = np.log1p(H)

    d = int(dims)
    span = maxs - mins
    spacing = tuple(float(span[i]) / (d - 1) if d > 1 else float(span[i] or 1.0) for i in range(3))

    grid = pv.ImageData(
        dimensions=np.array(H.shape, dtype=int),
        origin=(float(mins[0]), float(mins[1]), float(mins[2])),
        spacing=spacing,
    )
    grid.point_data["rho"] = H.ravel(order="F")
    return grid


def add_halo_structure(
    plotter: "pv.Plotter",
    grid: "pv.ImageData",
    title: str,
    *,
    opacity: Optional[Sequence[float]] = None,
    show_outline: bool = True,
    show_title: bool = True,
) -> None:
    vals = np.asarray(grid["rho"])
    vmin = float(np.quantile(vals, 0.70))
    vmax = float(np.quantile(vals, 0.999))
    op: Sequence[float] = (
        list(opacity)
        if opacity is not None
        else [0.0, 0.0, 0.01, 0.03, 0.08, 0.16, 0.28]
    )
    plotter.add_volume(
        grid,
        scalars="rho",
        cmap="Blues",
        clim=(vmin, vmax),
        opacity=op,
        blending="composite",
        shade=False,
        show_scalar_bar=False,
    )
    if show_outline:
        plotter.add_mesh(grid.outline(), color="#5c6aa5", opacity=0.25, line_width=3.0)
    if show_title and title and str(title).strip():
        plotter.add_text(title, position="upper_left", font_size=11, color="black", shadow=True)


def _apply_camera(plotter: "pv.Plotter", elev: float, azim: float, bounds: Sequence[float]) -> None:
    plotter.camera_position = "iso"
    plotter.reset_camera(bounds=list(bounds))
    plotter.camera.elevation = float(elev)
    plotter.camera.azimuth = float(azim)


def _screenshot_plotter(plotter: "pv.Plotter") -> np.ndarray:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        plotter.screenshot(path)
        return mpimg.imread(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def render_volume_split(
    catalogs: List[Dict[str, np.ndarray]],
    titles: List[str],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    elev: float,
    azim: float,
    window_size: Tuple[int, int],
    grid_dims: int,
    gaussian_sigma: float,
) -> np.ndarray:
    n = len(catalogs)
    w, h = window_size
    plotter = pv.Plotter(
        shape=(1, n), off_screen=True, window_size=(max(w, min(420 * n, 3600)), h)
    )
    bnds = _camera_bounds(xlim, ylim, zlim)
    mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=np.float64)
    maxs = np.array([xlim[1], ylim[1], zlim[1]], dtype=np.float64)
    for j, (cat, title) in enumerate(zip(catalogs, titles)):
        plotter.subplot(0, j)
        grid = halo_to_density_grid(
            cat, dims=grid_dims, sigma=gaussian_sigma, mins=mins, maxs=maxs
        )
        add_halo_structure(plotter, grid, title)
        plotter.set_background("white")
        _apply_camera(plotter, elev, azim, bnds)
    plotter.link_views()
    img = _screenshot_plotter(plotter)
    plotter.close()
    return img


def render_volume_overlay(
    orig: Dict[str, np.ndarray],
    decompresseds: List[Dict[str, np.ndarray]],
    dec_labels: Sequence[str],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    elev: float,
    azim: float,
    top_frac: float,
    window_size: Tuple[int, int],
    grid_dims: int,
    gaussian_sigma: float,
    *,
    show_title: bool,
) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=tuple(window_size))
    bnds = _camera_bounds(xlim, ylim, zlim)
    mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=np.float64)
    maxs = np.array([xlim[1], ylim[1], zlim[1]], dtype=np.float64)

    base_op = [0.0, 0.0, 0.01, 0.03, 0.08, 0.16, 0.28]
    cats: List[Dict[str, np.ndarray]] = [orig] + list(decompresseds)
    labels = ["original"] + [str(lab) for lab in dec_labels]
    first_grid: Optional["pv.ImageData"] = None
    for i, cat in enumerate(cats):
        grid = halo_to_density_grid(
            cat, dims=grid_dims, sigma=gaussian_sigma, mins=mins, maxs=maxs
        )
        if first_grid is None:
            first_grid = grid
        scale = 0.55 if i == 0 else 0.72
        op = [min(1.0, v * scale) for v in base_op]
        vals = np.asarray(grid["rho"])
        vmin = float(np.quantile(vals, 0.70))
        vmax = float(np.quantile(vals, 0.999))
        plotter.add_volume(
            grid,
            scalars="rho",
            cmap="Blues",
            clim=(vmin, vmax),
            opacity=op,
            blending="composite",
            shade=False,
            show_scalar_bar=False,
        )
    if first_grid is not None:
        plotter.add_mesh(first_grid.outline(), color="#5c6aa5", opacity=0.25)
    pct = int(round(float(np.clip(top_frac, 0.0, 1.0)) * 100))
    if show_title:
        plotter.add_text(
            f"Original + {len(decompresseds)} decompressed (top {pct}% mass)",
            position="upper_edge",
            font_size=11,
            color="black",
            shadow=True,
        )
    legend_lines = "\n".join(labels)
    plotter.add_text(legend_lines, position="upper_right", font_size=9, color="black", shadow=True)
    plotter.set_background("white")
    _apply_camera(plotter, elev, azim, bnds)
    img = _screenshot_plotter(plotter)
    plotter.close()
    return img


def render_volume_single(
    cat: Dict[str, np.ndarray],
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    elev: float,
    azim: float,
    window_size: Tuple[int, int],
    grid_dims: int,
    gaussian_sigma: float,
    *,
    show_title: bool,
) -> np.ndarray:
    # Match pressio_halo_compare.py behavior: square viewport with side=max(W,H,520).
    w, h = tuple(window_size)
    side = max(int(w), int(h), 520)
    plotter = pv.Plotter(off_screen=True, window_size=(side, side))
    bnds = _camera_bounds(xlim, ylim, zlim)
    mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=np.float64)
    maxs = np.array([xlim[1], ylim[1], zlim[1]], dtype=np.float64)
    grid = halo_to_density_grid(cat, dims=grid_dims, sigma=gaussian_sigma, mins=mins, maxs=maxs)
    add_halo_structure(plotter, grid, title if show_title else "")
    plotter.set_background("white")
    _apply_camera(plotter, elev, azim, bnds)
    img = _screenshot_plotter(plotter)
    plotter.close()
    return img


def save_composite_with_cdf(
    img_rgba: np.ndarray,
    dist_series: List[Tuple[str, np.ndarray]],
    title: str,
    ccdf_log_y: bool,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], hspace=0.32, wspace=0.28)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_rgba, aspect="auto")
    ax_img.axis("off")
    ax_cdf = fig.add_subplot(gs[1, 0])
    ax_ccdf = fig.add_subplot(gs[1, 1])
    plot_mass_cdf_ccdf(ax_cdf, ax_ccdf, dist_series, ccdf_log_y)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_image_only(img_rgba: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the raw rendered frame directly (no matplotlib re-layout/resampling),
    # so the cube appearance matches prior PyVista screenshots.
    mpimg.imsave(out_path, img_rgba)


def main() -> int:
    if pv is None:
        raise SystemExit("PyVista is required. Install in your environment (e.g. pip install pyvista).")

    args = parse_args()
    if int(args.grid_dims) < 2:
        raise SystemExit("--grid-dims must be at least 2.")
    orig = filter_top_mass(load_catalog(args.original), args.top_frac)
    dec_paths: List[str] = list(args.decompressed)
    if args.decompressed_label:
        if len(args.decompressed_label) != len(dec_paths):
            raise SystemExit(
                "Number of --decompressed-label must match number of --decompressed."
            )
        dec_labels: List[str] = list(args.decompressed_label)
    else:
        dec_labels = [label_with_w(p, args.halo_metrics_dir) for p in dec_paths]
    decompresseds = [
        filter_top_mass(load_catalog(p), args.top_frac) for p in dec_paths
    ]

    xlim, ylim, zlim = axis_limits(orig, *decompresseds) if dec_paths else axis_limits(orig)
    wsize = tuple(args.window_size)
    out_dir, out_prefix = output_prefix_and_dir(Path(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always render separate PNGs (no CDF/CCDF composite).
    img_o = render_volume_single(
        orig,
        "",
        xlim,
        ylim,
        zlim,
        args.elev,
        args.azim,
        wsize,
        args.grid_dims,
        args.gaussian_sigma,
        show_title=False,
    )
    out_original = out_dir / f"{out_prefix}_original.png"
    save_image_only(img_o, "", out_original)
    print(f"Saved: {out_original}")

    for dec_path, dec_cat in zip(dec_paths, decompresseds):
        meta = _parse_dec_meta(dec_path)
        if meta:
            comp, eb, _file_tag = meta
            w = wasserstein_for_dec(dec_path, args.halo_metrics_dir)
            w_token = safe_name("na" if w is None else f"{float(w):.6g}")
            out_name = f"{out_prefix}_{safe_name(comp)}_{safe_name(eb)}_W{w_token}.png"
        else:
            out_name = f"{out_prefix}_{safe_name(Path(dec_path).stem)}_Wna.png"
        out_path = out_dir / out_name
        img_d = render_volume_single(
            dec_cat,
            "",
            xlim,
            ylim,
            zlim,
            args.elev,
            args.azim,
            wsize,
            args.grid_dims,
            args.gaussian_sigma,
            show_title=False,
        )
        save_image_only(img_d, "", out_path)
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
