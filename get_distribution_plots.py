# halo_env 是 py3.9；若 PYTHONPATH/sys.path 含 libpressio-env，会加载 py3.11 的 numpy .so → ImportError
import os
import sys

os.environ.pop("PYTHONPATH", None)
_pp = "/anvil/projects/x-cis240669/libpressio-env"
sys.path = [p for p in sys.path if not (p.startswith(_pp) or _pp in p)]


def _fix_sys_path_for_venv_numpy():
    """
    Anvil 等集群上用 Spack 的 python 建 venv 时，sys.path 里可能先有 Spack 的 py-numpy
    （旧版），导致即使用 pip 在 venv 里装了新版 numpy，import numpy 仍先到 1.19.x，
    matplotlib 版本检查失败。强制 venv site-packages 优先，并去掉 py-numpy 前缀路径。
    """
    base = getattr(sys, "base_prefix", sys.prefix)
    prefix = sys.prefix
    venv_sp = os.path.join(
        prefix, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}", "site-packages"
    )
    cleaned = []
    for p in sys.path:
        if not p:
            cleaned.append(p)
            continue
        norm = p.replace("\\", "/")
        if "/py-numpy/" in norm and "site-packages" in norm:
            continue
        cleaned.append(p)
    sys.path[:] = cleaned
    if prefix != base and os.path.isdir(venv_sp):
        while venv_sp in sys.path:
            sys.path.remove(venv_sp)
        sys.path.insert(0, venv_sp)


_fix_sys_path_for_venv_numpy()

import pandas as pd
import numpy as np
import re
import argparse
from collections import OrderedDict
from typing import List, Optional, Tuple

from sklearn.neighbors import NearestNeighbors
# 集群无显示器时 plt.show() 无效；用 Agg 后端 + savefig 写文件
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ================================
# 1. 实验参数（文件名与 pipeline2 / halo_dual_pressio2 一致：sz3_1e-3.csv）
# ================================

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
HEATMAP_DIR = os.path.join(CSV_DIR, "heat_map")
COMPRESSOR = "sz3"  # 与 run_pressio_pipeline2 --compressor 一致
HALO_NYX_DIR = "/anvil/projects/x-cis240669/compression_framework/outputs/HALO/NYX"
DEFAULT_STANDARD_NYX_DIR = "/anvil/projects/x-cis240669/compression_framework/outputs/STANDARD/NYX"


def _rel_to_eb_str(r):
    """与 halo_dual_pressio2 文件名一致：1e-3、5e-2，不是 0.001。"""
    if r is None:
        return "default"
    s = np.format_float_scientific(r, precision=12, unique=True, trim="0")
    if "e" not in s.lower():
        return str(r)
    s = s.lower()
    i = s.index("e")
    mant = s[:i].rstrip("0").rstrip(".")
    if not mant or mant == "-":
        mant = "0"
    exp_part = s[i + 1 :]
    try:
        exp_i = int(exp_part)
    except ValueError:
        exp_i = exp_part
    return f"{mant}e{exp_i}"


def _safe_file_stem(path):
    """把 file name 转为适合放到输出文件名里的后缀。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or "unknown_file"


error_bounds = [1e-3, 5e-3, 5e-2]

def _load_wasserstein_lookup(halo_nyx_dir, compressor_name: str):
    """
    从 compression_framework/outputs/HALO/NYX 读取 *_halo.csv，
    用 (compressor_name, error_bound, file_name) 作为 key。
    """
    lookup = {}
    file_tag_to_inputs = {}
    for dataset in os.listdir(halo_nyx_dir):
        dataset_dir = os.path.join(halo_nyx_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        csv_path = os.path.join(dataset_dir, f"{compressor_name}_halo.csv")
        if not os.path.isfile(csv_path):
            continue
        table = pd.read_csv(csv_path)
        if table.empty:
            continue
        # 兼容列名：compressor name / compressor_name
        comp_col = "compressor name" if "compressor name" in table.columns else "compressor_name"
        input_col = "input" if "input" in table.columns else "file_name"
        for _, row in table.iterrows():
            comp = str(row.get(comp_col, "")).strip()
            eb_val = pd.to_numeric(row.get("error_bound"), errors="coerce")
            inp = str(row.get(input_col, "")).strip()
            w = pd.to_numeric(row.get("wasserstein_distance"), errors="coerce")
            if not comp or pd.isna(eb_val) or not inp or pd.isna(w):
                continue
            input_base = os.path.basename(inp)
            key = (comp, _rel_to_eb_str(float(eb_val)), input_base)
            lookup[key] = float(w)
            tag = _safe_file_stem(input_base)
            file_tag_to_inputs.setdefault(tag, set()).add(input_base)
    return lookup, file_tag_to_inputs

def _load_standard_psnr_ssim_lookup(standard_nyx_dir, compressor_name: str):
    """
    从 compression_framework/outputs/STANDARD/NYX 读取 *_standard.csv，
    用 (compressor_name, error_bound, file_name) 作为 key；
    同时返回 compression_ratio（列名 compression_ratio）。
    """
    psnr_lookup = {}
    ssim_lookup = {}
    cr_lookup = {}
    file_tag_to_inputs = {}
    if not os.path.isdir(standard_nyx_dir):
        return psnr_lookup, ssim_lookup, cr_lookup, file_tag_to_inputs

    for dataset in os.listdir(standard_nyx_dir):
        dataset_dir = os.path.join(standard_nyx_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        csv_path = os.path.join(dataset_dir, f"{compressor_name}_standard.csv")
        if not os.path.isfile(csv_path):
            continue

        table = pd.read_csv(csv_path)
        if table.empty:
            continue

        comp_col = "compressor name" if "compressor name" in table.columns else "compressor_name"
        input_col = "input" if "input" in table.columns else "file_name"
        for _, row in table.iterrows():
            comp = str(row.get(comp_col, "")).strip()
            eb_val = pd.to_numeric(row.get("error_bound"), errors="coerce")
            inp = str(row.get(input_col, "")).strip()
            if not comp or pd.isna(eb_val) or not inp:
                continue

            input_base = os.path.basename(inp)
            key = (comp, _rel_to_eb_str(float(eb_val)), input_base)
            psnr_lookup[key] = pd.to_numeric(row.get("psnr"), errors="coerce")
            ssim_lookup[key] = pd.to_numeric(row.get("ssim"), errors="coerce")
            cr_lookup[key] = pd.to_numeric(row.get("compression_ratio"), errors="coerce")

            tag = _safe_file_stem(input_base)
            file_tag_to_inputs.setdefault(tag, set()).add(input_base)

    return psnr_lookup, ssim_lookup, cr_lookup, file_tag_to_inputs

def _w_str(w):
    if pd.isna(w):
        return "N/A"
    return f"{w:.6g}"

def _metric_str(v):
    if pd.isna(v):
        return "N/A"
    return f"{float(v):.6g}"

def _eb_decimal_str(eb):
    """Display error bound as decimal string (avoid scientific notation)."""
    try:
        v = float(eb)
    except Exception:
        return str(eb)
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _pick_input_basename(file_tag, file_tag_to_inputs):
    candidates = sorted(file_tag_to_inputs.get(file_tag, []))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # 多个候选时，优先包含 file_tag 的名字
    for c in candidates:
        if file_tag in c:
            return c
    return candidates[0]


def _try_read_csv(folder, name_candidates):
    for name in name_candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return pd.read_csv(path), path
    return None, None


def _hist_png_path(file_dir: str, compressor: str, file_tag: str, eb) -> str:
    eb_stem = _rel_to_eb_str(eb).replace(".", "p")
    return os.path.join(
        file_dir,
        f"halo_distribution_{compressor}_{file_tag}_eb_{eb_stem}_hist.png",
    )


def _write_hist_png(
    log_o: np.ndarray,
    log_d: np.ndarray,
    out_path: str,
    *,
    title: str,
    eb_label: str,
    w_val: float,
    nbins: int = 56,
) -> bool:
    """Static overlaid histograms on log10(mass): original vs decompressed."""
    log_o = np.asarray(log_o, dtype=np.float64).ravel()
    log_d = np.asarray(log_d, dtype=np.float64).ravel()
    if log_o.size == 0 or log_d.size == 0:
        return False
    lo = float(min(log_o.min(), log_d.min()))
    hi = float(max(log_o.max(), log_d.max()))
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, int(nbins) + 1)
    h0, ebins = np.histogram(log_o, bins=edges, density=True)
    h1, _ = np.histogram(log_d, bins=ebins, density=True)
    centers = (ebins[:-1] + ebins[1:]) / 2
    widths = np.diff(ebins)
    w_line = ""
    if w_val is not None and not np.isnan(w_val):
        w_line = f"  |  W={float(w_val):.6g}"

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.bar(
        centers - widths * 0.22,
        h0,
        width=widths * 0.45,
        align="center",
        alpha=0.82,
        color="#1d4ed8",
        edgecolor="white",
        linewidth=0.3,
        label="original",
    )
    ax.bar(
        centers + widths * 0.22,
        h1,
        width=widths * 0.45,
        align="center",
        alpha=0.78,
        color="#c2410c",
        edgecolor="white",
        linewidth=0.3,
        label="decompressed",
    )
    ax.set_xlabel("log10(Halo Mass)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.text(
        0.02,
        0.98,
        f"rel={eb_label}{w_line}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color="#334155",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ymax = max(float(h0.max()), float(h1.max())) * 1.15
    if ymax > 0:
        ax.set_ylim(0, ymax)
    ax.set_xlim(lo, hi)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _filename_metric_token(prefix: str, val) -> str:
    """Safe token for filenames, e.g. W0p836618, CR22p9795."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))):
        return f"{prefix}na"
    s = f"{float(val):.6g}".replace(".", "p").replace("-", "m")
    return f"{prefix}{s}"


def _heatmap_output_paths(
    compressor: str,
    file_tag: str,
    eb,
    w_val,
    cr_val,
    psnr_val,
) -> Tuple[str, str]:
    """PNG paths under csv/heat_map/: main heatmap + separate colorbar figure."""
    eb_stem = _rel_to_eb_str(eb).replace(".", "p")
    wt = _filename_metric_token("W", w_val)
    crt = _filename_metric_token("CR", cr_val)
    pt = _filename_metric_token("PSNR", psnr_val)
    base = f"halo_heat_{compressor}_{file_tag}_eb_{eb_stem}_{wt}_{crt}_{pt}"
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    main = os.path.join(HEATMAP_DIR, f"{base}.png")
    cbar = os.path.join(HEATMAP_DIR, f"{base}_cbar.png")
    return main, cbar


def _merge_halos_nearest_heatmap(orig: pd.DataFrame, dec: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    For each original halo, pair mass with the nearest decompressed halo in (x,y) or (x,y,z)
    grid space (Euclidean); dlog = log10(m_dec_nn) − log10(m_orig). Positions (x,y) for
    binning are taken from the original halo.
    """
    need = {"x", "y", "mass"}
    if not need.issubset(orig.columns) or not need.issubset(dec.columns):
        print(
            "[get_distribution] heatmap: CSV missing x,y or mass columns",
            file=sys.stderr,
        )
        return None
    o = orig.loc[orig["mass"] > 0].copy()
    d = dec.loc[dec["mass"] > 0].copy()
    if len(o) == 0 or len(d) == 0:
        return None
    cols_xyz = ["x", "y"]
    if "z" in o.columns and "z" in d.columns:
        cols_xyz.append("z")
    orig_xyz = o[cols_xyz].to_numpy(dtype=np.float64)
    dec_xyz = d[cols_xyz].to_numpy(dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(dec_xyz)
    _, idx = nn.kneighbors(orig_xyz)
    idx = idx.ravel()
    m_o = o["mass"].to_numpy(dtype=np.float64)
    m_d_arr = d["mass"].to_numpy(dtype=np.float64)[idx]
    x = o["x"].to_numpy(dtype=np.float64)
    y = o["y"].to_numpy(dtype=np.float64)
    dlog = np.log10(m_d_arr) - np.log10(m_o)
    return pd.DataFrame({"x": x, "y": y, "dlog": dlog})


def _binned_mean_dlog_grid(
    merged: pd.DataFrame, grid_bins: int
) -> Tuple[Optional[np.ndarray], float, float, float, float]:
    """Return (Z_plot for imshow, xmin, xmax, ymin, ymax) or (None, ...)."""
    x = merged["x"].to_numpy(dtype=np.float64)
    y = merged["y"].to_numpy(dtype=np.float64)
    w = merged["dlog"].to_numpy(dtype=np.float64)
    if x.size == 0:
        return None, 0.0, 1.0, 0.0, 1.0
    grid_bins = max(8, int(grid_bins))
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    xedges = np.linspace(xmin, xmax, grid_bins + 1)
    yedges = np.linspace(ymin, ymax, grid_bins + 1)
    sum_w, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)
    cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = sum_w / cnt
    z2 = np.where(cnt > 0, z2, np.nan)
    z_plot = z2.T
    if not np.any(np.isfinite(z_plot)):
        return None, xmin, xmax, ymin, ymax
    return z_plot, xmin, xmax, ymin, ymax


def _comp_eb_unified_heatmap_abs_max(
    grouped: OrderedDict[str, List[float]],
    selected_tag: Optional[str],
    grid_bins: int,
) -> Optional[float]:
    """Pool all binned |Δlog10 m| cells across compressors/error bounds for one shared color scale."""
    parts: List[np.ndarray] = []
    for compressor, eb_use in grouped.items():
        flat_parent = os.path.join(CSV_DIR, compressor)
        if not os.path.isdir(flat_parent):
            continue
        tag = selected_tag
        if tag is None:
            tag = _resolve_single_file_tag(flat_parent, compressor)
        if tag is None:
            continue
        nested = os.path.join(CSV_DIR, compressor, tag)
        file_dir = nested if os.path.isdir(nested) else flat_parent
        for eb in eb_use:
            eb_str = _rel_to_eb_str(eb)
            orig, _ = _try_read_csv(
                file_dir,
                [
                    f"halo_original_{compressor}_{eb_str}_{tag}.csv",
                    f"halo_original_{compressor}_{eb_str}.csv",
                ],
            )
            dec, _ = _try_read_csv(
                file_dir,
                [
                    f"halo_decompressed_{compressor}_{eb_str}_{tag}.csv",
                    f"halo_decompressed_{compressor}_{eb_str}.csv",
                ],
            )
            if orig is None or dec is None:
                continue
            m = _merge_halos_nearest_heatmap(orig, dec)
            if m is None or m.empty:
                continue
            zp, _, _, _, _ = _binned_mean_dlog_grid(m, grid_bins)
            if zp is None:
                continue
            zf = zp[np.isfinite(zp)]
            if zf.size:
                parts.append(np.abs(zf))
    if not parts:
        return None
    cat = np.concatenate(parts)
    v = float(np.percentile(cat, 99.5))
    if not np.isfinite(v) or v <= 0:
        v = float(np.max(cat))
    return v


def _save_heatmap_colorbar_figure(out_path: str, vmin: float, vmax: float, cmap: str = "coolwarm") -> None:
    """Standalone vertical colorbar PNG (matches heatmap scale)."""
    fig = plt.figure(figsize=(1.25, 4.8))
    ax = fig.add_axes([0.35, 0.08, 0.22, 0.84])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label("mean Δlog10(mass)  (nearest dec − orig)", fontsize=11)
    fig.patch.set_facecolor("white")
    _d = os.path.dirname(out_path)
    if _d:
        os.makedirs(_d, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_diff_heatmap_png(
    merged: pd.DataFrame,
    out_path: str,
    out_cbar_path: str,
    *,
    grid_bins: int = 48,
    color_abs_max: Optional[float] = None,
) -> bool:
    """Heatmap axes only (x/y labels); color bar saved to out_cbar_path."""
    z_plot, xmin, xmax, ymin, ymax = _binned_mean_dlog_grid(merged, grid_bins)
    if z_plot is None:
        return False
    if color_abs_max is not None and float(color_abs_max) > 0:
        vmax = float(color_abs_max)
    else:
        vmax = float(np.nanmax(np.abs(z_plot)))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1e-9

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    fig.patch.set_facecolor("#f8fafc")
    ax.imshow(
        z_plot,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("x (grid index)", fontsize=20)
    ax.set_ylabel("y (grid index)", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_heatmap_colorbar_figure(out_cbar_path, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    return True


def _plot_for_one_file(
    file_dir,
    file_tag,
    w_lookup,
    psnr_lookup,
    ssim_lookup,
    file_tag_to_inputs,
    *,
    compressor: str,
    error_bounds_list,
    include_psnr=False,
    include_ssim=False,
    force=False,
    kde_only: bool = False,
    kde_output_path: Optional[str] = None,
    histogram: bool = False,
    histogram_only: bool = False,
    heatmap: bool = False,
    heatmap_only: bool = False,
    heatmap_bins: int = 48,
    heatmap_color_abs_max: Optional[float] = None,
    cr_lookup: Optional[dict] = None,
):
    if cr_lookup is None:
        cr_lookup = {}
    rows = []
    wasserstein_map = {}
    psnr_map = {}
    ssim_map = {}
    per_eb_log: OrderedDict[float, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
    per_eb_merge: OrderedDict[float, Optional[pd.DataFrame]] = OrderedDict()
    file_name_base = _pick_input_basename(file_tag, file_tag_to_inputs)
    out_png = os.path.join(file_dir, f"halo_distribution_{compressor}_{file_tag}.png")
    if kde_output_path:
        out_kde_png = os.path.abspath(kde_output_path)
    elif len(error_bounds_list) == 1:
        eb0 = error_bounds_list[0]
        eb_stem = _rel_to_eb_str(eb0).replace(".", "p")
        out_kde_png = os.path.join(
            file_dir,
            f"halo_distribution_{compressor}_{file_tag}_eb_{eb_stem}_kde.png",
        )
    else:
        out_kde_png = os.path.join(
            file_dir, f"halo_distribution_{compressor}_{file_tag}_displot_kde.png"
        )

    hist_paths = (
        [_hist_png_path(file_dir, compressor, file_tag, eb) for eb in error_bounds_list]
        if histogram
        else []
    )
    heat_paths: List[str] = []
    if heatmap:
        for eb in error_bounds_list:
            eb_str = _rel_to_eb_str(eb)
            if file_name_base:
                w_key = (compressor, eb_str, file_name_base)
                wv = w_lookup.get(w_key, np.nan)
                crv = cr_lookup.get(w_key, np.nan)
                pv = psnr_lookup.get(w_key, np.nan)
            else:
                wv, crv, pv = np.nan, np.nan, np.nan
            hm, hc = _heatmap_output_paths(compressor, file_tag, eb, wv, crv, pv)
            heat_paths.extend([hm, hc])
    skip_kde_violin = histogram_only or heatmap_only

    if not force:
        need_violin = not kde_only and not skip_kde_violin
        need_kde = not skip_kde_violin
        need_hist = histogram
        need_heat = heatmap
        ok_v = (not need_violin) or os.path.isfile(out_png)
        ok_k = (not need_kde) or os.path.isfile(out_kde_png)
        ok_h = (not need_hist) or (
            bool(hist_paths) and all(os.path.isfile(p) for p in hist_paths)
        )
        ok_t = (not need_heat) or (
            bool(heat_paths) and all(os.path.isfile(p) for p in heat_paths)
        )
        if ok_v and ok_k and ok_h and ok_t:
            where = []
            if need_violin:
                where.append("violin")
            if need_kde:
                where.append("kde")
            if need_hist:
                where.append("hist")
            if need_heat:
                where.append("heatmap")
            print(
                f"[get_distribution] skip {file_tag}: "
                f"{'+'.join(where) or 'plot'} already exist",
                file=sys.stderr,
            )
            return

    for eb in error_bounds_list:
        eb_str = _rel_to_eb_str(eb)
        orig, orig_path = _try_read_csv(
            file_dir,
            [
                f"halo_original_{compressor}_{eb_str}_{file_tag}.csv",
                f"halo_original_{compressor}_{eb_str}.csv",
            ],
        )
        dec, dec_path = _try_read_csv(
            file_dir,
            [
                f"halo_decompressed_{compressor}_{eb_str}_{file_tag}.csv",
                f"halo_decompressed_{compressor}_{eb_str}.csv",
            ],
        )

        if orig is None or dec is None:
            print(f"[get_distribution] skip eb={eb_str} in {file_dir} (csv missing)", file=sys.stderr)
            continue

        if file_name_base:
            w_key = (compressor, eb_str, file_name_base)
            w_val = w_lookup.get(w_key, np.nan)
            psnr_val = psnr_lookup.get(w_key, np.nan)
            ssim_val = ssim_lookup.get(w_key, np.nan)
        else:
            w_val = np.nan
            psnr_val = np.nan
            ssim_val = np.nan
        wasserstein_map[eb] = w_val
        psnr_map[eb] = psnr_val
        ssim_map[eb] = ssim_val

        mo = orig.loc[orig["mass"] > 0, "mass"]
        md = dec.loc[dec["mass"] > 0, "mass"]
        if len(mo) > 0 and len(md) > 0:
            per_eb_log[eb] = (
                np.log10(mo.to_numpy(dtype=np.float64)),
                np.log10(md.to_numpy(dtype=np.float64)),
            )

        if heatmap or heatmap_only:
            merged = _merge_halos_nearest_heatmap(orig, dec)
            per_eb_merge[eb] = merged
            if merged is None:
                print(
                    f"[get_distribution] heatmap: no paired halos for eb={eb_str}",
                    file=sys.stderr,
                )

        for m in orig["mass"]:
            rows.append({"error_bound": eb, "type": "original", "mass": m})
        for m in dec["mass"]:
            rows.append({"error_bound": eb, "type": "decompressed", "mass": m})

        print(f"[get_distribution] using {orig_path} + {dec_path}", file=sys.stderr)

    if not rows:
        print(f"[get_distribution] no usable rows for {file_dir}, skip plotting", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df = df[df["mass"] > 0].copy()
    if df.empty:
        print(f"[get_distribution] no positive masses for {file_dir}, skip plotting", file=sys.stderr)
        return

    if not skip_kde_violin:
        df["log_mass"] = np.log10(df["mass"])
        sorted_error_bounds = sorted(set(df["error_bound"].tolist()))
        eb_label_by_eb = {eb: _eb_decimal_str(eb) for eb in sorted_error_bounds}
        df["eb"] = df["error_bound"].map(eb_label_by_eb)
        label_order = [eb_label_by_eb[eb] for eb in sorted_error_bounds]

        if not kde_only:
            plt.figure(figsize=(10, 6))
            sns.violinplot(
                data=df,
                x="eb",
                y="log_mass",
                hue="type",
                order=label_order,
                split=True,
                inner="quartile",
            )
            plt.xticks(rotation=0)
            plt.xlabel("Error Bound (Wasserstein Distance)")
            plt.ylabel("log10(Halo Mass)")
            plt.title(f"Halo Mass Distribution: {file_tag}")
            plt.legend(title="Data Type")
            plt.tight_layout()

            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[get_distribution] wrote {out_png}", file=sys.stderr)

        g = sns.displot(
            data=df,
            x="log_mass",
            hue="type",
            col="eb",
            col_order=label_order,
            col_wrap=3,
            kind="kde",
            fill=True,
            common_norm=False,
            height=3.6,
            aspect=1.3,
        )
        for i, eb in enumerate(sorted_error_bounds):
            if i >= len(g.axes.flat):
                break
            ax = g.axes.flat[i]
            w_val = wasserstein_map.get(eb, np.nan)
            metric_lines = [f"W={_w_str(w_val)}"]
            if include_psnr:
                metric_lines.append(f"PSNR={_metric_str(psnr_map.get(eb, np.nan))}")
            if include_ssim:
                metric_lines.append(f"SSIM={_metric_str(ssim_map.get(eb, np.nan))}")
            ax.text(
                0.03,
                0.97,
                "\n".join(metric_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )
        g.set_axis_labels("log10(Halo Mass)", "Density")
        g.fig.subplots_adjust(top=0.86)
        g.fig.suptitle(f"Halo Mass KDE: {file_tag}")
        _kde_dir = os.path.dirname(out_kde_png)
        if _kde_dir:
            os.makedirs(_kde_dir, exist_ok=True)
        g.fig.savefig(out_kde_png, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        print(f"[get_distribution] wrote {out_kde_png}", file=sys.stderr)

    if histogram:
        for eb, (log_o, log_d) in per_eb_log.items():
            hp = _hist_png_path(file_dir, compressor, file_tag, eb)
            wv = float(wasserstein_map.get(eb, np.nan))
            ok = _write_hist_png(
                log_o,
                log_d,
                hp,
                title=f"Halo mass (histogram): {compressor}  |  {file_tag}",
                eb_label=_eb_decimal_str(eb),
                w_val=wv,
            )
            if ok:
                print(f"[get_distribution] wrote {hp}", file=sys.stderr)

    if heatmap:
        vmax_shared: Optional[float] = heatmap_color_abs_max
        if vmax_shared is None:
            pool: List[np.ndarray] = []
            for _, merged in per_eb_merge.items():
                if merged is None or merged.empty:
                    continue
                zp, _, _, _, _ = _binned_mean_dlog_grid(merged, heatmap_bins)
                if zp is None:
                    continue
                zf = zp[np.isfinite(zp)]
                if zf.size:
                    pool.append(np.abs(zf))
            if pool:
                cat = np.concatenate(pool)
                vmax_shared = float(np.percentile(cat, 99.5))
                if not np.isfinite(vmax_shared) or vmax_shared <= 0:
                    vmax_shared = float(np.max(cat))
        for eb, merged in per_eb_merge.items():
            if merged is None or merged.empty:
                continue
            eb_str = _rel_to_eb_str(eb)
            if file_name_base:
                w_key = (compressor, eb_str, file_name_base)
                wv = w_lookup.get(w_key, np.nan)
                crv = cr_lookup.get(w_key, np.nan)
                pv = psnr_lookup.get(w_key, np.nan)
            else:
                wv, crv, pv = np.nan, np.nan, np.nan
            tp, tpc = _heatmap_output_paths(compressor, file_tag, eb, wv, crv, pv)
            ok = _write_diff_heatmap_png(
                merged,
                tp,
                tpc,
                grid_bins=heatmap_bins,
                color_abs_max=vmax_shared,
            )
            if ok:
                print(f"[get_distribution] wrote {tp}", file=sys.stderr)
                print(f"[get_distribution] wrote {tpc}", file=sys.stderr)


def _discover_flat_tags(csv_root, compressor_name: str):
    tags = set()
    pat = re.compile(rf"^halo_original_{re.escape(compressor_name)}_[^_]+_(.+)\.csv$")
    for name in os.listdir(csv_root):
        m = pat.match(name)
        if m:
            tags.add(m.group(1))
    return sorted(tags)


def _has_flat_halo_csv(comp_dir: str, compressor_name: str) -> bool:
    try:
        for f in os.listdir(comp_dir):
            if f.startswith(f"halo_original_{compressor_name}_") and f.endswith(".csv"):
                return True
    except OSError:
        pass
    return False


def _expand_comp_eb_specs(raw_list: List[str]) -> List[Tuple[str, float]]:
    """
    Parse --comp-eb entries: each item may be 'name=rel' or comma-separated
    'a=1e-6,b=1e-3'. Order is preserved for grouping.
    """
    pairs: List[Tuple[str, float]] = []
    for item in raw_list:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(
                    f"--comp-eb expects NAME=REL (commas allowed between pairs), got: {part!r}"
                )
            name, eb_s = part.split("=", 1)
            name = name.strip()
            if not name:
                raise SystemExit(f"empty compressor name in --comp-eb: {part!r}")
            try:
                eb = float(eb_s.strip())
            except ValueError as exc:
                raise SystemExit(
                    f"invalid error bound in --comp-eb {part!r}: {eb_s.strip()!r}"
                ) from exc
            pairs.append((name, eb))
    return pairs


def _group_pairs_by_compressor(pairs: List[Tuple[str, float]]) -> OrderedDict[str, List[float]]:
    grouped: OrderedDict[str, List[float]] = OrderedDict()
    for name, eb in pairs:
        grouped.setdefault(name, []).append(eb)
    return grouped


def _resolve_single_file_tag(comp_dir: str, compressor_name: str) -> Optional[str]:
    """
    If halo/csv/<compressor>/ contains exactly one dataset (nested folder or flat CSV tag),
    return that tag; else None.
    """
    try:
        nested = sorted(
            d
            for d in os.listdir(comp_dir)
            if os.path.isdir(os.path.join(comp_dir, d)) and not d.startswith(".")
        )
    except OSError:
        nested = []
    flat_tags = (
        _discover_flat_tags(comp_dir, compressor_name)
        if _has_flat_halo_csv(comp_dir, compressor_name)
        else []
    )
    tags = sorted(set(nested) | set(flat_tags))
    if len(tags) == 1:
        return tags[0]
    return None


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate halo distribution plots per file tag/folder."
    )
    p.add_argument(
        "--file-name",
        default=None,
        help="Original input file name/path (e.g. NVB_..._z5.hdf5).",
    )
    p.add_argument(
        "--file-tag",
        default=None,
        help="File tag/folder name under halo/csv (e.g. NVB_..._z5). "
        "Optional with --compressor when only one dataset exists under that folder.",
    )
    p.add_argument(
        "--compressor",
        default=None,
        help="Compressor subfolder under halo/csv (e.g. sz3, sperr). Not used together with --comp-eb.",
    )
    p.add_argument(
        "--comp-eb",
        action="append",
        default=None,
        metavar="NAME=REL",
        help="One or more compressor=error-bound pairs. Repeat flag and/or use commas: "
        "--comp-eb sperr=1e-6 --comp-eb 'sz3=1e-6,zfp=5e-5'. Same compressor listed "
        "multiple times merges error bounds into one KDE figure.",
    )
    p.add_argument(
        "--eb",
        action="append",
        default=None,
        metavar="REL",
        help="Relative error bound (repeatable), e.g. --eb 5e-5. Default: built-in list. "
        "Used with --compressor only (not with --comp-eb).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate plots even if output PNGs already exist.",
    )
    p.add_argument(
        "--standard-nyx-dir",
        default=DEFAULT_STANDARD_NYX_DIR,
        help="Path to STANDARD/NYX folder for loading PSNR/SSIM lookup.",
    )
    p.add_argument(
        "--psnr",
        action="store_true",
        help="Show PSNR in KDE subplot annotations.",
    )
    p.add_argument(
        "--ssim",
        action="store_true",
        help="Show SSIM in KDE subplot annotations.",
    )
    p.add_argument(
        "--kde-only",
        action="store_true",
        help="Only write the KDE figure (skip violin plot).",
    )
    p.add_argument(
        "--histogram",
        action="store_true",
        help="Also write static overlaid histogram PNG(s) per error bound "
        "(original vs decompressed, log10 mass).",
    )
    p.add_argument(
        "--histogram-only",
        action="store_true",
        help="Only write histogram PNG(s); skip violin and KDE.",
    )
    p.add_argument(
        "--heatmap",
        action="store_true",
        help="Also write (x,y) mean Δlog10(mass) heatmaps "
        "(each orig halo vs spatially nearest dec halo).",
    )
    p.add_argument(
        "--heatmap-only",
        "--heat-map-only",
        "--heap-map-only",
        "--heap_map_only",
        dest="heatmap_only",
        action="store_true",
        help="Only write difference heatmap PNG(s); skip violin, KDE, and histogram.",
    )
    p.add_argument(
        "--heatmap-bins",
        type=int,
        default=48,
        metavar="N",
        help="Number of bins along x and y for the heatmap (default 48).",
    )
    p.add_argument(
        "--output-kde",
        default=None,
        metavar="PATH",
        help="Explicit path for the KDE PNG (default: under csv folder with compressor/tag/eb in name).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if not os.path.isdir(CSV_DIR):
        raise SystemExit(f"csv dir not found: {CSV_DIR}")

    selected_tag = None
    if args.file_tag:
        selected_tag = _safe_file_stem(args.file_tag)
    elif args.file_name:
        selected_tag = _safe_file_stem(args.file_name)

    eb_list_default = list(error_bounds)
    eb_list_arg = [float(x) for x in args.eb] if args.eb else None
    eb_list_batch = eb_list_arg if eb_list_arg is not None else eb_list_default

    def _run_plot(
        file_dir: str,
        plot_tag: str,
        compressor: str,
        eb_list,
        w_lookup,
        file_tag_to_inputs,
        psnr_lookup,
        ssim_lookup,
        cr_lookup,
        *,
        kde_output_path: Optional[str] = None,
        heatmap_color_abs_max: Optional[float] = None,
    ) -> None:
        _plot_for_one_file(
            file_dir,
            plot_tag,
            w_lookup,
            psnr_lookup,
            ssim_lookup,
            file_tag_to_inputs,
            compressor=compressor,
            error_bounds_list=eb_list,
            include_psnr=args.psnr,
            include_ssim=args.ssim,
            force=args.force,
            kde_only=args.kde_only,
            kde_output_path=kde_output_path,
            histogram=args.histogram or args.histogram_only,
            histogram_only=args.histogram_only,
            heatmap=args.heatmap or args.heatmap_only,
            heatmap_only=args.heatmap_only,
            heatmap_bins=max(8, int(args.heatmap_bins)),
            heatmap_color_abs_max=heatmap_color_abs_max,
            cr_lookup=cr_lookup,
        )

    # 多个 compressor + error bound 写在一起：--comp-eb sperr=1e-6 sz3=1e-6 或逗号分隔
    if args.comp_eb:
        if args.compressor is not None:
            raise SystemExit("Do not combine --comp-eb with --compressor; use only --comp-eb.")
        pairs = _expand_comp_eb_specs(args.comp_eb)
        if not pairs:
            raise SystemExit("--comp-eb: no valid NAME=REL pairs.")
        grouped = _group_pairs_by_compressor(pairs)
        multi_comp = len(grouped) > 1
        if args.output_kde and multi_comp:
            print(
                "[get_distribution] ignoring --output-kde when --comp-eb lists multiple compressors",
                file=sys.stderr,
            )
        hb = max(8, int(args.heatmap_bins))
        heat_unified: Optional[float] = None
        if args.heatmap or args.heatmap_only:
            heat_unified = _comp_eb_unified_heatmap_abs_max(grouped, selected_tag, hb)
            if heat_unified is not None:
                print(
                    f"[get_distribution] heatmap shared color scale |Δ| ≤ {heat_unified:.6g} "
                    f"(99.5%ile over all binned cells in this run)",
                    file=sys.stderr,
                )
        for compressor, eb_use in grouped.items():
            flat_parent = os.path.join(CSV_DIR, compressor)
            if not os.path.isdir(flat_parent):
                print(
                    f"[get_distribution] skip compressor {compressor!r}: missing {flat_parent}",
                    file=sys.stderr,
                )
                continue
            tag = selected_tag
            if tag is None:
                tag = _resolve_single_file_tag(flat_parent, compressor)
                if tag is None:
                    raise SystemExit(
                        f"--file-tag is required for compressor {compressor!r}: "
                        f"multiple datasets under {flat_parent}."
                    )
            nested = os.path.join(CSV_DIR, compressor, tag)
            if os.path.isdir(nested):
                file_dir = nested
            else:
                file_dir = flat_parent
            if not eb_use:
                continue
            w_lookup, file_tag_to_inputs = _load_wasserstein_lookup(HALO_NYX_DIR, compressor)
            psnr_lookup, ssim_lookup, cr_lookup = {}, {}, {}
            if args.psnr or args.ssim or args.heatmap or args.heatmap_only:
                psnr_lookup, ssim_lookup, cr_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
                    os.path.abspath(args.standard_nyx_dir), compressor
                )
                for t, inputs in standard_tag_to_inputs.items():
                    file_tag_to_inputs.setdefault(t, set()).update(inputs)
            kde_out = None if multi_comp else args.output_kde
            _run_plot(
                file_dir,
                tag,
                compressor,
                eb_use,
                w_lookup,
                file_tag_to_inputs,
                psnr_lookup,
                ssim_lookup,
                cr_lookup,
                kde_output_path=kde_out,
                heatmap_color_abs_max=heat_unified,
            )
        return

    # 显式：halo/csv/<compressor>/<file_tag>/ 或 同 compress 下扁平文件名带 tag
    if args.compressor is not None:
        compressor = args.compressor.strip()
        flat_parent = os.path.join(CSV_DIR, compressor)
        if not os.path.isdir(flat_parent):
            raise SystemExit(f"compressor csv directory not found: {flat_parent}")
        tag = selected_tag
        if tag is None:
            tag = _resolve_single_file_tag(flat_parent, compressor)
            if tag is None:
                raise SystemExit(
                    f"--file-tag is required: multiple datasets under {flat_parent}. "
                    f"List nested dirs / matching halo_original_{compressor}_*.csv tags."
                )
        nested = os.path.join(CSV_DIR, compressor, tag)
        if os.path.isdir(nested):
            file_dir = nested
        else:
            file_dir = flat_parent
        eb_use = eb_list_batch
        if not eb_use:
            raise SystemExit("no error bounds: pass --eb REL (repeatable) or rely on defaults.")
        w_lookup, file_tag_to_inputs = _load_wasserstein_lookup(HALO_NYX_DIR, compressor)
        psnr_lookup, ssim_lookup, cr_lookup = {}, {}, {}
        if args.psnr or args.ssim or args.heatmap or args.heatmap_only:
            psnr_lookup, ssim_lookup, cr_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
                os.path.abspath(args.standard_nyx_dir), compressor
            )
            for t, inputs in standard_tag_to_inputs.items():
                file_tag_to_inputs.setdefault(t, set()).update(inputs)
        _run_plot(
            file_dir,
            tag,
            compressor,
            eb_use,
            w_lookup,
            file_tag_to_inputs,
            psnr_lookup,
            ssim_lookup,
            cr_lookup,
            kde_output_path=args.output_kde,
        )
        return

    top_dirs = sorted(
        d
        for d in os.listdir(CSV_DIR)
        if os.path.isdir(os.path.join(CSV_DIR, d))
        and not d.startswith(".")
        and d != "heat_map"
    )
    any_plotted = False
    for compressor in top_dirs:
        comp_dir = os.path.join(CSV_DIR, compressor)
        nested = sorted(
            d
            for d in os.listdir(comp_dir)
            if os.path.isdir(os.path.join(comp_dir, d)) and not d.startswith(".")
        )
        flat_tags = (
            _discover_flat_tags(comp_dir, compressor)
            if _has_flat_halo_csv(comp_dir, compressor)
            else []
        )
        tags = sorted(set(nested) | set(flat_tags))
        if not tags and _has_flat_halo_csv(comp_dir, COMPRESSOR):
            w_lookup, file_tag_to_inputs = _load_wasserstein_lookup(HALO_NYX_DIR, COMPRESSOR)
            psnr_lookup, ssim_lookup, cr_lookup = {}, {}, {}
            if args.psnr or args.ssim or args.heatmap or args.heatmap_only:
                psnr_lookup, ssim_lookup, cr_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
                    os.path.abspath(args.standard_nyx_dir), COMPRESSOR
                )
                for t, inputs in standard_tag_to_inputs.items():
                    file_tag_to_inputs.setdefault(t, set()).update(inputs)
            legacy_tags = _discover_flat_tags(comp_dir, COMPRESSOR)
            if not legacy_tags:
                legacy_tags = [compressor]
            if selected_tag is not None:
                legacy_tags = [t for t in legacy_tags if t == selected_tag]
            for plot_tag in legacy_tags:
                _run_plot(
                    comp_dir,
                    plot_tag,
                    COMPRESSOR,
                    eb_list_batch,
                    w_lookup,
                    file_tag_to_inputs,
                    psnr_lookup,
                    ssim_lookup,
                    cr_lookup,
                )
                any_plotted = True
            continue
        if not tags:
            continue
        if selected_tag is not None:
            tags = [t for t in tags if t == selected_tag]
        if not tags:
            continue
        w_lookup, file_tag_to_inputs = _load_wasserstein_lookup(HALO_NYX_DIR, compressor)
        psnr_lookup, ssim_lookup, cr_lookup = {}, {}, {}
        if args.psnr or args.ssim or args.heatmap or args.heatmap_only:
            psnr_lookup, ssim_lookup, cr_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
                os.path.abspath(args.standard_nyx_dir), compressor
            )
            for t, inputs in standard_tag_to_inputs.items():
                file_tag_to_inputs.setdefault(t, set()).update(inputs)
        for plot_tag in tags:
            nested_path = os.path.join(comp_dir, plot_tag)
            if os.path.isdir(nested_path):
                file_dir = nested_path
            else:
                file_dir = comp_dir
            _run_plot(
                file_dir,
                plot_tag,
                compressor,
                eb_list_batch,
                w_lookup,
                file_tag_to_inputs,
                psnr_lookup,
                ssim_lookup,
                cr_lookup,
            )
            any_plotted = True

    if any_plotted:
        return

    flat_tags = _discover_flat_tags(CSV_DIR, COMPRESSOR)
    if flat_tags:
        if selected_tag is not None:
            if selected_tag not in flat_tags:
                raise SystemExit(
                    f"requested file tag not found in flat csv files: {selected_tag}"
                )
            flat_tags = [selected_tag]
        w_lookup, file_tag_to_inputs = _load_wasserstein_lookup(HALO_NYX_DIR, COMPRESSOR)
        psnr_lookup, ssim_lookup, cr_lookup = {}, {}, {}
        if args.psnr or args.ssim or args.heatmap or args.heatmap_only:
            psnr_lookup, ssim_lookup, cr_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
                os.path.abspath(args.standard_nyx_dir), COMPRESSOR
            )
            for t, inputs in standard_tag_to_inputs.items():
                file_tag_to_inputs.setdefault(t, set()).update(inputs)
        for plot_tag in flat_tags:
            _run_plot(
                CSV_DIR,
                plot_tag,
                COMPRESSOR,
                eb_list_batch,
                w_lookup,
                file_tag_to_inputs,
                psnr_lookup,
                ssim_lookup,
                cr_lookup,
            )
        return

    raise SystemExit(f"no per-file folders or tagged csv files under {CSV_DIR}")


if __name__ == "__main__":
    main()