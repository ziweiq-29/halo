# halo_env 是 py3.9；若 PYTHONPATH/sys.path 含 libpressio-env，会加载 py3.11 的 numpy .so → ImportError
import os
import sys
os.environ.pop("PYTHONPATH", None)
_pp = "/anvil/projects/x-cis240669/libpressio-env"
sys.path = [p for p in sys.path if not (p.startswith(_pp) or _pp in p)]

import pandas as pd
import numpy as np
import re
import argparse
# 集群无显示器时 plt.show() 无效；用 Agg 后端 + savefig 写文件
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1. 实验参数（文件名与 pipeline2 / halo_dual_pressio2 一致：sz3_1e-3.csv）
# ================================

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
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
    用 (compressor_name, error_bound, file_name) 作为 key。
    """
    psnr_lookup = {}
    ssim_lookup = {}
    file_tag_to_inputs = {}
    if not os.path.isdir(standard_nyx_dir):
        return psnr_lookup, ssim_lookup, file_tag_to_inputs

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

            tag = _safe_file_stem(input_base)
            file_tag_to_inputs.setdefault(tag, set()).add(input_base)

    return psnr_lookup, ssim_lookup, file_tag_to_inputs

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
):
    rows = []
    wasserstein_map = {}
    psnr_map = {}
    ssim_map = {}
    file_name_base = _pick_input_basename(file_tag, file_tag_to_inputs)
    out_png = os.path.join(file_dir, f"halo_distribution_{compressor}_{file_tag}.png")
    out_kde_png = os.path.join(
        file_dir, f"halo_distribution_{compressor}_{file_tag}_displot_kde.png"
    )

    if (not force) and os.path.isfile(out_png) and os.path.isfile(out_kde_png):
        print(f"[get_distribution] skip {file_tag}: plots already exist", file=sys.stderr)
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

    df["log_mass"] = np.log10(df["mass"])
    sorted_error_bounds = sorted(set(df["error_bound"].tolist()))
    eb_label_by_eb = {eb: _eb_decimal_str(eb) for eb in sorted_error_bounds}
    df["eb"] = df["error_bound"].map(eb_label_by_eb)
    label_order = [eb_label_by_eb[eb] for eb in sorted_error_bounds]

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
    # 在每个 KDE 子图中显式标注该 error bound 对应的 Wasserstein 值。
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
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
        )
    g.set_axis_labels("log10(Halo Mass)", "Density")
    g.fig.subplots_adjust(top=0.86)
    g.fig.suptitle(f"Halo Mass KDE: {file_tag}")
    g.fig.savefig(out_kde_png, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[get_distribution] wrote {out_kde_png}", file=sys.stderr)


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
        help="File tag/folder name under halo/csv (e.g. NVB_..._z5).",
    )
    p.add_argument(
        "--compressor",
        default=None,
        help="Compressor subfolder under halo/csv (e.g. sz3, sperr). Requires --file-tag.",
    )
    p.add_argument(
        "--eb",
        action="append",
        default=None,
        metavar="REL",
        help="Relative error bound (repeatable), e.g. --eb 5e-5. Default: built-in list.",
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

    def _run_plot(
        file_dir: str,
        plot_tag: str,
        compressor: str,
        eb_list,
        w_lookup,
        file_tag_to_inputs,
        psnr_lookup,
        ssim_lookup,
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
        )

    # 显式：halo/csv/<compressor>/<file_tag>/ 或 同 compress 下扁平文件名带 tag
    if args.compressor is not None:
        if not args.file_tag:
            raise SystemExit("--compressor requires --file-tag.")
        compressor = args.compressor.strip()
        tag = selected_tag
        nested = os.path.join(CSV_DIR, compressor, tag)
        flat_parent = os.path.join(CSV_DIR, compressor)
        if os.path.isdir(nested):
            file_dir = nested
        elif os.path.isdir(flat_parent):
            file_dir = flat_parent
        else:
            raise SystemExit(
                f"expected csv under {nested} or directory {flat_parent}"
            )
        eb_use = eb_list_arg if eb_list_arg is not None else eb_list_default
        w_lookup, file_tag_to_inputs = _load_wasserstein_lookup(HALO_NYX_DIR, compressor)
        psnr_lookup, ssim_lookup = {}, {}
        if args.psnr or args.ssim:
            psnr_lookup, ssim_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
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
        )
        return

    top_dirs = sorted(
        d
        for d in os.listdir(CSV_DIR)
        if os.path.isdir(os.path.join(CSV_DIR, d)) and not d.startswith(".")
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
            psnr_lookup, ssim_lookup = {}, {}
            if args.psnr or args.ssim:
                psnr_lookup, ssim_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
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
                    eb_list_default,
                    w_lookup,
                    file_tag_to_inputs,
                    psnr_lookup,
                    ssim_lookup,
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
        psnr_lookup, ssim_lookup = {}, {}
        if args.psnr or args.ssim:
            psnr_lookup, ssim_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
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
                eb_list_default,
                w_lookup,
                file_tag_to_inputs,
                psnr_lookup,
                ssim_lookup,
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
        psnr_lookup, ssim_lookup = {}, {}
        if args.psnr or args.ssim:
            psnr_lookup, ssim_lookup, standard_tag_to_inputs = _load_standard_psnr_ssim_lookup(
                os.path.abspath(args.standard_nyx_dir), COMPRESSOR
            )
            for t, inputs in standard_tag_to_inputs.items():
                file_tag_to_inputs.setdefault(t, set()).update(inputs)
        for plot_tag in flat_tags:
            _run_plot(
                CSV_DIR,
                plot_tag,
                COMPRESSOR,
                eb_list_default,
                w_lookup,
                file_tag_to_inputs,
                psnr_lookup,
                ssim_lookup,
            )
        return

    raise SystemExit(f"no per-file folders or tagged csv files under {CSV_DIR}")


if __name__ == "__main__":
    main()