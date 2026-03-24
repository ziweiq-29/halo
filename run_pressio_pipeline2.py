#!/usr/bin/env python3
"""
Pipeline2：CSV 统一写 halo/csv；默认 halo_dual_pressio2 + 多 rel / HDF5；
不改动 run_pressio_pipeline.py，run_halo / run_halo_pressio 仍用原版。
"""
import argparse
import subprocess
import sys
import os
import re
import json
import shlex
import glob

# halo_env 等若 PYTHONPATH 含 libpressio-env，会加载 py3.11 的 numpy -> ImportError
os.environ.pop("PYTHONPATH", None)
_pp = "/anvil/projects/x-cis240669/libpressio-env"
sys.path = [p for p in sys.path if not (p.startswith(_pp) or _pp in p)]

import numpy as np

# ------------ Command-line args ------------
parser = argparse.ArgumentParser(
    description="Pipeline2: external -> halo_dual_pressio2, CSV under halo/csv; top-level runs pressio"
)
parser.add_argument("--external_mode", action="store_true")
parser.add_argument("--input", help="Input path")
parser.add_argument("--decompressed", help="Decompressed path (LibPressio injects)")
parser.add_argument("--original_input", help="Original input path (HDF5); pass in external:command")
parser.add_argument("--dim", type=int, action="append", help="Dimensions (repeat --dim)")
parser.add_argument("--rel", type=float, default=None,
                    help="Single error bound only. For multiple bounds use --rels or --error-bounds.")
parser.add_argument("--rels", "--error-bounds", type=float, nargs="+", default=None, dest="rels",
                    help="Multiple error bounds → pressio runs once each → halo/csv gets "
                         "halo_original_<eb>.csv + halo_decompressed_<eb>.csv per bound. "
                         "Example: --rels 1e-3 5e-3 5e-2")
parser.add_argument("--parallel", action="store_true", help="Run each rel in parallel (thread pool)")
parser.add_argument("--artifact-root", default=None, help="CSV dir (default: script_dir/csv)")
parser.add_argument("--compressor", default="sz3")
parser.add_argument("--halo_exe", default="/anvil/projects/x-cis240669/halo/reeber/build/examples/amr-connected-components/amr_connected_components_float")
parser.add_argument("--external_script", default="/anvil/projects/x-cis240669/halo/halo_dual_pressio2.py")
parser.add_argument("--pressio", default="pressio")
parser.add_argument("-I", "--hdf5-dataset", dest="hdf5_dataset", default=None,
                    help="HDF5 dataset for pressio -I (default /native_fields/baryon_density if .h5/.hdf5)")
parser.add_argument(
    "--verbose-pressio",
    action="store_true",
    help="After each pressio run, print its stdout/stderr to stderr (for debugging).",
)
args, unknown = parser.parse_known_args()

input_file = args.input
dims = args.dim or []
if args.rels:
    rel_list = list(args.rels)
elif args.rel is not None:
    rel_list = [args.rel]
else:
    rel_list = [1e-1]
rel = rel_list[0]
compressor = args.compressor
halo_exe = args.halo_exe
script_dir = os.path.dirname(os.path.abspath(__file__))
external_script = args.external_script if os.path.isabs(args.external_script) else os.path.join(script_dir, args.external_script)
pressio = args.pressio
original_input = args.original_input or args.input
csv_dir = args.artifact_root or os.path.join(script_dir, "csv")


def _safe_file_stem(path):
    """与 halo_dual_pressio2._safe_file_stem 一致（artifact_dir / 文件名 tag）。"""
    if not path:
        return "unknown_file"
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or "unknown_file"


def _eb_for_csv(compressor, r):
    """与 halo_dual_pressio2 中 eb 一致：有 compressor 时为 sz3_1e-3 等"""
    eb_str = _rel_to_eb_str(r)
    if compressor:
        return f"{compressor}_{eb_str}"
    return eb_str


def _nested_csv_paths(csv_dir, compressor, r, original_path):
    eb_str = _rel_to_eb_str(r)
    comp = compressor.strip()
    file_tag = _safe_file_stem(original_path or "")
    base = os.path.join(csv_dir, comp, file_tag)
    orig = os.path.join(base, f"halo_original_{comp}_{eb_str}_{file_tag}.csv")
    dec = os.path.join(base, f"halo_decompressed_{comp}_{eb_str}_{file_tag}.csv")
    return orig, dec


def _csv_artifacts_exist(csv_dir, compressor, r, original_path=None):
    """halo_dual 嵌套目录 csv/<compressor>/<file_tag>/ 或旧版扁平 csv/*.csv"""
    if original_path:
        o, d = _nested_csv_paths(csv_dir, compressor, r, original_path)
        if os.path.isfile(o) and os.path.isfile(d):
            return True
    eb = _eb_for_csv(compressor, r)
    orig = os.path.join(csv_dir, f"halo_original_{eb}.csv")
    dec = os.path.join(csv_dir, f"halo_decompressed_{eb}.csv")
    return os.path.isfile(orig) and os.path.isfile(dec)


def _rel_artifacts_ok(csv_dir, compressor, r, original_path):
    return _csv_artifacts_exist(csv_dir, compressor, r, original_path=original_path)


def _dump_pressio_io(proc, limit=24000):
    if proc.stdout and proc.stdout.strip():
        print("[pipeline2] pressio stdout:\n" + proc.stdout[-limit:], file=sys.stderr)
    if proc.stderr and proc.stderr.strip():
        print("[pipeline2] pressio stderr:\n" + proc.stderr[-limit:], file=sys.stderr)


def _fail_missing_csv_after_pressio(csv_dir, compressor, r, original_path, proc):
    o, d = _nested_csv_paths(csv_dir, compressor, r, original_path)
    eb = _rel_to_eb_str(r)
    print(
        f"[pipeline2] expected CSV not found after pressio (exit {proc.returncode}).\n"
        f"  nested: {o}\n"
        f"          {d}\n"
        f"  or flat: halo_original_{_eb_for_csv(compressor, r)}.csv (+ decompressed)",
        file=sys.stderr,
    )
    _dump_pressio_io(proc)
    print("external:api=json:1", flush=True)
    print(json.dumps({"dists": [], "mass_orig": [], "mass_dec": []}), flush=True)
    sys.exit(1)


def _rel_to_eb_str(r):
    """与 halo_dual_pressio2 文件名一致：科学计数法 1e-3、5e-2 等"""
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


def _is_top_level_pressio_run():
    return bool(args.external_mode and input_file and len(dims) >= 3)


# probe：无 --decompressed 且不是「顶层带 input+dim」时，空 JSON 退出
if not args.decompressed and not _is_top_level_pressio_run():
    print("external:api=json:1")
    print(json.dumps({"dists": []}))
    sys.exit(0)

# ------------ decompressed：只调 halo_dual_pressio2，CSV 进 csv_dir；默认不写 npy ------------
if args.decompressed:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "")
    os.makedirs(csv_dir, exist_ok=True)
    if not env.get("HALO_ARTIFACT_DIR", "").strip():
        env["HALO_ARTIFACT_DIR"] = csv_dir
    env["HALO_CSV_ONLY"] = "1"
    env.pop("HALO_SAVE_NPY", None)

    eb_str = _rel_to_eb_str(rel)
    env["HALO_EB_STR"] = eb_str
    # 子进程往往没带 --compressor，默认会变成 sz3；优先保留 pressio external 里 env 设的 HALO_COMPRESSOR
    env["HALO_COMPRESSOR"] = os.environ.get("HALO_COMPRESSOR", "").strip() or compressor
    halo_cmd = [
        sys.executable, external_script,
        "--input", args.input,
        "--decompressed", args.decompressed,
        "--external_exe", halo_exe,
        "--original_input", original_input or args.input,
        "--rel", str(rel),
    ]
    for d in dims:
        halo_cmd.extend(["--dim", str(d)])

    result = subprocess.run(halo_cmd, capture_output=True, text=True, env=env)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)

    out_dir = env.get("HALO_ARTIFACT_DIR", csv_dir).strip() or csv_dir
    for name in ("dists.npy", "mass_orig.npy", "mass_dec.npy"):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            print(f"[pipeline2] no .npy in {out_dir} (CSV-only ok)", file=sys.stderr)
            print("external:api=json:1")
            print(json.dumps({"dists": [], "mass_orig": [], "mass_dec": []}))
            sys.exit(0)

    dists = np.load(os.path.join(out_dir, "dists.npy"))
    mass_orig = np.load(os.path.join(out_dir, "mass_orig.npy"))
    mass_dec = np.load(os.path.join(out_dir, "mass_dec.npy"))
    print("external:api=json:1")
    print(json.dumps({
        "dists": dists.tolist(),
        "mass_orig": mass_orig.tolist(),
        "mass_dec": mass_dec.tolist(),
    }))
    sys.exit(0)

# ------------ Top-level：跑 pressio，external 指回本脚本 + env ------------
if args.external_mode and input_file and len(dims) >= 3:
    pipeline_path = os.path.abspath(__file__)
    os.makedirs(csv_dir, exist_ok=True)

    def run_one_rel(r):
        ad = csv_dir
        eb_str = _rel_to_eb_str(r)
        if _csv_artifacts_exist(ad, compressor, r, original_input):
            print(
                f"[pipeline2] skip rel={r} (eb={eb_str}): "
                f"halo_*_{compressor}_{eb_str}.csv already in {ad}",
                file=sys.stderr,
            )
            # 不用 "[external] skip:" 前缀，避免后面误走「external 未写 npy」空 JSON 分支
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="",
                stderr="[pipeline2] skip: csv already exists for this error bound",
            ), ad
        ext_cmd = (
            f"env HALO_ARTIFACT_DIR={shlex.quote(ad)} HALO_CSV_ONLY=1 "
            f"HALO_EB_STR={shlex.quote(eb_str)} "
            f"HALO_COMPRESSOR={shlex.quote(compressor)} "
            f"{shlex.quote(sys.executable)} {shlex.quote(pipeline_path)} "
            f"--external_mode --compressor {shlex.quote(compressor)} "
            f"--original_input {shlex.quote(original_input)} --rel {r}"
        )
        cmd = [pressio, "-i", input_file]
        input_lower = (input_file or "").lower()
        dataset = args.hdf5_dataset
        if dataset is None and (input_lower.endswith(".h5") or input_lower.endswith(".hdf5")):
            dataset = "/native_fields/baryon_density"
        if dataset:
            cmd.extend(["-I", dataset])
        cmd.extend([
            "-d", str(dims[0]), "-d", str(dims[1]), "-d", str(dims[2]),
            "-t", "float",
            "-b", f"compressor={compressor}",
            "-o", f"rel={r}",
            "-m", "error_stat", "-m", "size", "-m", "external",
            "-M", "all",
            "-o", "external:use_many=1",
            "-o", f"external:command={ext_cmd}",
        ])
        run_env = os.environ.copy()
        run_env.pop("PYTHONPATH", None)
        run_env["HALO_ARTIFACT_DIR"] = ad
        run_env["HALO_CSV_ONLY"] = "1"
        run_env["HALO_COMPRESSOR"] = compressor
        print(
            f"[pipeline2] compressor={compressor} rel={r} (eb={eb_str}) -> "
            f"halo_*_{compressor}_{eb_str}.csv",
            file=sys.stderr,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
        if args.verbose_pressio:
            _dump_pressio_io(proc)
        return proc, ad

    if len(rel_list) == 1:
        result, npy_dir = run_one_rel(rel_list[0])
        if (
            result.returncode == 0
            and "[external] skip:" not in (result.stderr or "")
            and not _rel_artifacts_ok(npy_dir, compressor, rel_list[0], original_input)
        ):
            _fail_missing_csv_after_pressio(
                npy_dir, compressor, rel_list[0], original_input, result
            )
    elif args.parallel:
        import concurrent.futures as cf
        ok = 0
        with cf.ThreadPoolExecutor(max_workers=min(len(rel_list), 8)) as ex:
            futs = {ex.submit(run_one_rel, r): r for r in rel_list}
            for f in cf.as_completed(futs):
                res, ad = f.result()
                r = futs[f]
                if res.returncode != 0:
                    print(f"[pipeline2] pressio failed rel={r}", file=sys.stderr)
                    print(res.stderr, file=sys.stderr)
                    sys.exit(1)
                if "[external] skip:" in (res.stderr or ""):
                    print(f"[pipeline2] skip rel={r}", file=sys.stderr)
                    ok += 1
                    continue
                if not _rel_artifacts_ok(ad, compressor, r, original_input):
                    _fail_missing_csv_after_pressio(ad, compressor, r, original_input, res)
                ok += 1
        print(f"[pipeline2] parallel done {ok}/{len(rel_list)}; CSV under {csv_dir}", file=sys.stderr)
        sys.exit(0 if ok == len(rel_list) else 1)
    else:
        last = None
        last_ad = None
        for r in rel_list:
            last, last_ad = run_one_rel(r)
            if last.returncode != 0:
                print(f"[pipeline2] pressio failed rel={r}", file=sys.stderr)
                print(last.stderr, file=sys.stderr)
                sys.exit(1)
            if "[external] skip:" in (last.stderr or ""):
                print(f"[pipeline2] skip rel={r}", file=sys.stderr)
                continue
            if not _rel_artifacts_ok(last_ad, compressor, r, original_input):
                _fail_missing_csv_after_pressio(last_ad, compressor, r, original_input, last)
        result = last
        npy_dir = last_ad

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    if "[external] skip:" in (result.stderr or ""):
        print("external:api=json:1")
        print(json.dumps({"dists": []}))
        sys.exit(0)

    if not glob.glob(os.path.join(csv_dir, "**", "halo_*.csv"), recursive=True):
        print(
            f"[pipeline2] no halo_*.csv under {csv_dir} (recursive); check pressio stderr",
            file=sys.stderr,
        )
        if result.stderr:
            print(result.stderr[-4000:], file=sys.stderr)

    # 无 npy 时直接空 JSON（CSV-only）
    if not all(os.path.exists(os.path.join(csv_dir, n)) for n in ("dists.npy", "mass_orig.npy", "mass_dec.npy")):
        print(f"[pipeline2] CSV-only exit 0; see {csv_dir}", file=sys.stderr)
        print("external:api=json:1", flush=True)
        print(json.dumps({"dists": [], "mass_orig": [], "mass_dec": []}), flush=True)
        sys.exit(0)

    dists = np.load(os.path.join(csv_dir, "dists.npy"))
    mass_orig = np.load(os.path.join(csv_dir, "mass_orig.npy"))
    mass_dec = np.load(os.path.join(csv_dir, "mass_dec.npy"))
    print("external:api=json:1", flush=True)
    print(json.dumps({
        "dists": dists.tolist(),
        "mass_orig": mass_orig.tolist(),
        "mass_dec": mass_dec.tolist(),
    }), flush=True)
    sys.exit(0)

sys.exit(1)
