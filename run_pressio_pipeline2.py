#!/usr/bin/env python3
"""
Pipeline2：CSV 统一写 halo/csv；默认 halo_dual_pressio2 + 多 rel / HDF5；
不改动 run_pressio_pipeline.py，run_halo / run_halo_pressio 仍用原版。
"""
import argparse
import subprocess
import sys
import os
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
    env["HALO_COMPRESSOR"] = compressor
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
        ext_cmd = (
            f"env HALO_ARTIFACT_DIR={shlex.quote(ad)} HALO_CSV_ONLY=1 "
            f"HALO_EB_STR={shlex.quote(eb_str)} "
            f"HALO_COMPRESSOR={shlex.quote(compressor)} "
            f"{shlex.quote(sys.executable)} {shlex.quote(pipeline_path)} "
            f"--external_mode --original_input {shlex.quote(original_input)} --rel {r}"
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
        return proc, ad

    if len(rel_list) == 1:
        result, npy_dir = run_one_rel(rel_list[0])
    elif args.parallel:
        import concurrent.futures as cf
        ok = 0
        with cf.ThreadPoolExecutor(max_workers=min(len(rel_list), 8)) as ex:
            futs = {ex.submit(run_one_rel, r): r for r in rel_list}
            for f in cf.as_completed(futs):
                res, _ = f.result()
                if res.returncode == 0:
                    ok += 1
                else:
                    print(f"[pipeline2] pressio failed rel={futs[f]}", file=sys.stderr)
                    print(res.stderr, file=sys.stderr)
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
        result = last
        npy_dir = last_ad

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    if "[external] skip:" in (result.stderr or ""):
        print("external:api=json:1")
        print(json.dumps({"dists": []}))
        sys.exit(0)

    if not glob.glob(os.path.join(csv_dir, "halo_*.csv")):
        print(f"[pipeline2] no halo_*.csv in {csv_dir}; check pressio stderr", file=sys.stderr)
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
