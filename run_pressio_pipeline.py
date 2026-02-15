#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import json
import numpy as np

# ------------ Command-line args ------------
parser = argparse.ArgumentParser(description="Pipeline: as external cmd delegates to halo_dual_pressio; as top-level runs pressio")
parser.add_argument("--external_mode", action="store_true")
parser.add_argument("--input", help="Input path")
parser.add_argument("--decompressed", help="Decompressed path (LibPressio injects when calling us as external)")
parser.add_argument("--original_input", help="Original input path (HDF5/.f32); pass in external:command")
parser.add_argument("--dim", type=int, action="append", help="Dimensions (e.g. --dim 512 --dim 512 --dim 512)")
parser.add_argument("--rel", type=float, default=1e-1, help="Relative error for compressor (default: 0.1)")
parser.add_argument("--compressor", default="sz3", help="Compressor name for pressio (default: sz3)")
parser.add_argument("--halo_exe", default="/home/ziweiq2/halo/reeber/build/examples/amr-connected-components/amr_connected_components_float", help="Path to halo executable")
parser.add_argument("--external_script", default="/home/ziweiq2/halo/halo_dual_pressio.py", help="Path to halo_dual_pressio.py")
parser.add_argument("--pressio", default="pressio", help="Pressio command (default: pressio)")
# Use parse_known_args to ignore LibPressio's additional arguments (--api, --input, --decompressed, etc.)
args, unknown = parser.parse_known_args()

# ------------ Dataset & Paths ------------
input_file = args.input
dims = args.dim or []
rel = args.rel
compressor = args.compressor
halo_exe = args.halo_exe
script_dir = os.path.dirname(os.path.abspath(__file__))
external_script = args.external_script if os.path.isabs(args.external_script) else os.path.join(script_dir, args.external_script)
pressio = args.pressio
original_input = args.original_input or args.input

# ------------ 被 LibPressio 作为 external 调用 -> 转给 halo_dual_pressio，只输出 dists 等 ------------
# 首次 launch（probe）无 --decompressed，输出默认格式避免 return 1
if not args.decompressed:
    print("external:api=json:1")
    print(json.dumps({"dists": []}))
    sys.exit(0)

if args.decompressed:
    halo_cmd = [sys.executable, external_script, "--input", args.input, "--decompressed", args.decompressed,
                "--external_exe", halo_exe, "--original_input", original_input or args.input]
    for d in dims:
        halo_cmd.extend(["--dim", str(d)])
    result = subprocess.run(halo_cmd, capture_output=True, text=True)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)
    # 读取 halo_dual_pressio 保存的 .npy，输出 LibPressio 格式
    out_dir = os.path.dirname(os.path.abspath(external_script))
    dists_path = os.path.join(out_dir, "dists.npy")
    mass_orig_path = os.path.join(out_dir, "mass_orig.npy")
    mass_dec_path = os.path.join(out_dir, "mass_dec.npy")
    if not os.path.exists(dists_path) or not os.path.exists(mass_orig_path) or not os.path.exists(mass_dec_path):
        print(f"[run_pressio_pipeline] missing .npy in {out_dir}", file=sys.stderr)
        print("external:api=json:1")
        print(json.dumps({"dists": []}))
        sys.exit(1)
    dists = np.load(dists_path)
    mass_orig = np.load(mass_orig_path)
    mass_dec = np.load(mass_dec_path)
    print("external:api=json:1")
    print(json.dumps({"dists": dists.tolist(), "mass_orig": mass_orig.tolist(), "mass_dec": mass_dec.tolist()}))
    sys.exit(0)

# ------------ Top-level: 运行 pressio，external:command 指向自己 ------------
if args.external_mode and input_file and len(dims) >= 3:
    # Run Pressio and capture output
    pipeline_path = os.path.abspath(__file__)
    ext_cmd = f"python {pipeline_path} --external_mode --original_input {original_input}"
    pressio_cmd = [
        pressio,
        "-i", input_file,
        "-d", str(dims[0]), "-d", str(dims[1]), "-d", str(dims[2]),
        "-t", "float",
        "-b", f"compressor={compressor}",
        "-o", f"rel={rel}",
        "-m", "error_stat", "-m", "size", "-m", "external",
        "-M", "all",
        "-o", "external:use_many=1",
        "-o", f"external:command={ext_cmd}",
    ]
    # print("Pressio command: ", pressio_cmd)

    result = subprocess.run(pressio_cmd, capture_output=True, text=True)
    
    # Debug: check pressio return code
    if result.returncode != 0:
        print(f"❌ Pressio command failed with return code {result.returncode}", file=sys.stderr)
        print(f"Pressio stdout:\n{result.stdout}", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        # Output default metrics even on pressio failure
        print("external:api=1")
        print("data=0.0")
        sys.exit(1)

    # ---- Check if skip occurred (i.e., not full field) ----
    if "[external] skip:" in result.stderr:
        # Output default metrics when skipping
        print("external:api=1")
        print("data=0.0")
        sys.exit(0)

    # ---- Now expect halo_metrics.csv to exist ----
    if not os.path.exists("halo_metrics.csv"):
        print("❌ Error: halo_metrics.csv was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        # Output default metrics even on error (libpressio requires format)
        print("external:api=1")
        print("data=0.0")
        sys.exit(1)

    # ---- Read dists from script dir (halo_dual_pressio writes there) ----
    npy_dir = os.path.dirname(os.path.abspath(external_script))
    dists_path = os.path.join(npy_dir, "dists.npy")
    mass_orig_path = os.path.join(npy_dir, "mass_orig.npy")
    mass_dec_path = os.path.join(npy_dir, "mass_dec.npy")
    if not os.path.exists(dists_path):
        print("❌ Error: dists.npy was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        print("external:api=json:1", file=sys.stdout, flush=True)
        print(json.dumps({"dists": []}), file=sys.stdout, flush=True)
        sys.exit(1)
    if not os.path.exists(mass_orig_path):
        print("❌ Error: mass_orig.npy was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        print("external:api=json:1", file=sys.stdout, flush=True)
        print(json.dumps({"dists": []}), file=sys.stdout, flush=True)
        sys.exit(1)
    if not os.path.exists(mass_dec_path):
        print("❌ Error: mass_dec.npy was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        print("external:api=json:1", file=sys.stdout, flush=True)
        print(json.dumps({"dists": []}), file=sys.stdout, flush=True)
        sys.exit(1)

    dists = np.load(dists_path)
    mass_orig = np.load(mass_orig_path)
    mass_dec = np.load(mass_dec_path)
    # Convert numpy array to list for JSON serialization
    dists_list = dists.tolist()
    mass_orig_list = mass_orig.tolist()
    mass_dec_list = mass_dec.tolist()
    # ---- Output in libpressio external metric format (JSON) ----
    print("external:api=json:1", file=sys.stdout, flush=True)
    # Output dists + mass + 3 lengths for inspection
    metrics = {
        "dists": dists_list,
        "mass_orig": mass_orig_list,
        "mass_dec": mass_dec_list,
        # "len_dists": len(dists_list),
        # "len_mass_orig": len(mass_orig_list),
        # "len_mass_dec": len(mass_dec_list),
    }
    print(json.dumps(metrics), file=sys.stdout, flush=True)
    sys.exit(0)

# print("Usage: As external (LibPressio calls): needs --decompressed. As top-level: python run_pressio_pipeline.py --external_mode -i <file> -d 512 -d 512 -d 512", file=sys.stderr)
sys.exit(1)
