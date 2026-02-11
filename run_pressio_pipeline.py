#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import csv
import json
import numpy as np

# ------------ Command-line args ------------
parser = argparse.ArgumentParser()
parser.add_argument("--external_mode", action="store_true")
parser.add_argument("--input", help="Path to input data file (e.g. .f32)")
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
dims = args.dim
rel = args.rel
compressor = args.compressor
halo_exe = args.halo_exe
external_script = args.external_script
pressio = args.pressio



if args.external_mode:
    # Run Pressio and capture output
    pressio_cmd = [
        pressio,
        "-i", input_file,
        "-d", str(dims[0]), "-d", str(dims[1]), "-d", str(dims[2]),
        "-t", "float",
        # compressor, "-o", f"rel={rel}",
        "-b", f"compressor={compressor}",
        "-o", f"rel={rel}",
        "-m", "error_stat", "-m", "size", "-m", "external",
        "-M", "all",
        "-o", "external:use_many=1",
        "-o", f"external:command=python3 {external_script} --external_exe {halo_exe} --original_input {input_file}"
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

    # ---- Read dists from dists.npy ----
    if not os.path.exists("dists.npy"):
        print("❌ Error: dists.npy was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        # Output default metrics even on error (libpressio requires format)
        print("external:api=json:1", file=sys.stdout, flush=True)
        print(json.dumps({"dists": []}), file=sys.stdout, flush=True)
        sys.exit(1)
    
    if not os.path.exists("mass_orig.npy"):
        print("❌ Error: mass_orig.npy was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        # Output default metrics even on error (libpressio requires format)
        print("external:api=json:1", file=sys.stdout, flush=True)
        print(json.dumps({"dists": []}), file=sys.stdout, flush=True)
        sys.exit(1)
        
    if not os.path.exists("mass_dec.npy"):
        print("❌ Error: mass_dec.npy was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        # Output default metrics even on error (libpressio requires format)
        print("external:api=json:1", file=sys.stdout, flush=True)
        print(json.dumps({"dists": []}), file=sys.stdout, flush=True)
        sys.exit(1)
    
    # Load dists array
    dists = np.load("dists.npy")
    mass_orig = np.load("mass_orig.npy")
    mass_dec = np.load("mass_dec.npy")
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