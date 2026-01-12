#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import csv

# ------------ Command-line args ------------
parser = argparse.ArgumentParser()
parser.add_argument("--external_mode", action="store_true")
# Use parse_known_args to ignore LibPressio's additional arguments (--api, --input, --decompressed, etc.)
args, unknown = parser.parse_known_args()

# ------------ Dataset & Paths ------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

input_file = "/home/ziweiq2/LibPressio/dataset/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"
dims = [512, 512, 512]
halo_exe = "/home/ziweiq2/halo/reeber/build/examples/amr-connected-components/amr_connected_components_float"
external_script = os.path.join(script_dir, "halo_dual_pressio.py")  # Use absolute path
pressio = "pressio"
rel = 1e-1

if args.external_mode:
    # Run Pressio and capture output
    pressio_cmd = [
        pressio,
        "-i", input_file,
        "-d", str(dims[0]), "-d", str(dims[1]), "-d", str(dims[2]),
        "-t", "float",
        "sz3", "-o", f"rel={rel}",
        "-m", "error_stat", "-m", "size", "-m", "external",
        "-M", "all",
        "-o", "external:use_many=1",
        "-o", f"external:command=python3 {external_script} --external_exe {halo_exe} --original_input {input_file}"
    ]

    # Run pressio in the script's directory to ensure relative paths work
    result = subprocess.run(pressio_cmd, capture_output=True, text=True, cwd=script_dir)
    
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
    halo_metrics_path = os.path.join(script_dir, "halo_metrics.csv")
    if not os.path.exists(halo_metrics_path):
        print("❌ Error: halo_metrics.csv was not created by the external script", file=sys.stderr)
        print(f"Pressio stderr:\n{result.stderr}", file=sys.stderr)
        # Output default metrics even on error (libpressio requires format)
        print("external:api=1")
        print("data=0.0")
        sys.exit(1)

    # ---- Read mean from halo_metrics.csv ----
    with open(halo_metrics_path) as f:
        r = next(csv.DictReader(f))
        mean = float(r["mean"])

    # ---- Output in libpressio external metric format ----
    print("external:api=1")
    print(f"data={mean}")
    sys.exit(0)
