#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import h5py
import subprocess
import os
import sys
sys.stdout = sys.stderr
# print("🚀 Script started!", file=sys.stderr)
sys.stderr.flush()

def run_cmd(cmd):

    print("▶ Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Command failed:", result.stderr)
        sys.exit(1)
    return result


def read_halo_output(filename: str) -> pd.DataFrame:

    results = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            id_, x, y, z, n_cell, n_vert, mass, *extra = parts
            results.append({
                "id": int(id_),
                "x": int(x),
                "y": int(y),
                "z": int(z),
                "n_cell": int(n_cell),
                "n_vert": int(n_vert),
                "mass": float(mass)
            })
    return pd.DataFrame(results)


def run_halo_analysis(binary_file, dims, exe_path, tag, eval_uuid):
    print("run run_halo_analysis()", file=sys.stderr)
    sys.stderr.flush()

    # ✅ 用 eval_uuid 让文件名唯一，避免并行冲突
    tmp_h5 = f"{tag}_{eval_uuid}.h5"
    tmp_out = f"halo_output_{tag}_{eval_uuid}.txt"

    data = np.fromfile(binary_file, dtype=np.float32).reshape(tuple(reversed(dims)))
    with h5py.File(tmp_h5, "w") as f:
        grp = f.require_group("native_fields")
        grp.create_dataset("baryon_density", data=data)

    cmd = [
        exe_path,
        "-b", "128", "-n", "-w",
        "-f", "native_fields/baryon_density",
        tmp_h5, "none", "none", tmp_out
    ]
    run_cmd(cmd)

    if not os.path.exists(tmp_out):
        print(f"❌ Halo output not found for {tag}: {tmp_out}", file=sys.stderr)
        sys.exit(1)

    df = read_halo_output(tmp_out)
    print(f"halo:{tag}_num_halos={len(df)}")
    print(f"halo:{tag}_total_mass={df['mass'].sum():.4e}")
    return df



def cleanup(eval_uuid):
    files = [
        f"original_{eval_uuid}.h5",
        f"decompressed_{eval_uuid}.h5",
        f"halo_output_original_{eval_uuid}.txt",
        f"halo_output_decompressed_{eval_uuid}.txt"
    ]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            # print(f"🧹 Removed temp file: {f}", file=sys.stderr)

def main():
    
    print("🚀 Script started!", file=sys.stderr)
    sys.stderr.flush()

    parser = argparse.ArgumentParser(description="Pressio external command for dual halo analysis")
    parser.add_argument("--input", required=True)                 
    parser.add_argument("--external_exe", required=True)                   
    parser.add_argument("--decompressed", required=True)         
    parser.add_argument("--dim", type=int, action="append")  
    parser.add_argument("--eval_uuid", default="default", help="Unique ID for parallel safe temp files")
    
    args, unknown = parser.parse_known_args()


    dims = args.dim if args.dim else []
    print("sys.argv:", sys.argv, file=sys.stderr)
    print(f"🔹 Unknown: {unknown}", file=sys.stderr)
    print(f"▶ args.input: {args.input}", file=sys.stderr)
    print(f"▶ args.decompressed: {args.decompressed}", file=sys.stderr)
    print(f"▶ Halo executable: {args.external_exe}", file=sys.stderr)
    print(f"📏 dims(from --dim): {dims}", file=sys.stderr)
    sys.stderr.flush()


    orig_df = run_halo_analysis(args.input, dims, args.external_exe, tag="original", eval_uuid=args.eval_uuid)
    dec_df  = run_halo_analysis(args.decompressed, dims, args.external_exe, tag="decompressed", eval_uuid=args.eval_uuid)

    
    orig_df.to_csv("halo_original.csv", index=False)
    dec_df.to_csv("halo_decompressed.csv", index=False)
    cleanup(args.eval_uuid)
    




    
if __name__ == "__main__":
    main()