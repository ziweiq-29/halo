#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import h5py
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

# 将所有输出重定向到 stderr，保持与 Pressio external 的行为一致
sys.stdout = sys.stderr

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    return result

def read_halo_output(filename: str) -> pd.DataFrame:
    rows = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            id_, x, y, z, n_cell, n_vert, mass, *extra = parts
            try:
                rows.append({
                    "id": int(id_),
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "n_cell": int(n_cell),
                    "n_vert": int(n_vert),
                    "mass": float(mass),
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)

def write_h5_from_binary(binary_file, dims, out_h5):
    data = np.fromfile(binary_file, dtype=np.float32)
    expected = int(np.prod(dims))
    if data.size != expected:
        print(f"❌ size mismatch: file has {data.size}, dims product is {expected}")
        sys.exit(1)
    data = data.reshape(tuple(reversed(dims)))
    with h5py.File(out_h5, "w") as f:
        grp = f.require_group("native_fields")
        if "baryon_density" in grp:
            del grp["baryon_density"]
        grp.create_dataset("baryon_density", data=data)

def run_halo_analysis(binary_file, dims, exe_path, tag, eval_uuid):
    tmp_h5  = f"{tag}_{eval_uuid}.h5"
    tmp_out = f"halo_output_{tag}_{eval_uuid}.txt"

    write_h5_from_binary(binary_file, dims, tmp_h5)

    cmd = [
        exe_path, "-b", "128", "-n", "-w",
        "-f", "native_fields/baryon_density",
        tmp_h5, "none", "none", tmp_out
    ]
    run_cmd(cmd)

    if not os.path.exists(tmp_out):
        print(f"❌ halo output not found: {tmp_out}")
        sys.exit(1)

    df = read_halo_output(tmp_out)
    print(f"halo:{tag}_num_halos={len(df)}")
    print(f"halo:{tag}_total_mass={df['mass'].sum():.4e}")
    return df, [tmp_h5, tmp_out]

def compute_metrics(df_orig: pd.DataFrame, df_dec: pd.DataFrame):

    dfm = pd.merge(df_orig[['id','mass']], df_dec[['id','mass']],
                   on='id', suffixes=('_orig','_decomp'))
    mass_orig = dfm['mass_orig'].to_numpy()
    mass_dec  = dfm['mass_decomp'].to_numpy()

    orig_xyz = df_orig[['x','y','z']].to_numpy()
    dec_xyz  = df_dec[['x','y','z']].to_numpy()
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dec_xyz)
    dists, _ = nn.kneighbors(orig_xyz)
    dists = dists.flatten()

    print("Nearest Neighbor Distance Stats:")
    mean   = float(np.mean(dists))
    median = float(np.median(dists))
    p90    = float(np.percentile(dists, 90))
    p99    = float(np.percentile(dists, 99))
    dmax   = float(np.max(dists))
    wmass  = float(wasserstein_distance(mass_orig, mass_dec))

    return {
        "mean": mean, "median": median, "p90": p90, "p99": p99, "max": dmax,
        "w_mass": wmass,
        "num_halos_orig": int(len(df_orig)),
        "num_halos_decomp": int(len(df_dec)),
    }

def cleanup(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Pressio external: dual halo + metrics (merged)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--decompressed", required=True)
    parser.add_argument("--external_exe", required=True)
    parser.add_argument("--dim", type=int, action="append", required=True)
    parser.add_argument("--eval_uuid", default="default")
    args, _ = parser.parse_known_args()

    dims = args.dim

  
    tmp_to_clean = []
    df_orig, tmp1 = run_halo_analysis(args.input, dims, args.external_exe, "original",     args.eval_uuid)
    df_dec,  tmp2 = run_halo_analysis(args.decompressed, dims, args.external_exe, "decompressed", args.eval_uuid)
    tmp_to_clean.extend(tmp1 + tmp2)


    df_orig.to_csv("halo_original.csv", index=False)
    df_dec.to_csv("halo_decompressed.csv", index=False)


    m = compute_metrics(df_orig, df_dec)

    print("Nearest Neighbor Distance Stats:")
    print(f"Mean:  {m['mean']:.4f}")
    print(f"Median:{m['median']:.4f}")
    print(f"90%:   {m['p90']:.4f}")
    print(f"99%:   {m['p99']:.4f}")
    print(f"Max:   {m['max']:.4f}")
    print(f"Wasserstein Distance between mass distributions: {m['w_mass']}")


    out_row = [[
        m["num_halos_orig"], m["num_halos_decomp"],
        m["mean"], m["median"], m["p90"], m["p99"], m["w_mass"]
    ]]
    pd.DataFrame(out_row, columns=[
        "num_halos_orig","num_halos_decomp","mean","median","p90","p99","wasserstein"
    ]).to_csv("halo_metrics.csv", index=False)


    cleanup(tmp_to_clean)

if __name__ == "__main__":
    main()
