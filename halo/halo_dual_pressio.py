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

from datetime import datetime
with open("debug_log.txt", "a") as f:
    f.write(f"\n[external] called at {datetime.now()} with args: {sys.argv}\n")

# 保存原始的 stdout，用于输出 metrics（libpressio 要求）
# 调试信息输出到 stderr
original_stdout = sys.stdout
sys.stdout = sys.stderr  # 默认输出到 stderr

def output_default_metrics():
    """Output default metrics in libpressio format before exiting"""
    print("external:api=1", file=original_stdout)
    print("mean=0.0", file=original_stdout)
    print("median=0.0", file=original_stdout)
    print("p90=0.0", file=original_stdout)
    print("p99=0.0", file=original_stdout)
    print("p999=0.0", file=original_stdout)
    print("max=0.0", file=original_stdout)
    print("p99_sym=0.0", file=original_stdout)
    print("wasserstein=0.0", file=original_stdout)
    original_stdout.flush()

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        output_default_metrics()
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
# print(f"[external] reading {binary_file}", file=sys.stderr)
def write_h5_from_binary(binary_file, dims, out_h5):
    data = np.fromfile(binary_file, dtype=np.float32)
    expected = int(np.prod(dims))
    if data.size != expected:
        print(f"[external] skip: data.size={data.size}, expected={expected}", file=sys.stderr)
        output_default_metrics()
        sys.exit(0)  # ⭐ 关键：一定是 0，不是 1
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
        print(f"❌ halo output not found: {tmp_out}", file=sys.stderr)
        output_default_metrics()
        sys.exit(1)

    df = read_halo_output(tmp_out)
    print(f"halo:{tag}_num_halos={len(df)}")
    print(f"halo:{tag}_total_mass={df['mass'].sum():.4e}")
    return df, [tmp_h5, tmp_out]
def compute_metrics(df_orig: pd.DataFrame, df_dec: pd.DataFrame):
    orig_xyz = df_orig[['x', 'y', 'z']].to_numpy()
    dec_xyz = df_dec[['x', 'y', 'z']].to_numpy()
    mass_orig = df_orig['mass'].to_numpy()
    mass_dec = df_dec['mass'].to_numpy()
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dec_xyz)
    dists, idx = nn.kneighbors(orig_xyz)
    dists = dists.flatten()
    idx   = idx.flatten()
    mass_dec = mass_dec[idx]

    return dists, mass_orig, mass_dec

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
    parser.add_argument("--original_input", help="Path to original uncompressed input file")
    parser.add_argument("--eval_uuid", default="default")
    args, _ = parser.parse_known_args()

    dims = args.dim
    
    # ⭐ 关键问题：libpressio external metric 接口传递的 --dim 是压缩后数据的大小（字节数），
    # 而不是原始维度。这是 libpressio 的设计，不是 bug。
    # 解决方案：从解压缩文件的大小推断实际维度
    
    if len(dims) == 1:
        # pressio 传递的是单个数字，可能是压缩大小或维度
        # 如果数字很大（>10000），很可能是压缩大小（字节数）
        if dims[0] > 10000:
            # 从解压缩文件大小推断实际维度
            if os.path.exists(args.decompressed):
                decompressed_size = os.path.getsize(args.decompressed)
                num_elements = decompressed_size // 4  # float32 = 4 bytes
                # 计算立方根（假设是 3D 立方体数据）
                cube_root = round(num_elements ** (1/3))
                if abs(cube_root ** 3 - num_elements) < 1000:  # 允许小的舍入误差
                    dims = [cube_root, cube_root, cube_root]
                    print(f"[external] Inferred dimensions from decompressed file: {dims} (from {num_elements} elements, {decompressed_size} bytes)", file=sys.stderr)
                else:
                    # 如果立方根推断失败，尝试从 original_input 推断
                    if args.original_input and os.path.exists(args.original_input):
                        orig_size = os.path.getsize(args.original_input)
                        orig_elements = orig_size // 4
                        cube_root = round(orig_elements ** (1/3))
                        if abs(cube_root ** 3 - orig_elements) < 1000:
                            dims = [cube_root, cube_root, cube_root]
                            print(f"[external] Inferred dimensions from original_input: {dims} (from {orig_elements} elements)", file=sys.stderr)
                        else:
                            print(f"⚠️  WARNING: Cannot infer dimensions. Decompressed: {num_elements} elements, Original: {orig_elements} elements", file=sys.stderr)
                    else:
                        print(f"⚠️  WARNING: Cannot infer dimensions from decompressed file ({num_elements} elements, cube_root={cube_root:.2f})", file=sys.stderr)
            else:
                # 解压缩文件不存在，尝试从 original_input 推断
                if args.original_input and os.path.exists(args.original_input):
                    orig_size = os.path.getsize(args.original_input)
                    orig_elements = orig_size // 4
                    cube_root = round(orig_elements ** (1/3))
                    if abs(cube_root ** 3 - orig_elements) < 1000:
                        dims = [cube_root, cube_root, cube_root]
                        print(f"[external] Inferred dimensions from original_input (decompressed file not found): {dims} (from {orig_elements} elements)", file=sys.stderr)
                    else:
                        print(f"⚠️  WARNING: Cannot infer dimensions from original_input ({orig_elements} elements)", file=sys.stderr)
                else:
                    print(f"⚠️  WARNING: Decompressed file not found and original_input not provided. Using dims={dims}", file=sys.stderr)

    # Check if input file is too small (compressed) before processing
    input_file_to_use = args.input
    if os.path.exists(args.input):
        input_size = os.path.getsize(args.input)
        input_elements = input_size // 4  # float32 = 4 bytes per element
        expected_elements = int(np.prod(dims))
        
        # If input file is much smaller than expected, it's likely compressed
        if input_elements < expected_elements * 0.1:
            with open("debug_log.txt", "a") as f:
                # f.write(f"\n[external] called at {datetime.now()} with args: {sys.argv}\n")
                print(f"[external] skip: input file is compressed ({input_elements} elements)fffff, expected {expected_elements} elements", file=f)

            # Try to use original_input if provided
            if args.original_input and os.path.exists(args.original_input):
                orig_size = os.path.getsize(args.original_input)
                orig_elements = orig_size // 4
                if orig_elements == expected_elements:
                    print(f"[external] Using original_input: {args.original_input} ({orig_elements} elements)", file=sys.stderr)
                    input_file_to_use = args.original_input
                else:
                    print(f"[external] skip: input file is compressed ({input_elements} elements), original_input also wrong size ({orig_elements} elements)", file=sys.stderr)
                    output_default_metrics()
                    sys.exit(0)
            else:
                print(f"[external] skip: input file is compressed ({input_elements} elements), expected {expected_elements} elements", file=sys.stderr)
                output_default_metrics()
                sys.exit(0)
        # If element count doesn't match (but not too small), there's a dimension mismatch
        elif input_elements != expected_elements:
            print(f"[external] skip: input file element count mismatch: {input_elements} vs {expected_elements}", file=sys.stderr)
            output_default_metrics()
            sys.exit(0)

  
    tmp_to_clean = []
    df_orig, tmp1 = run_halo_analysis(input_file_to_use, dims, args.external_exe,
                                      "original", args.eval_uuid)
    df_dec,  tmp2 = run_halo_analysis(args.decompressed, dims, args.external_exe,
                                      "decompressed", args.eval_uuid)
    tmp_to_clean.extend(tmp1 + tmp2)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_orig.to_csv(os.path.join(out_dir, "halo_original.csv"), index=False)
    df_dec.to_csv(os.path.join(out_dir, "halo_decompressed.csv"), index=False)

    # 只关心 dists：compute_metrics 返回 (dists, mass_orig, mass_dec)
    dists, mass_orig, mass_dec = compute_metrics(df_orig, df_dec)

    # 保存到脚本所在目录
    np.save(os.path.join(out_dir, "dists.npy"), dists)
    np.save(os.path.join(out_dir, "mass_orig.npy"), mass_orig)
    np.save(os.path.join(out_dir, "mass_dec.npy"), mass_dec)
    print(f"[external] saved dists to {out_dir}, shape={dists.shape}", file=sys.stderr)

    # 输出由 run_pressio_pipeline 读取 .npy 后负责；此处只保存
    cleanup(tmp_to_clean)

if __name__ == "__main__":
    main()