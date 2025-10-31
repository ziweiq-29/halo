#!/usr/bin/env python3
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import glob

input_file = "/home/ziweiq2/LibPressio/dataset/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"
dims = [512, 512, 512]
halo_exe = "/home/ziweiq2/halo/reeber/build/examples/amr-connected-components/amr_connected_components_float"

rel_errors = np.logspace(-6, -1, num=20).tolist()  
# rel_errors.extend([1e-1 ,5e-2 ,1e-2, 5e-3 ,1e-3, 5e-4, 1e-4 ,5e-5, 1e-5, 5e-6, 1e-6])
rel_errors = sorted(list(set(rel_errors)), reverse=True) 
external_script = "halo_dual_pressio.py"
pressio = "pressio"
rel_errors_cr=np.logspace(-7, np.log10(6.15e-6), num=20)

csv_rows = []
csv_rows_cr=[]
for rel in rel_errors_cr:
    print(f"\n===== Running rel={rel} =====")

    pressio_cmd = [
        pressio,
        "-i", input_file,
        "-d", str(dims[0]), "-d", str(dims[1]), "-d", str(dims[2]),
        "-t", "float",
        "sz3", "-o", f"rel={rel}",
        "-m", "error_stat", "-m", "size", "-m", "external",
        "-M", "all",
        "-o", "external:use_many=1",
        "-o", f"external:command=python3 {external_script} --external_exe {halo_exe}"
    ]

    pressio_out = subprocess.run(pressio_cmd, check=True, capture_output=True, text=True)


    comp_ratio = None
    for line in pressio_out.stderr.splitlines():
        if "size:compression_ratio" in line:
            try:
                comp_ratio = float(line.split("=")[1].strip())
            except:
                comp_ratio = None
            break

    if comp_ratio is None:
        print("⚠️  Warning: compression_ratio not found for this rel. Check if -m size is enabled.")
    else:
        print(f"✅ compression_ratio = {comp_ratio}")
    csv_rows_cr.append([rel,comp_ratio])
    
    os.rename("halo_original.csv", f"halo_original_rel{rel}.csv")
    os.rename("halo_decompressed.csv", f"halo_decompressed_rel{rel}.csv")
    os.rename("halo_metrics.csv", f"halo_metrics_rel{rel}.csv") 


        # print("✅ Deleted all halo_metrics_rel*.csv files")
        
        

for rel in rel_errors:
    print(f"\n===== Running rel={rel} =====")

    pressio_cmd = [
        pressio,
        "-i", input_file,
        "-d", str(dims[0]), "-d", str(dims[1]), "-d", str(dims[2]),
        "-t", "float",
        "sz3", "-o", f"rel={rel}",
        "-m", "error_stat", "-m", "size", "-m", "external",
        "-M", "all",
        "-o", "external:use_many=1",
        "-o", f"external:command=python3 {external_script} --external_exe {halo_exe}"
    ]
    # subprocess.run(pressio_cmd, check=True)
    # ✅ Run Pressio and capture output
    pressio_out = subprocess.run(pressio_cmd, check=True, capture_output=True, text=True)





    with open("halo_metrics.csv") as f:
        r = next(csv.DictReader(f))
        mean = float(r["mean"])
        median = float(r["median"])
        p90 = float(r["p90"])
        p99 = float(r["p99"])
        wdist = float(r["wasserstein"])
        p999 = float(r["p999"])
        max = float(r["max"])




    csv_rows.append([rel, mean, median, p90, p99,p999, wdist,max])

    os.rename("halo_original.csv", f"halo_original_rel{rel}.csv")
    os.rename("halo_decompressed.csv", f"halo_decompressed_rel{rel}.csv")
    os.rename("halo_metrics.csv", f"halo_metrics_rel{rel}.csv") 

    for f in os.listdir("."):
        if f.startswith(".pressio"):
            os.remove(f)



for pattern in ["halo_metrics_rel*.csv", "halo_decompressed_rel*.csv", "halo_original_rel*.csv"]:
    for f in glob.glob(pattern):
        os.remove(f)
        print(f"✅ Deleted {f}")
df = pd.DataFrame(csv_rows, columns=[ "rel_error","mean", "median", "p90", "p99","p999","wasserstein","max" ]) 
df_cr = pd.DataFrame(csv_rows_cr,columns=["rel_errors_cr","compression_ratio"])
df.to_csv("metrics_summary.csv", index=False)
df_cr.to_csv("metrics_summary_cr.csv", index=False)



plot_dir = "metrics_plots"
os.makedirs(plot_dir, exist_ok=True)

def plot_metric(col_name, filename, ylabel):
    plt.figure()
    plt.plot(df["rel_error"], df[col_name], marker="o")
    plt.xscale("log")
    plt.xlabel("Relative Error")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Relative Error")
    plt.grid()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def plot_metric_cr(col_name, filename, ylabel):
    plt.figure()
    plt.plot(df_cr["rel_errors_cr"], df_cr[col_name], marker="o")
    plt.xscale("log")
    plt.xlabel("Relative Error")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Relative Error")
    plt.grid()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
    
plot_metric("mean", "mean_vs_relerror.png", "Mean NN Distance")
plot_metric("median", "median_vs_relerror.png", "Median NN Distance")
plot_metric("p90", "p90_vs_relerror.png", "90% NN Distance")
plot_metric("p99", "p99_vs_relerror.png", "99% NN Distance")
plot_metric("p999", "p999_vs_relerror.png", "99.9% NN Distance")
plot_metric("wasserstein", "wasserstein_vs_relerror.png", "Wasserstein Mass Distance")
plot_metric("max","max_vs_relerror.png","Maximum Distance")
plot_metric_cr("compression_ratio", "compression_ratio_vs_relerror.png", "Compression Ratio")  