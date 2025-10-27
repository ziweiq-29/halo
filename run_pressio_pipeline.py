#!/usr/bin/env python3
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv


input_file = "/home/ziweiq2/LibPressio/dataset/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"
dims = [512, 512, 512]
halo_exe = "/home/ziweiq2/halo/reeber/build/examples/amr-connected-components/amr_connected_components_float"
rel_errors = [1e-1 ,5e-2 ,1e-2, 5e-3 ,1e-3, 5e-4, 1e-4 ,5e-5, 1e-5, 5e-6, 1e-6]
external_script = "halo_dual_pressio.py"
pressio = "pressio"


csv_rows = []
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
    subprocess.run(pressio_cmd, check=True)


    with open("halo_metrics.csv") as f:
        r = next(csv.DictReader(f))
        mean = float(r["mean"])
        median = float(r["median"])
        p90 = float(r["p90"])
        p99 = float(r["p99"])
        wdist = float(r["wasserstein"])

    csv_rows.append([rel, mean, median, p90, p99, wdist])

    os.rename("halo_original.csv", f"halo_original_rel{rel}.csv")
    os.rename("halo_decompressed.csv", f"halo_decompressed_rel{rel}.csv")
    os.rename("halo_metrics.csv", f"halo_metrics_rel{rel}.csv") 

    for f in os.listdir("."):
        if f.startswith(".pressio"):
            os.remove(f)


df = pd.DataFrame(csv_rows, columns=["rel_error", "mean", "median", "p90", "p99", "wasserstein"])
df.to_csv("metrics_summary.csv", index=False)
print("\n✅ Saved metrics_summary.csv")


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

plot_metric("mean", "mean_vs_relerror.png", "Mean NN Distance")
plot_metric("median", "median_vs_relerror.png", "Median NN Distance")
plot_metric("p90", "p90_vs_relerror.png", "90% NN Distance")
plot_metric("p99", "p99_vs_relerror.png", "99% NN Distance")
plot_metric("wasserstein", "wasserstein_vs_relerror.png", "Wasserstein Mass Distance")

