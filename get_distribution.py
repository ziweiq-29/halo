# halo_env 是 py3.9；若 PYTHONPATH/sys.path 含 libpressio-env，会加载 py3.11 的 numpy .so → ImportError
import os
import sys
os.environ.pop("PYTHONPATH", None)
_pp = "/anvil/projects/x-cis240669/libpressio-env"
sys.path = [p for p in sys.path if not (p.startswith(_pp) or _pp in p)]

import pandas as pd
import numpy as np
# 集群无显示器时 plt.show() 无效；用 Agg 后端 + savefig 写文件
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1. 实验参数（文件名与 pipeline2 / halo_dual_pressio2 一致：sz3_1e-3.csv）
# ================================

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
COMPRESSOR = "sz3"  # 与 run_pressio_pipeline2 --compressor 一致


def _rel_to_eb_str(r):
    """与 halo_dual_pressio2 文件名一致：1e-3、5e-2，不是 0.001。"""
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


error_bounds = [1e-3, 5e-3, 5e-2, 5e-4, 5e-5]

# Wasserstein distance（你的统计表）
wasserstein = {
    0.001: 647.035,
    0.005: 10613,
    0.05: 69455.6,
    0.0005: 1101.37,
    0.00005: 94.5272
    
}



rows = []

for eb in error_bounds:
    eb_str = _rel_to_eb_str(eb)
    orig_file = os.path.join(CSV_DIR, f"halo_original_{COMPRESSOR}_{eb_str}.csv")
    dec_file = os.path.join(CSV_DIR, f"halo_decompressed_{COMPRESSOR}_{eb_str}.csv")

    orig = pd.read_csv(orig_file)
    dec  = pd.read_csv(dec_file)

    # original halos
    for m in orig["mass"]:
        rows.append({
            "error_bound": eb,
            "type": "original",
            "mass": m
        })

    # decompressed halos
    for m in dec["mass"]:
        rows.append({
            "error_bound": eb,
            "type": "decompressed",
            "mass": m
        })

df = pd.DataFrame(rows)

# ================================
# 3. 处理 halo mass（log scale）
# ================================

df["log_mass"] = np.log10(df["mass"])

# ================================
# 4. 画 violin plot
# ================================

plt.figure(figsize=(10,6))

sns.violinplot(
    data=df,
    x="error_bound",
    y="log_mass",
    hue="type",
    split=True,
    inner="quartile"
)

# ================================
# 5. 添加 Wasserstein 到 x-axis
# ================================

labels = [
    f"{eb}\nW={wasserstein[eb]}"
    for eb in error_bounds
]
sorted_error_bounds = sorted(error_bounds)
sorted_labels = [labels[error_bounds.index(eb)] for eb in sorted_error_bounds]
plt.xticks(range(len(sorted_error_bounds)), sorted_labels)

# ================================
# 6. 图像标签
# ================================

plt.xlabel("Error Bound (Wasserstein Distance)")
plt.ylabel("log10(Halo Mass)")
plt.title("Halo Mass Distribution: Original vs Decompressed")

plt.legend(title="Data Type")

plt.tight_layout()

out_png = os.path.join(CSV_DIR, f"halo_distribution_{COMPRESSOR}.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"[get_distribution] wrote {out_png}", file=sys.stderr)