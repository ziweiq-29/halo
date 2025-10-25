import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors



df_original = pd.read_csv("halo_original.csv")
df_decompressed = pd.read_csv("halo_decompressed.csv")
df_merged = pd.merge(df_original[['id', 'mass']], 
                     df_decompressed[['id', 'mass']], 
                     on='id', 
                     suffixes=('_orig', '_decomp'))
print("merged lenth", len(df_merged))
mass_orig = df_merged['mass_orig'].values
mass_decomp = df_merged['mass_decomp'].values



orig_xyz = df_original[['x','y','z']].values
dec_xyz  = df_decompressed[['x','y','z']].values

nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dec_xyz)
distances, indices = nn.kneighbors(orig_xyz)


distances = distances.flatten()

print("Nearest Neighbor Distance Stats:")
print(f"Mean:  {np.mean(distances):.4f}")
print(f"Median:{np.median(distances):.4f}")
print(f"90%:   {np.percentile(distances, 90):.4f}")
print(f"99%:   {np.percentile(distances, 99):.4f}")
print(f"Max:   {np.max(distances):.4f}")

distance = wasserstein_distance(mass_orig, mass_decomp)
print("Wasserstein Distance between mass distributions:", distance)
