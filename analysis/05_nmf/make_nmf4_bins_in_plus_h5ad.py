import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import pandas as pd
import scanpy as sc

IN_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca__PLUS_nmf4.h5ad"
OUT_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca__PLUS_nmf4_bins.h5ad"

adata = sc.read_h5ad(IN_H5AD)

if "lt_nmf4" not in adata.obs.columns:
    raise RuntimeError("lt_nmf4 not found in adata.obs")

# 3 jednaka tercila: low/mid/high
adata.obs["nmf4_bin"] = pd.qcut(adata.obs["lt_nmf4"], q=3, labels=["low","mid","high"])
adata.obs["nmf4_bin"] = adata.obs["nmf4_bin"].astype("category")

print("nmf4_bin counts:")
print(adata.obs["nmf4_bin"].value_counts())

adata.write_h5ad(OUT_H5AD)
print("SAVED:", OUT_H5AD)
