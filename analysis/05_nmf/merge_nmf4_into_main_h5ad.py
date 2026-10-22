import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import scanpy as sc
import pandas as pd

MAIN_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
NMF4_H5AD = DATA_ROOT + "/velocity/03_outputs/tabular/latent_time_programs_nmf/merged_3samples__cellrank_gpcca__nmf4.h5ad"
OUT_H5AD  = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca__PLUS_nmf4.h5ad"

adata_main = sc.read_h5ad(MAIN_H5AD)
adata_nmf4 = sc.read_h5ad(NMF4_H5AD)

# uzmi samo potrebne kolone iz nmf4 objekta
need = [c for c in ["nmf4", "nmf4_bin", "lt_nmf4", "lt_nmf2"] if c in adata_nmf4.obs.columns]
if not need:
    raise RuntimeError("Nijedna od kolona nmf4/nmf4_bin/lt_nmf4/lt_nmf2 nije pronađena u NMF4_H5AD.")

obs_add = adata_nmf4.obs[need].copy()

# join po obs_names (cell IDs)
before = adata_main.n_obs
adata_main.obs = adata_main.obs.join(obs_add, how="left")

# report
missing = adata_main.obs[need].isna().sum()
print("Main cells:", before)
print("Columns added:", need)
print("Missing values after join (should be 0 or vrlo malo):")
print(missing)

# snimi
adata_main.write_h5ad(OUT_H5AD)
print("SAVED:", OUT_H5AD)
