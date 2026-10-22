"""
Cluster markers on the full log-normalised transcriptome (use_raw=True), Wilcoxon + BH.
Writes markers_allclusters_top100_RAW.csv used by Figure 3C (manuscript).
Extracted from section 1B of QUICK_FILLERS.py; nothing else changed.
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os, warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

BASE_VEL = DATA_ROOT + "/velocity/03_outputs"
WITHRAW_H5AD = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"
DYN_H5AD = DATA_ROOT + "/velocity_compare_dynamical_3samples/merged_3samples_dynamical_velocity.h5ad"
CR_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"

# ============================================================
# 1B. STAGE 3 CLUSTER MARKERI - RERUN NA RAW
# ============================================================
print("=" * 70)
print("1B. STAGE 3 cluster markeri - RAW rerun")
print("=" * 70)

OUT_TAB_S3 = DATA_ROOT + "/scanpy_out_MDA231/tables_stage3_RAW"
OUT_FIG_S3 = DATA_ROOT + "/scanpy_out_MDA231/figures_stage3_RAW"
os.makedirs(OUT_TAB_S3, exist_ok=True)
os.makedirs(OUT_FIG_S3, exist_ok=True)

ad = sc.read_h5ad(WITHRAW_H5AD)
print(f"Loaded: {ad.shape}, raw: {ad.raw.X.shape}")
print(f"Leiden clusters: {sorted(ad.obs['leiden'].unique(), key=lambda x: int(x))}")

sc.tl.rank_genes_groups(
    ad, groupby="leiden", method="wilcoxon",
    corr_method="benjamini-hochberg", use_raw=True,
    key_added="stage3_RAW_markers",
)

clusters = sorted(ad.obs["leiden"].unique(), key=lambda x: int(x) if str(x).isdigit() else 999)
all_rows = []
for cl in clusters:
    df = sc.get.rank_genes_groups_df(ad, group=cl, key="stage3_RAW_markers")
    df.insert(0, "cluster", cl)
    df_top = df.head(100).copy()
    out_path = os.path.join(OUT_TAB_S3, f"markers_cluster_{cl}_top100_RAW.csv")
    df_top.to_csv(out_path, index=False)
    all_rows.append(df_top)

df_all = pd.concat(all_rows, ignore_index=True)
df_all.to_csv(os.path.join(OUT_TAB_S3, "markers_allclusters_top100_RAW.csv"), index=False)
print(f"SAVED: {OUT_TAB_S3}/")
print(f"  Pre: HVG-only (2000 gena tested)")
print(f"  Sad: RAW (36601 gena tested)")
print(f"  Cluster 0 top 5 PRE (HVG)... vs SAD (RAW):")

# Quick comparison: load old top 100 and new top 100 for cluster 0
try:
    old = pd.read_csv(DATA_ROOT + "/scanpy_out_MDA231/tables_stage3/markers_cluster_0_top100.csv")
    new = df_all[df_all["cluster"] == "0"].head(10)
    print(f"  PRE (HVG):  {old.head(5)['names'].tolist()}")
    print(f"  SAD (RAW):  {new.head(5)['names'].tolist()}")
except Exception as e:
    print(f"  (could not compare: {e})")

# Quick heatmap top 5 per cluster
sc.settings.figdir = OUT_FIG_S3
sc.pl.rank_genes_groups_heatmap(
    ad, key="stage3_RAW_markers", n_genes=5, groupby="leiden",
    show=False, save="_top5_per_cluster_RAW.png", use_raw=True,
)
print(f"SAVED heatmap: {OUT_FIG_S3}")

