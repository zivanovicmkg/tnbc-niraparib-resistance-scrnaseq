import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import numpy as np
import pandas as pd
import scanpy as sc

IN_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
CSV_ALL = DATA_ROOT + "/velocity/03_outputs/tabular/latent_time_programs/genes_correlated_with_latent_time_ALL_fdr.csv"

OUT_DIR = DATA_ROOT + "/velocity/03_outputs/figures/latent_time_validation"
os.makedirs(OUT_DIR, exist_ok=True)

# koliko gena uzimamo iz top lista
TOPN = 10

adata = sc.read_h5ad(IN_H5AD)

df = pd.read_csv(CSV_ALL)
pos = df[df["spearman_rho"] > 0].head(TOPN)["gene"].tolist()
neg = df[df["spearman_rho"] < 0].head(TOPN)["gene"].tolist()

# zadrži samo gene koji su stvarno u adata
pos = [g for g in pos if g in adata.var_names]
neg = [g for g in neg if g in adata.var_names]

print("TOP + genes used:", pos)
print("TOP - genes used:", neg)

# napravi score-ove u .obs
sc.tl.score_genes(adata, gene_list=pos, score_name="LateScore_posTop")
sc.tl.score_genes(adata, gene_list=neg, score_name="EarlyScore_negTop")

# snimi mali tabelarni summary za latent-time binove
tmp = adata.obs[["latent_time","LateScore_posTop","EarlyScore_negTop"]].copy()
tmp["lt_bin"] = pd.qcut(tmp["latent_time"], q=10, labels=False, duplicates="drop")
bin_means = tmp.groupby("lt_bin")[["LateScore_posTop","EarlyScore_negTop"]].mean()
out_csv = os.path.join(OUT_DIR, "meta_scores_by_latent_time_bins.csv")
bin_means.to_csv(out_csv)
print("SAVED:", out_csv)

# plotovi (matplotlib, bez seaborn da bude stabilno)
import matplotlib.pyplot as plt

# 1) UMAP LateScore
sc.pl.umap(
    adata,
    color=["LateScore_posTop"],
    show=False
)
plt.savefig(os.path.join(OUT_DIR, "UMAP_LateScore_posTop.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2) UMAP EarlyScore
sc.pl.umap(
    adata,
    color=["EarlyScore_negTop"],
    show=False
)
plt.savefig(os.path.join(OUT_DIR, "UMAP_EarlyScore_negTop.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3) Late vs Early scatter
plt.figure(figsize=(5,4))
plt.scatter(tmp["EarlyScore_negTop"], tmp["LateScore_posTop"], s=2, alpha=0.3)
plt.xlabel("EarlyScore (top -rho genes)")
plt.ylabel("LateScore (top +rho genes)")
plt.title("Early vs Late meta-scores")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_Early_vs_Late_scores.png"), dpi=300)
plt.close()

# 4) trend duž latent time binova
b = bin_means.reset_index()
plt.figure(figsize=(5,4))
plt.plot(b["lt_bin"], b["EarlyScore_negTop"], marker="o", label="EarlyScore (-rho)")
plt.plot(b["lt_bin"], b["LateScore_posTop"], marker="o", label="LateScore (+rho)")
plt.xlabel("Latent time (binned)")
plt.ylabel("Mean meta-score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "meta_scores_trend_by_latent_time.png"), dpi=300)
plt.close()

print("DONE. Figures saved in:", OUT_DIR)
