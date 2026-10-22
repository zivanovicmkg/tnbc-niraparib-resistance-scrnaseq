import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

IN_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
OUT_DIR = DATA_ROOT + "/velocity/03_outputs/tabular/latent_time_programs"
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(IN_H5AD)

latent = adata.obs["latent_time"].to_numpy()
X = adata.X
if not isinstance(X, np.ndarray):
    X = X.toarray()

results = []
for i, gene in enumerate(adata.var_names):
    expr = X[:, i]
    if np.all(expr == 0):
        continue
    rho, p = spearmanr(expr, latent)
    results.append((gene, rho, p))

df = pd.DataFrame(results, columns=["gene", "spearman_rho", "pval"])
df["abs_rho"] = df["spearman_rho"].abs()

# BH-FDR
df["fdr"] = multipletests(df["pval"].values, method="fdr_bh")[1]

df = df.sort_values(["fdr", "abs_rho"], ascending=[True, False])

out_all = os.path.join(OUT_DIR, "genes_correlated_with_latent_time_ALL_fdr.csv")
df.to_csv(out_all, index=False)

# “strogo” (možeš menjati pragove)
sig = df[(df["fdr"] < 0.01) & (df["abs_rho"] > 0.3)].copy()
out_sig = os.path.join(OUT_DIR, "genes_correlated_with_latent_time_SIG_fdr01_absrho03.csv")
sig.to_csv(out_sig, index=False)

print("SAVED:", out_all)
print("SAVED:", out_sig)
print("Total genes tested:", df.shape[0])
print("Significant genes:", sig.shape[0])

print("\nTOP 10 +rho (lowest fdr):")
print(df[df["spearman_rho"] > 0].head(10)[["gene","spearman_rho","pval","fdr"]])

print("\nTOP 10 -rho (lowest fdr):")
print(df[df["spearman_rho"] < 0].head(10)[["gene","spearman_rho","pval","fdr"]])
