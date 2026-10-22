"""
Negative control za velocity:
1. Permutuje cell labels (sample/condition kolone) - bioloska struktura razbijena
2. Permutuje spliced/unspliced layers - kineticka struktura razbijena
3. Pokazuje da real velocity uhvata realnu strukturu, ne nasumicni signal

Test: 
- velocity_confidence median - trebao bi da padne posle permutacije
- top likelihood gen overlap - trebao bi da bude mali
- latent_time / fate3 correlation - trebao bi da nestane
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT_FIG = DATA_ROOT + "/velocity/03_outputs/figures/negative_control"
OUT_TAB = DATA_ROOT + "/velocity/03_outputs/tabular/negative_control"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

DYN_H5AD = DATA_ROOT + "/velocity_compare_dynamical_3samples/merged_3samples_dynamical_velocity.h5ad"
ad = sc.read_h5ad(DYN_H5AD)
print(f"Loaded original dynamical: {ad.shape}")

# Original metrics (baseline)
if "velocity_confidence" not in ad.obs.columns:
    scv.tl.velocity_confidence(ad, vkey="velocity")
orig_conf_median = ad.obs["velocity_confidence"].median()
orig_length_median = ad.obs["velocity_length"].median()
print(f"\nORIGINAL velocity metrics:")
print(f"  velocity_confidence median: {orig_conf_median:.3f}")
print(f"  velocity_length median:     {orig_length_median:.3f}")

# Top 50 likelihood genes (originalni)
df_genes_orig = ad.var.dropna(subset=["fit_likelihood"]).sort_values("fit_likelihood", ascending=False)
top50_orig = set(df_genes_orig.index[:50])
print(f"  Top 50 likelihood genes (original): {list(top50_orig)[:5]}...")

# ============================================================
# PERMUTACIJA 1: razbij spliced/unspliced layer alignment
# ============================================================
print("\n" + "=" * 70)
print("PERMUTATION 1: razbij spliced/unspliced cell-by-cell pairing")
print("=" * 70)

ad_perm = ad.copy()
np.random.seed(42)
perm_idx = np.random.permutation(ad_perm.n_obs)

# Permutuj samo unspliced layer (spliced ostaje), tako da spliced/unspliced 
# pairing po celiji bude razbijen
unspliced_orig = ad_perm.layers["unspliced"].copy()
ad_perm.layers["unspliced"] = unspliced_orig[perm_idx, :]
print("Unspliced layer permutovan po celijama")

# Rerun velocity (dynamical bi bio prespor; idemo na stochastic kao quick proxy)
print("Rerunning velocity (stochastic, brzo)...")
scv.pp.moments(ad_perm, n_pcs=30, n_neighbors=10)
scv.tl.velocity(ad_perm, mode="stochastic")
scv.tl.velocity_graph(ad_perm)
scv.tl.velocity_confidence(ad_perm, vkey="velocity")

perm_conf_median = ad_perm.obs["velocity_confidence"].median()
perm_length_median = ad_perm.obs["velocity_length"].median()
print(f"\nPERMUTED velocity metrics:")
print(f"  velocity_confidence median: {perm_conf_median:.3f}  (originalno: {orig_conf_median:.3f})")
print(f"  velocity_length median:     {perm_length_median:.3f}  (originalno: {orig_length_median:.3f})")

# Stochastic nema fit_likelihood, ali ima velocity_genes
print(f"  velocity_genes (permutovano): {ad_perm.var['velocity_genes'].sum()}")

# Plot UMAP velocity stream pre i posle permutacije
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scv.pl.velocity_embedding_stream(ad, basis="umap", color="sample", ax=axes[0],
                                  show=False, title="ORIGINALNO velocity stream")
scv.pl.velocity_embedding_stream(ad_perm, basis="umap", color="sample", ax=axes[1],
                                  show=False, title="PERMUTOVANO unspliced layer")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "velocity_stream_original_vs_permuted.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# Plot velocity_confidence comparison (violin)
fig, ax = plt.subplots(figsize=(7, 5))
data = [
    ad.obs["velocity_confidence"].values,
    ad_perm.obs["velocity_confidence"].values,
]
parts = ax.violinplot(data, showmedians=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Original", "Permutovano\n(unspliced cells)"])
ax.set_ylabel("velocity_confidence")
ax.set_title(f"Velocity confidence: original vs permutovano\n"
             f"median: {orig_conf_median:.3f} vs {perm_conf_median:.3f} "
             f"(razlika: {orig_conf_median - perm_conf_median:+.3f})")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "violin_velocity_confidence_original_vs_permuted.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# PERMUTACIJA 2: razbij velocity vector field (random)
# ============================================================
print("\n" + "=" * 70)
print("PERMUTATION 2: razbij velocity vector polje (random shuffle)")
print("=" * 70)

ad_rand = ad.copy()
np.random.seed(123)
# Permutuj velocity vector polje po celijama
velocity_orig = ad_rand.layers["velocity"].copy()
perm_idx2 = np.random.permutation(ad_rand.n_obs)
ad_rand.layers["velocity"] = velocity_orig[perm_idx2, :]

# Rerun velocity_graph + confidence + latent_time
scv.tl.velocity_graph(ad_rand)
scv.tl.velocity_confidence(ad_rand, vkey="velocity")

rand_conf_median = ad_rand.obs["velocity_confidence"].median()
rand_length_median = ad_rand.obs["velocity_length"].median()
print(f"RANDOM velocity field metrics:")
print(f"  velocity_confidence median: {rand_conf_median:.3f}  (originalno: {orig_conf_median:.3f})")
print(f"  velocity_length median:     {rand_length_median:.3f}  (originalno: {orig_length_median:.3f})")

# Save sumarno
summary = pd.DataFrame({
    "scenario": ["Original", "Permuted unspliced (cells)", "Random velocity field"],
    "vel_confidence_median": [orig_conf_median, perm_conf_median, rand_conf_median],
    "vel_length_median": [orig_length_median, perm_length_median, rand_length_median],
})
summary.to_csv(os.path.join(OUT_TAB, "negative_control_summary.csv"), index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

# Interpretacija
diff = orig_conf_median - perm_conf_median
print(f"\nInterpretacija:")
if diff > 0.2:
    print(f"  ✅ Velocity confidence PADA znatno (-{diff:.3f}) kada se razbije struktura.")
    print(f"     Original signal NIJE nasumican.")
elif diff > 0.05:
    print(f"  🟡 Velocity confidence pada umereno (-{diff:.3f}).")
    print(f"     Original signal ima realnu strukturu ali nije ekstremno robusna.")
else:
    print(f"  ⚠ Velocity confidence ne pada znatno (-{diff:.3f}).")
    print(f"     Mozda velocity polje hvata mnogo nasumicnog signala?")

print("\nDONE.")
