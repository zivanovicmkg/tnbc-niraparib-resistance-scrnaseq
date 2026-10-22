"""
Parameter sweep v2 - PRAVI sweep koji počinje od loom fajlova (sirovi spliced/unspliced).

Razlog v2: prethodni run je počeo od h5ad koji je već prošao filter_and_normalize,
pa scvelo je samo overrider-ovao filter ('Skip filtering by dispersion since...').
Sada idemo od sirovih loom-ova, gde svaka varijanta zaista dobija svoj n_top_genes filter.

7 kombinacija:
  Reference: n_neighbors=10, n_pcs=30, n_top_genes=2000  (originalno)
  Variant 1: n_neighbors=15, n_pcs=30, n_top_genes=2000
  Variant 2: n_neighbors=30, n_pcs=30, n_top_genes=2000
  Variant 3: n_neighbors=10, n_pcs=20, n_top_genes=2000
  Variant 4: n_neighbors=10, n_pcs=50, n_top_genes=2000
  Variant 5: n_neighbors=10, n_pcs=30, n_top_genes=1500
  Variant 6: n_neighbors=10, n_pcs=30, n_top_genes=3000

Vreme: ~10-15 min po varijanti × 7 = ~1-2h.
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import anndata as ad_lib
import matplotlib.pyplot as plt

OUT_FIG = DATA_ROOT + "/velocity/03_outputs/figures/parameter_sweep_v2"
OUT_TAB = DATA_ROOT + "/velocity/03_outputs/tabular/parameter_sweep_v2"
OUT_OBJ = DATA_ROOT + "/velocity/03_outputs/objects/parameter_sweep_v2"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_OBJ, exist_ok=True)

# Loom fajlovi (sirovi spliced/unspliced)
LOOMS = {
    "C231": DATA_ROOT + "/velocity/01_inputs/looms/C231.loom",
    "N231": DATA_ROOT + "/velocity/01_inputs/looms/N231.loom",
    "N436": DATA_ROOT + "/velocity/01_inputs/looms/N436.loom",
}

# Učitaj loom-ove + concatenate
print("=" * 70)
print("LOADING LOOM FILES")
print("=" * 70)

ad_list = []
for sample, path in LOOMS.items():
    print(f"  Loading {sample}: {path}")
    a = ad_lib.read_loom(path)
    a.var_names_make_unique()
    a.obs["sample"] = sample
    ad_list.append(a)
    print(f"    {sample} shape: {a.shape}, layers: {list(a.layers.keys())}")

print("\nConcatenating...")
ad_merged = ad_list[0].concatenate(ad_list[1:], batch_key="sample_batch", batch_categories=list(LOOMS.keys()))
print(f"Merged shape: {ad_merged.shape}")
print(f"Layers: {list(ad_merged.layers.keys())}")
print(f"Sample distribution:")
print(ad_merged.obs["sample"].value_counts())

# Filter to keep only cells that were in original analysis (17,953 cells)
# Provera: koliko obs imamo
n_obs_loom = ad_merged.n_obs
print(f"\nLoom total cells: {n_obs_loom}")

# Učitaj original h5ad da znamo koje ćelije su bile u analizi
ORIG_H5AD = DATA_ROOT + "/velocity_compare_dynamical_3samples/merged_3samples_dynamical_velocity.h5ad"
ad_orig = sc.read_h5ad(ORIG_H5AD)
orig_n = ad_orig.n_obs
print(f"Original analysis had: {orig_n} cells")

# Maping ćelija od loom -> orig (preko sample + barcode)
import re

def loom_barcode(name):
    """Extract barcode from loom obs_names: 'C231_cellsorted_CB_coord_KSBFU:AACCAACTCGTACACAx'"""
    m = re.search(r":([ACGT]+)x?$", name)
    return m.group(1) if m else None

def orig_barcode(name):
    """Original obs_names format - check first"""
    # if format is 'C231_cellsorted_CB_coord_KSBFU:AACCAACTCGTACACAx' it is same
    m = re.search(r":([ACGT]+)x?$", name)
    if m:
        return m.group(1)
    # else try matching from start
    m = re.match(r"^([ACGT]+)", name)
    return m.group(1) if m else None

print(f"\nOriginal obs_names primer: {ad_orig.obs_names[:3].tolist()}")
print(f"Loom obs_names primer: {ad_merged.obs_names[:3].tolist()}")

# Build orig keys
orig_keys = set()
for s, n in zip(ad_orig.obs["sample"].astype(str), ad_orig.obs_names):
    bc = orig_barcode(n)
    if bc:
        orig_keys.add(f"{s}|{bc}")

# Check loom keys
loom_keys = []
for s, n in zip(ad_merged.obs["sample"].astype(str), ad_merged.obs_names):
    bc = loom_barcode(n)
    if bc:
        loom_keys.append(f"{s}|{bc}")
    else:
        loom_keys.append(None)

# Filter merged to only cells in original
mask = np.array([k in orig_keys if k else False for k in loom_keys])
print(f"\nMatched cells (loom → orig): {mask.sum()} / {n_obs_loom}")

if mask.sum() < 0.9 * orig_n:
    print(f"  WARNING: only {mask.sum()/orig_n:.1%} of original cells matched.")
    print(f"  Loom obs format moze biti drugaciji. Probacemo bez filtriranja.")
    ad_input = ad_merged.copy()
else:
    ad_input = ad_merged[mask].copy()
    print(f"  Using filtered: {ad_input.shape}")

# Save the input
ad_input.write(os.path.join(OUT_OBJ, "_INPUT_merged_loom.h5ad"))
print(f"\nSAVED input: {os.path.join(OUT_OBJ, '_INPUT_merged_loom.h5ad')}")

# 7 parameter setova
SWEEP = [
    {"label": "ref_n10_pcs30_g2000",  "n_neighbors": 10, "n_pcs": 30, "n_top_genes": 2000},
    {"label": "v1_n15_pcs30_g2000",   "n_neighbors": 15, "n_pcs": 30, "n_top_genes": 2000},
    {"label": "v2_n30_pcs30_g2000",   "n_neighbors": 30, "n_pcs": 30, "n_top_genes": 2000},
    {"label": "v3_n10_pcs20_g2000",   "n_neighbors": 10, "n_pcs": 20, "n_top_genes": 2000},
    {"label": "v4_n10_pcs50_g2000",   "n_neighbors": 10, "n_pcs": 50, "n_top_genes": 2000},
    {"label": "v5_n10_pcs30_g1500",   "n_neighbors": 10, "n_pcs": 30, "n_top_genes": 1500},
    {"label": "v6_n10_pcs30_g3000",   "n_neighbors": 10, "n_pcs": 30, "n_top_genes": 3000},
]

print("\n" + "=" * 70)
print("PARAMETER SWEEP")
print("=" * 70)

def run_pipeline(adata_in, n_neighbors, n_pcs, n_top_genes, label):
    print(f"\n--- {label} ---")
    t0 = time.time()
    
    ad = adata_in.copy()
    
    # Sirovi pipeline
    scv.pp.filter_and_normalize(ad, min_shared_counts=20, n_top_genes=n_top_genes)
    print(f"  filter_and_normalize: {ad.shape}")
    
    scv.pp.moments(ad, n_pcs=n_pcs, n_neighbors=n_neighbors)
    print(f"  moments: n_pcs={n_pcs}, n_neighbors={n_neighbors}")
    
    print(f"  recover_dynamics (sporo)...")
    scv.tl.recover_dynamics(ad, n_jobs=8)
    
    scv.tl.velocity(ad, mode="dynamical")
    scv.tl.velocity_graph(ad)
    scv.tl.velocity_confidence(ad, vkey="velocity")
    
    try:
        scv.tl.latent_time(ad)
    except Exception as e:
        print(f"  latent_time fail: {e}")
    
    elapsed = (time.time() - t0) / 60
    print(f"  DONE in {elapsed:.1f} min")
    return ad

# Run all
results = {}
top_lik_per_variant = {}
top_velgenes_per_variant = {}

for v in SWEEP:
    label = v["label"]
    out_path = os.path.join(OUT_OBJ, f"{label}.h5ad")
    
    if os.path.exists(out_path):
        print(f"\n=== {label}: exists, loading ===")
        ad_v = sc.read_h5ad(out_path)
    else:
        try:
            ad_v = run_pipeline(
                ad_input, v["n_neighbors"], v["n_pcs"], v["n_top_genes"], label
            )
            ad_v.write(out_path)
        except Exception as e:
            print(f"  ERROR u {label}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Top likelihood genes
    if "fit_likelihood" in ad_v.var.columns:
        df_l = ad_v.var.dropna(subset=["fit_likelihood"]).copy()
        if "velocity_genes" in df_l.columns:
            df_l = df_l[df_l["velocity_genes"] == True].copy()
        df_l = df_l.sort_values("fit_likelihood", ascending=False)
        top50_lik = df_l.index[:50].tolist()
    else:
        top50_lik = []
    
    # All velocity genes (for second comparison)
    if "velocity_genes" in ad_v.var.columns:
        velgenes = set(ad_v.var.index[ad_v.var["velocity_genes"] == True].tolist())
    else:
        velgenes = set()
    
    top_lik_per_variant[label] = set(top50_lik)
    top_velgenes_per_variant[label] = velgenes
    
    results[label] = {
        "n_cells": ad_v.n_obs,
        "n_genes_after_filter": ad_v.n_vars,
        "n_velocity_genes": len(velgenes),
        "n_likelihood_passed": int(ad_v.var.dropna(subset=["fit_likelihood"]).shape[0])
            if "fit_likelihood" in ad_v.var.columns else None,
        "vel_confidence_median": float(ad_v.obs["velocity_confidence"].median()),
        "vel_confidence_mean": float(ad_v.obs["velocity_confidence"].mean()),
        "vel_length_median": float(ad_v.obs["velocity_length"].median()),
        "latent_time_mean": float(ad_v.obs["latent_time"].mean()) if "latent_time" in ad_v.obs.columns else None,
        "latent_time_median": float(ad_v.obs["latent_time"].median()) if "latent_time" in ad_v.obs.columns else None,
        "top10_likelihood": ", ".join(top50_lik[:10]),
    }

# ============================================================
# Stability analiza
# ============================================================
print("\n" + "=" * 70)
print("STABILITY ANALYSIS")
print("=" * 70)

df_metrics = pd.DataFrame(results).T
df_metrics.to_csv(os.path.join(OUT_TAB, "velocity_metrics_per_param.csv"))
print("\nMetrike po parameter setu:")
print(df_metrics[["n_cells", "n_genes_after_filter", "n_velocity_genes",
                  "vel_confidence_median", "vel_length_median", "latent_time_mean"]].to_string())

# Jaccard top 50 likelihood
labels = list(top_lik_per_variant.keys())
n = len(labels)
jacc_lik = np.zeros((n, n))
for i, la in enumerate(labels):
    for j, lb in enumerate(labels):
        a = top_lik_per_variant[la]
        b = top_lik_per_variant[lb]
        if not a or not b:
            jacc_lik[i, j] = np.nan
        else:
            jacc_lik[i, j] = len(a & b) / len(a | b)

df_jacc_lik = pd.DataFrame(jacc_lik, index=labels, columns=labels)
df_jacc_lik.to_csv(os.path.join(OUT_TAB, "top50_likelihood_jaccard.csv"))
print("\nJaccard - Top 50 likelihood genes:")
print(df_jacc_lik.round(2).to_string())

# Jaccard velocity_genes (svi)
jacc_vel = np.zeros((n, n))
for i, la in enumerate(labels):
    for j, lb in enumerate(labels):
        a = top_velgenes_per_variant[la]
        b = top_velgenes_per_variant[lb]
        if not a or not b:
            jacc_vel[i, j] = np.nan
        else:
            jacc_vel[i, j] = len(a & b) / len(a | b)

df_jacc_vel = pd.DataFrame(jacc_vel, index=labels, columns=labels)
df_jacc_vel.to_csv(os.path.join(OUT_TAB, "all_velocity_genes_jaccard.csv"))
print("\nJaccard - svi velocity_genes:")
print(df_jacc_vel.round(2).to_string())

# Heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for idx, (data, title) in enumerate([
    (jacc_lik, "Top 50 likelihood genes"),
    (jacc_vel, "All velocity_genes"),
]):
    ax = axes[idx]
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(data[i, j]):
                color = "white" if data[i, j] > 0.5 else "black"
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label="Jaccard")
    ax.set_title(title)
plt.suptitle("Parameter sweep stability (v2)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "stability_heatmaps.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# Reference stability
ref_label = "ref_n10_pcs30_g2000"
if ref_label in labels:
    ref_idx = labels.index(ref_label)
    ref_to_others_lik = [jacc_lik[ref_idx, i] for i in range(n) if i != ref_idx]
    ref_to_others_vel = [jacc_vel[ref_idx, i] for i in range(n) if i != ref_idx]
    mean_jacc_lik = np.nanmean(ref_to_others_lik)
    mean_jacc_vel = np.nanmean(ref_to_others_vel)
    
    print(f"\n{'='*70}")
    print(f"REFERENCE STABILITY (vs '{ref_label}')")
    print(f"{'='*70}")
    print(f"  Top 50 likelihood mean Jaccard:  {mean_jacc_lik:.3f}")
    print(f"  All velocity_genes mean Jaccard: {mean_jacc_vel:.3f}")
    
    if mean_jacc_lik > 0.5 and mean_jacc_vel > 0.5:
        print(f"\n  ✅ STABILNO: top driveri se >50% preklapaju kroz parametre.")
    elif mean_jacc_lik > 0.3 or mean_jacc_vel > 0.3:
        print(f"\n  🟡 PARCIJALNO STABILNO: 30-50% preklapanje.")
    else:
        print(f"\n  ⚠ NESTABILNO.")

print("\nDONE.")
