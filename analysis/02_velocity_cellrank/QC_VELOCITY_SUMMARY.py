"""
Per-sample QC summary za velocity analizu.
Output: tabular/qc_velocity_summary.tsv + figures/qc_velocity_panels.png

Sadrzi:
  - cells per sample
  - median genes/cell, total counts/cell, % MT
  - % spliced, % unspliced, % ambiguous
  - velocity_length distribution
  - velocity_confidence distribution
  - latent_time distribution
  - fate0 i fate3 fractions per sample
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

# Glavni objekat - ima sve sto treba (CellRank + fate + latent_time + velocity layers)
H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"

OUT_TAB = DATA_ROOT + "/velocity/03_outputs/tabular"
OUT_FIG = DATA_ROOT + "/velocity/03_outputs/figures"
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

print(f"Loading: {H5AD}")
ad = sc.read_h5ad(H5AD)
print(f"Shape: {ad.shape}")
print(f"Layers: {list(ad.layers.keys())}")
print(f"obs cols: {list(ad.obs.columns)}\n")

# === Velocity confidence (compute on the fly if not present) ===
if "velocity_confidence" not in ad.obs.columns:
    print("Computing velocity_confidence...")
    scv.tl.velocity_confidence(ad, vkey="velocity")
if "velocity_length" not in ad.obs.columns:
    print("Computing velocity_length...")
    scv.tl.velocity_confidence(ad, vkey="velocity")  # this also computes velocity_length

print(f"obs cols (after): {list(ad.obs.columns)}\n")

# === Per-sample summary ===
SAMPLE_COL = "sample"
samples = sorted(ad.obs[SAMPLE_COL].unique())
print(f"Samples: {samples}\n")

rows = []
for s in samples:
    mask = ad.obs[SAMPLE_COL] == s
    sub = ad[mask].copy()
    n_cells = sub.n_obs

    # gene/count metrics (initial_size_* are scvelo's pre-norm totals)
    median_total = float(np.median(sub.obs["initial_size"])) if "initial_size" in sub.obs.columns else np.nan
    median_unspliced = float(np.median(sub.obs["initial_size_unspliced"])) if "initial_size_unspliced" in sub.obs.columns else np.nan
    median_spliced = float(np.median(sub.obs["initial_size_spliced"])) if "initial_size_spliced" in sub.obs.columns else np.nan

    # spliced/unspliced fraction (from layers)
    try:
        sp = sub.layers["spliced"].sum()
        un = sub.layers["unspliced"].sum()
        ambig = sub.layers["ambiguous"].sum() if "ambiguous" in sub.layers.keys() else 0
        total = sp + un + ambig
        pct_spliced = 100 * sp / total if total > 0 else np.nan
        pct_unspliced = 100 * un / total if total > 0 else np.nan
        pct_ambig = 100 * ambig / total if total > 0 else np.nan
    except Exception as e:
        pct_spliced = pct_unspliced = pct_ambig = np.nan
        print(f"  spliced/unspliced calc failed for {s}: {e}")

    # velocity metrics
    vlen = sub.obs["velocity_length"].dropna() if "velocity_length" in sub.obs.columns else pd.Series([])
    vconf = sub.obs["velocity_confidence"].dropna() if "velocity_confidence" in sub.obs.columns else pd.Series([])

    # latent time
    lt = sub.obs["latent_time"].dropna() if "latent_time" in sub.obs.columns else pd.Series([])

    # fate fractions
    f0 = sub.obs["fate_0"] if "fate_0" in sub.obs.columns else pd.Series([])
    f3 = sub.obs["fate_3"] if "fate_3" in sub.obs.columns else pd.Series([])
    pct_f0_high = 100 * (f0 >= 0.9).sum() / len(f0) if len(f0) > 0 else np.nan
    pct_f3_high = 100 * (f3 >= 0.9).sum() / len(f3) if len(f3) > 0 else np.nan

    rows.append({
        "sample": s,
        "n_cells": n_cells,
        "median_total_counts": round(median_total, 1),
        "median_spliced": round(median_spliced, 1),
        "median_unspliced": round(median_unspliced, 1),
        "pct_spliced": round(pct_spliced, 2),
        "pct_unspliced": round(pct_unspliced, 2),
        "pct_ambiguous": round(pct_ambig, 2),
        "velocity_length_median": round(float(vlen.median()), 4) if len(vlen) > 0 else np.nan,
        "velocity_length_q25": round(float(vlen.quantile(0.25)), 4) if len(vlen) > 0 else np.nan,
        "velocity_length_q75": round(float(vlen.quantile(0.75)), 4) if len(vlen) > 0 else np.nan,
        "velocity_confidence_mean": round(float(vconf.mean()), 4) if len(vconf) > 0 else np.nan,
        "velocity_confidence_median": round(float(vconf.median()), 4) if len(vconf) > 0 else np.nan,
        "latent_time_mean": round(float(lt.mean()), 4) if len(lt) > 0 else np.nan,
        "latent_time_median": round(float(lt.median()), 4) if len(lt) > 0 else np.nan,
        "pct_fate0_high (≥0.9)": round(pct_f0_high, 2),
        "pct_fate3_high (≥0.9)": round(pct_f3_high, 2),
    })

df = pd.DataFrame(rows)
out_tsv = os.path.join(OUT_TAB, "qc_velocity_summary.tsv")
df.to_csv(out_tsv, sep="\t", index=False)

print("=" * 70)
print("PER-SAMPLE QC SUMMARY")
print("=" * 70)
print(df.to_string(index=False))
print(f"\nSAVED: {out_tsv}")

# === Panels: 6 plotova u jednoj figuri ===
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 1. Cells per sample
ax = axes[0, 0]
ax.bar(df["sample"], df["n_cells"])
ax.set_title("Cells per sample")
ax.set_ylabel("n_cells")
for i, v in enumerate(df["n_cells"]):
    ax.text(i, v + 100, str(v), ha="center", fontsize=9)

# 2. % spliced / unspliced / ambig (stacked)
ax = axes[0, 1]
bottom = np.zeros(len(df))
for col, lbl in [("pct_spliced", "spliced"), ("pct_unspliced", "unspliced"), ("pct_ambiguous", "ambig")]:
    vals = df[col].values
    ax.bar(df["sample"], vals, bottom=bottom, label=lbl)
    bottom += vals
ax.set_title("Spliced / unspliced / ambiguous (%)")
ax.set_ylabel("%")
ax.legend(fontsize=8)

# 3. velocity_length distribution (violin)
ax = axes[0, 2]
data = [ad.obs.loc[ad.obs[SAMPLE_COL] == s, "velocity_length"].dropna().values for s in samples]
parts = ax.violinplot(data, showmedians=True)
ax.set_xticks(range(1, len(samples) + 1))
ax.set_xticklabels(samples)
ax.set_title("Velocity length")
ax.set_ylabel("velocity_length")

# 4. velocity_confidence distribution
ax = axes[1, 0]
data = [ad.obs.loc[ad.obs[SAMPLE_COL] == s, "velocity_confidence"].dropna().values for s in samples]
parts = ax.violinplot(data, showmedians=True)
ax.set_xticks(range(1, len(samples) + 1))
ax.set_xticklabels(samples)
ax.set_title("Velocity confidence")
ax.set_ylabel("velocity_confidence")

# 5. latent_time per sample
ax = axes[1, 1]
data = [ad.obs.loc[ad.obs[SAMPLE_COL] == s, "latent_time"].dropna().values for s in samples]
parts = ax.violinplot(data, showmedians=True)
ax.set_xticks(range(1, len(samples) + 1))
ax.set_xticklabels(samples)
ax.set_title("Latent time")
ax.set_ylabel("latent_time")

# 6. Fate fractions
ax = axes[1, 2]
x = np.arange(len(df))
w = 0.35
ax.bar(x - w/2, df["pct_fate0_high (≥0.9)"], w, label="fate_0 ≥0.9")
ax.bar(x + w/2, df["pct_fate3_high (≥0.9)"], w, label="fate_3 ≥0.9")
ax.set_xticks(x)
ax.set_xticklabels(df["sample"])
ax.set_title("Cells with fate ≥0.9 (%)")
ax.set_ylabel("%")
ax.legend(fontsize=9)

plt.tight_layout()
out_png = os.path.join(OUT_FIG, "qc_velocity_panels.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"SAVED: {out_png}")

print("\nDONE.")
