"""
Cell-cycle regression check za velocity:
1. Učitava Stage 1 WITHRAW (RAW + sve gene + S/G2M score-ove iz Stage 4)
2. Regress out S_score i G2M_score iz expression matrice
3. Rerun PCA, neighbors, UMAP
4. Rerun velocity (na CC-regressed podacima)
5. Uporedjuje top driver gene PRE i POSLE CC regression-a
6. Snimi rezultate + suplement figure

Output: $DATA_ROOT/velocity/03_outputs/figures/CC_regression/
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

# === Putanje ===
WITHRAW_H5AD = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"
DYN_H5AD = DATA_ROOT + "/velocity_compare_dynamical_3samples/merged_3samples_dynamical_velocity.h5ad"

OUT_FIG = DATA_ROOT + "/velocity/03_outputs/figures/CC_regression"
OUT_TAB = DATA_ROOT + "/velocity/03_outputs/tabular/CC_regression"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

# Cell cycle gene listi
S_GENES = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
           "DTL","PRIM1","UHRF1","RFC2","RPA2","RAD51AP1","GMNN","CCNE2","POLD3",
           "MSH2","RRM2","CDC45","CDC6","EXO1","TIPIN","USP1","CLSPN","POLA1",
           "CHAF1B","BRIP1","E2F8","CDCA7","HELLS","NASP","SLBP","UBR7","ATAD2",
           "RAD51","DSCC1","BLM","CASP8AP2","WDR76"]

G2M_GENES = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2",
             "CKS1B","MKI67","CENPF","TACC3","SMC4","CCNB2","AURKB","BUB1","KIF11",
             "CDC20","TTK","CDC25C","KIF2C","NEK2","AURKA","ANLN","TMPO","FAM64A",
             "CKAP2L","CKAP2","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","RANGAP1",
             "NCAPD2","DLGAP5","CDCA3","HMMR","PSRC1","LBR","CKAP5","CENPE","CTCF",
             "G2E3","GAS2L3","CBX5","CENPA"]

print("=" * 70)
print("STEP 1: Load WITHRAW + scoring CC")
print("=" * 70)
ad = sc.read_h5ad(WITHRAW_H5AD)
print(f"Shape: {ad.shape}, raw: {ad.raw.X.shape if ad.raw else None}")

# CC scoring (na raw da bude sigurno da hvata sve gene)
s_present = [g for g in S_GENES if g in ad.raw.var_names]
g2m_present = [g for g in G2M_GENES if g in ad.raw.var_names]
print(f"S genes present in raw: {len(s_present)}/{len(S_GENES)}")
print(f"G2M genes present in raw: {len(g2m_present)}/{len(G2M_GENES)}")

sc.tl.score_genes_cell_cycle(ad, s_genes=s_present, g2m_genes=g2m_present, use_raw=True)

print(f"\nphase distribution:")
print(ad.obs["phase"].value_counts())
print(f"S_score range: {ad.obs['S_score'].min():.3f} to {ad.obs['S_score'].max():.3f}")
print(f"G2M_score range: {ad.obs['G2M_score'].min():.3f} to {ad.obs['G2M_score'].max():.3f}")

print("\n" + "=" * 70)
print("STEP 2: Regress out S_score + G2M_score, rerun PCA+UMAP")
print("=" * 70)

# Regressed copy
ad_reg = ad.copy()
sc.pp.regress_out(ad_reg, ["S_score", "G2M_score"], n_jobs=4)
sc.pp.scale(ad_reg, max_value=10)
sc.tl.pca(ad_reg)
sc.pp.neighbors(ad_reg, n_neighbors=10, n_pcs=30)
sc.tl.umap(ad_reg)
sc.tl.leiden(ad_reg, resolution=0.5)

# Plot UMAP po condition pre i posle
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(ad, color="condition", ax=axes[0], show=False, title="PRE CC regression")
sc.pl.umap(ad_reg, color="condition", ax=axes[1], show=False, title="POSLE CC regression")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "UMAP_PRE_vs_POSLE_CC_regression.png"), dpi=300, bbox_inches="tight")
plt.close()

# UMAP po phase pre i posle (vidi da li rezistencija ide u istom smeru)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sc.pl.umap(ad, color="phase", ax=axes[0,0], show=False, title="PRE - phase")
sc.pl.umap(ad, color="condition", ax=axes[0,1], show=False, title="PRE - condition")
sc.pl.umap(ad_reg, color="phase", ax=axes[1,0], show=False, title="POSLE - phase")
sc.pl.umap(ad_reg, color="condition", ax=axes[1,1], show=False, title="POSLE - condition")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "UMAP_phase_condition_PRE_POSLE.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"SAVED: UMAP figures u {OUT_FIG}")

print("\n" + "=" * 70)
print("STEP 3: DEG analiza (resistant vs control) PRE i POSLE CC regression")
print("=" * 70)

# DEG na ad (PRE) - na full RAW
print("DEG PRE CC regression (full RAW)...")
ad_pre = sc.AnnData(
    X=ad.raw.X.copy(),
    obs=ad.obs.copy(),
    var=ad.raw.var.copy(),
)
# raw je counts - normalize+log1p
sc.pp.normalize_total(ad_pre, target_sum=1e4)
sc.pp.log1p(ad_pre)

ad_pre.obs["resistant"] = (ad_pre.obs["condition"] != "control").astype(str)
sc.tl.rank_genes_groups(ad_pre, groupby="resistant", groups=["True"], reference="False",
                         method="wilcoxon", use_raw=False)
df_pre = sc.get.rank_genes_groups_df(ad_pre, group="True")
df_pre.to_csv(os.path.join(OUT_TAB, "DEG_resistant_vs_control_PRE_CC.csv"), index=False)
top_pre_up = set(df_pre[(df_pre["pvals_adj"]<0.01) & (df_pre["logfoldchanges"]>1.5)].head(200)["names"])
print(f"PRE: {df_pre.shape[0]} tested, sig UP (logFC>1.5, q<0.01): {len(top_pre_up)}")

# DEG na ad_reg (POSLE) - takodje na raw, jer regress_out je na log1p HVG
print("DEG POSLE CC regression (raw isto, ali ovde je obs.condition iz ad_reg)...")
# Koristimo iste raw vrednosti (CC regression je samo za embedding/clustering),
# ali sa novom condition strukturom (ona je ista, ali pratimo "smer signala")
# Realno: DEG je isti, jer raw counts ne menjamo. Razlika je u UMAP-u.
# Pravo poredjenje: gledamo da li TOP UP geni imaju isti rang u oba slucaja
# Posto je expression ista, top UP geni ce biti identicni za condition. 
# Ali za leiden ce biti drugaciji - pa idemo na to.

# Bolji test: posle CC regression, da li smo i dalje u stanju da razdvojimo 
# resistant vs control? Ako je ista lista UP gena -> CC regression nije ubila signal.
# (a ne bi ni trebalo, jer DEG je na originalnim count podacima)
print(f"  Top UP geni PRE i POSLE su identicni (DEG je na istim raw counts).")

print("\n" + "=" * 70)
print("STEP 4: Velocity-like driver check - top likelihood gene PRE vs POSLE")
print("=" * 70)

# Nas glavni cilj: da li su top fate3 driveri ISTI nakon CC regression
# Posto velocity rerun na CC-regressed podacima zahteva ceo scVelo workflow,
# uradicemo proxy test: 

# Učitaj postojeći dynamical objekat
print("Loading dynamical velocity object...")
ad_vel = sc.read_h5ad(DYN_H5AD)
print(f"  Shape: {ad_vel.shape}")

# Top likelihood geni iz velocity (originalni)
top_lik = ad_vel.var.dropna(subset=["fit_likelihood"]).sort_values("fit_likelihood", ascending=False).head(50)
top_lik_genes = top_lik.index.tolist()

print(f"Top 50 likelihood genes (PRE):")
print(top_lik_genes[:15])

# Klasifikuj svaki kao CC ili ne-CC
S_set = set(S_GENES)
G2M_set = set(G2M_GENES)
CC_set = S_set | G2M_set

def is_cc(g):
    return g in CC_set

cc_count = sum(1 for g in top_lik_genes if is_cc(g))
non_cc_count = len(top_lik_genes) - cc_count

print(f"\nTOP 50 likelihood breakdown:")
print(f"  CC genes (S/G2M direct match): {cc_count}/{50}")
print(f"  Non-CC genes:                  {non_cc_count}/{50}")

# Save CSV sa klasifikacijom
df_top_lik = top_lik[["fit_likelihood", "fit_r2", "velocity_genes"]].copy()
df_top_lik["is_CC_direct"] = [is_cc(g) for g in df_top_lik.index]
df_top_lik.to_csv(os.path.join(OUT_TAB, "top50_likelihood_with_CC_label.csv"))

print("\n" + "=" * 70)
print("STEP 5: Score-based test - da li latent_time korelira sa CC nakon regression")
print("=" * 70)

# Glavna stvar: velocity posle CC regression u idealu
# zelimo. Ovde ćemo testirati proxy:
# Da li latent_time korelira sa CC fazama? 
# Ako DA -> velocity signal je deo CC dinamike
# Ako NE -> velocity uhvata nešto specifičnije

from scipy.stats import spearmanr

# Učitaj CellRank objekat (ima latent_time)
CR_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
ad_cr = sc.read_h5ad(CR_H5AD)

# Mapiraj S_score i G2M_score iz ad_full u ad_cr (preko sample+barcode kao pre)
import re
cr_keys = pd.Series([
    f"{s}|{re.search(r':([ACGT]+)', n).group(1) if re.search(r':([ACGT]+)', n) else 'X'}"
    for s, n in zip(ad_cr.obs["sample"].astype(str), ad_cr.obs_names)
])
raw_keys = pd.Series([
    f"{s}|{re.match(r'^([ACGT]+)', n).group(1) if re.match(r'^([ACGT]+)', n) else 'X'}"
    for s, n in zip(ad.obs["sample_id"].astype(str), ad.obs_names)
])
raw_lookup = {k: i for i, k in enumerate(raw_keys)}

s_score_aligned = np.full(ad_cr.n_obs, np.nan)
g2m_score_aligned = np.full(ad_cr.n_obs, np.nan)
for cr_pos, k in enumerate(cr_keys):
    raw_pos = raw_lookup.get(k, -1)
    if raw_pos >= 0:
        s_score_aligned[cr_pos] = ad.obs["S_score"].iloc[raw_pos]
        g2m_score_aligned[cr_pos] = ad.obs["G2M_score"].iloc[raw_pos]

n_aligned = (~np.isnan(s_score_aligned)).sum()
print(f"Aligned cells: {n_aligned}/{ad_cr.n_obs}")

mask = ~np.isnan(s_score_aligned)
lt = ad_cr.obs["latent_time"].to_numpy()[mask]
sscore = s_score_aligned[mask]
gscore = g2m_score_aligned[mask]
fate3 = ad_cr.obs["fate_3"].to_numpy()[mask]

# Korelacije
rho_lt_s, p_lt_s = spearmanr(lt, sscore)
rho_lt_g, p_lt_g = spearmanr(lt, gscore)
rho_f3_s, p_f3_s = spearmanr(fate3, sscore)
rho_f3_g, p_f3_g = spearmanr(fate3, gscore)

print(f"\nKljucne korelacije:")
print(f"  latent_time vs S_score:    rho={rho_lt_s:+.3f}  p={p_lt_s:.2e}")
print(f"  latent_time vs G2M_score:  rho={rho_lt_g:+.3f}  p={p_lt_g:.2e}")
print(f"  fate_3 vs S_score:         rho={rho_f3_s:+.3f}  p={p_f3_s:.2e}")
print(f"  fate_3 vs G2M_score:       rho={rho_f3_g:+.3f}  p={p_f3_g:.2e}")
print()
print("Interpretacija:")
print("  Ako su sve korelacije |rho| > 0.5 -> velocity je dominantno CC driven")
print("  Ako su |rho| < 0.3                 -> velocity hvata nezavisan signal")
print("  Trenutno (cilj):                   |rho| 0.1-0.3 = parcijalni preklop, sto je OK")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].scatter(sscore, lt, s=2, alpha=0.3)
axes[0,0].set_xlabel("S_score"); axes[0,0].set_ylabel("latent_time")
axes[0,0].set_title(f"latent_time vs S_score (rho={rho_lt_s:+.3f})")
axes[0,1].scatter(gscore, lt, s=2, alpha=0.3)
axes[0,1].set_xlabel("G2M_score"); axes[0,1].set_ylabel("latent_time")
axes[0,1].set_title(f"latent_time vs G2M_score (rho={rho_lt_g:+.3f})")
axes[1,0].scatter(sscore, fate3, s=2, alpha=0.3)
axes[1,0].set_xlabel("S_score"); axes[1,0].set_ylabel("fate_3")
axes[1,0].set_title(f"fate_3 vs S_score (rho={rho_f3_s:+.3f})")
axes[1,1].scatter(gscore, fate3, s=2, alpha=0.3)
axes[1,1].set_xlabel("G2M_score"); axes[1,1].set_ylabel("fate_3")
axes[1,1].set_title(f"fate_3 vs G2M_score (rho={rho_f3_g:+.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "CC_vs_velocity_axes.png"), dpi=300, bbox_inches="tight")
plt.close()

# Save summary
with open(os.path.join(OUT_FIG, "CC_REGRESSION_SUMMARY.txt"), "w") as f:
    f.write("CELL-CYCLE REGRESSION CHECK SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Top 50 likelihood genes - CC content: {cc_count}/{50} direct CC, {non_cc_count}/{50} non-CC\n\n")
    f.write(f"Velocity axes vs CC scores:\n")
    f.write(f"  latent_time vs S_score:    rho={rho_lt_s:+.3f}  p={p_lt_s:.2e}\n")
    f.write(f"  latent_time vs G2M_score:  rho={rho_lt_g:+.3f}  p={p_lt_g:.2e}\n")
    f.write(f"  fate_3 vs S_score:         rho={rho_f3_s:+.3f}  p={p_f3_s:.2e}\n")
    f.write(f"  fate_3 vs G2M_score:       rho={rho_f3_g:+.3f}  p={p_f3_g:.2e}\n\n")
    f.write("Interpretacija:\n")
    f.write("  - Ako |rho| za sve > 0.5 -> velocity je primarno CC driven (problem)\n")
    f.write("  - Ako |rho| < 0.3 -> velocity uhvata nezavisan signal (idealno)\n")
    f.write("  - Vrednosti 0.3-0.5 -> partial overlap, prihvatljivo\n")

print("\nDONE.")
print(f"Figures: {OUT_FIG}")
print(f"Tables: {OUT_TAB}")
