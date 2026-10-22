"""
Prosiruje 3 ekspresione analize sa 2,000 HVG na 36,601 RAW gena:
1. DEG fate_3 UP/DOWN (Wilcoxon rank-sum)
2. DEG fate_3 high vs low (LT-controlled)
3. Spearman korelacija svih gena sa latent_time

V2 mapping:
- CellRank obs:  'C231_cellsorted_CB_coord_KSBFU:AACCAACTCGTACACAx'  -> sample=C231, bc=AACCAACTCGTACACA
- WITHRAW obs:   'AAACCCACAGCATCTA-1-C231'                          -> sample=C231, bc=AAACCCACAGCATCTA
- Mapping kljuc: '{sample}|{barcode}'
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

CELLRANK_H5AD = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca__PLUS_nmf4_bins.h5ad"
WITHRAW_H5AD = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"

OUT_DIR = DATA_ROOT + "/velocity/03_outputs/tabular/RAW_expanded"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. UCITAVANJE
# ============================================================
print("Loading CellRank object (HVG, sa fate kolonama)...")
ad_cr = sc.read_h5ad(CELLRANK_H5AD)
print(f"  CellRank shape: {ad_cr.shape}")

print("Loading WITHRAW object (full RAW = 36,601 gen)...")
ad_raw = sc.read_h5ad(WITHRAW_H5AD)
print(f"  WITHRAW shape: {ad_raw.shape}, raw shape: {ad_raw.raw.X.shape}")

# ============================================================
# 2. CISTI BARCODE EKSTRAKT iz oba objekta
# ============================================================
def extract_cr_barcode(name):
    """C231_cellsorted_CB_coord_KSBFU:AACCAACTCGTACACAx -> AACCAACTCGTACACA"""
    m = re.search(r":([ACGT]+)", name)
    return m.group(1) if m else None

def extract_raw_barcode(name):
    """AAACCCACAGCATCTA-1-C231 -> AAACCCACAGCATCTA"""
    m = re.match(r"^([ACGT]+)", name)
    return m.group(1) if m else None

cr_keys = pd.Series([
    f"{s}|{extract_cr_barcode(n)}"
    for s, n in zip(ad_cr.obs["sample"].astype(str), ad_cr.obs_names)
])

raw_keys = pd.Series([
    f"{s}|{extract_raw_barcode(n)}"
    for s, n in zip(ad_raw.obs["sample_id"].astype(str), ad_raw.obs_names)
])

# Mapa raw_key -> raw_pos
raw_lookup = {k: i for i, k in enumerate(raw_keys)}

# Za svaku CellRank celiju, pronadji poziciju u WITHRAW
matches = [raw_lookup.get(k, -1) for k in cr_keys]
n_matched = sum(1 for m in matches if m >= 0)
print(f"\nMapping: matched {n_matched} / {len(cr_keys)} ćelija")

if n_matched < 0.95 * len(cr_keys):
    print("ERROR: vise od 5% celija ne mapira. Stop.")
    exit(1)

# ============================================================
# 3. PREBACI fate kolone iz CellRank u WITHRAW
# ============================================================
cols_to_transfer = ["fate_0", "fate_3", "latent_time", "lt_nmf4", "lt_nmf2", "nmf4_bin"]
print(f"\nTransfer columns: {cols_to_transfer}")

# Inicijalizuj nove kolone u ad_raw
for col in cols_to_transfer:
    if col not in ad_cr.obs.columns:
        continue
    if col == "nmf4_bin":
        new_col = pd.Series([None] * ad_raw.n_obs, index=ad_raw.obs_names, dtype=object)
    else:
        new_col = pd.Series([np.nan] * ad_raw.n_obs, index=ad_raw.obs_names, dtype=float)
    
    cr_vals = ad_cr.obs[col].values
    for cr_pos, raw_pos in enumerate(matches):
        if raw_pos >= 0:
            new_col.iloc[raw_pos] = cr_vals[cr_pos]
    
    ad_raw.obs[col] = new_col.values
    n_filled = ad_raw.obs[col].notna().sum()
    print(f"  {col}: {n_filled} non-null")

# ============================================================
# 4. PREBACI NA RAW MATRICU + log1p
# ============================================================
print("\n=== Switching to RAW matrix for downstream ===")

ad_full = sc.AnnData(
    X=ad_raw.raw.X.copy(),
    obs=ad_raw.obs.copy(),
    var=ad_raw.raw.var.copy(),
)
print(f"Full RAW adata: {ad_full.shape}")

# Drop celije koje nisu mapirane (bez fate)
mask_complete = ad_full.obs["fate_3"].notna()
ad_full = ad_full[mask_complete].copy()
print(f"After dropping unmapped: {ad_full.shape}")

# Ovo su raw counts -> normalize + log1p
print("Normalizing + log1p (raw counts)...")
sc.pp.normalize_total(ad_full, target_sum=1e4)
sc.pp.log1p(ad_full)
print(f"  X range: {float(ad_full.X.min()):.3f} - {float(ad_full.X.max()):.3f}")

# ============================================================
# A. DEG fate_3 vs ostatak — RAW, 36,601 gen
# ============================================================
print("\n=== A. DEG fate_3 vs other (RAW, 36,601 gen) ===")
ad_full.obs["fate3_group"] = (ad_full.obs["fate_3"] >= 0.9).map({True: "fate3_high", False: "other"})
ad_full.obs["fate3_group"] = ad_full.obs["fate3_group"].astype("category")
print(ad_full.obs["fate3_group"].value_counts())

sc.tl.rank_genes_groups(
    ad_full, groupby="fate3_group", groups=["fate3_high"], reference="other",
    method="wilcoxon", use_raw=False
)
df_fate3 = sc.get.rank_genes_groups_df(ad_full, group="fate3_high")
out = os.path.join(OUT_DIR, "DEG_fate3_high_vs_other__RAW_wilcoxon.csv")
df_fate3.to_csv(out, index=False)
n_total = df_fate3.shape[0]
n_sig = (df_fate3["pvals_adj"] < 0.05).sum()
n_sig_up = ((df_fate3["pvals_adj"] < 0.05) & (df_fate3["logfoldchanges"] > 1)).sum()
n_sig_dn = ((df_fate3["pvals_adj"] < 0.05) & (df_fate3["logfoldchanges"] < -1)).sum()
print(f"SAVED: {out}")
print(f"  total tested: {n_total}, sig (q<0.05): {n_sig}")
print(f"  sig UP   (logFC>1): {n_sig_up}")
print(f"  sig DOWN (logFC<-1): {n_sig_dn}")
print(f"\nTOP 15 UP fate3_high:")
top_up = df_fate3.nlargest(15, "scores")[["names", "scores", "logfoldchanges", "pvals_adj"]]
print(top_up.to_string(index=False))

# ============================================================
# B. DEG fate3_high vs low (LT-controlled)
# ============================================================
print("\n=== B. DEG fate3 high vs low (RAW, LT-controlled) ===")
ad_f3 = ad_full[ad_full.obs["fate_3"] >= 0.9].copy()
ad_f3 = ad_f3[ad_f3.obs["nmf4_bin"].notna()].copy()
print(f"Fate3 cells: {ad_f3.n_obs}")
print(ad_f3.obs["nmf4_bin"].value_counts())

low_lt = ad_f3.obs.loc[ad_f3.obs["nmf4_bin"] == "low", "latent_time"].astype(float)
high_lt = ad_f3.obs.loc[ad_f3.obs["nmf4_bin"] == "high", "latent_time"].astype(float)
lo1, hi1 = np.quantile(low_lt, [0.10, 0.90])
lo2, hi2 = np.quantile(high_lt, [0.10, 0.90])
lt_lo, lt_hi = max(lo1, lo2), min(hi1, hi2)
print(f"LT overlap window: [{lt_lo:.3f}, {lt_hi:.3f}]")

mask_lt = (ad_f3.obs["latent_time"].astype(float) >= lt_lo) & (ad_f3.obs["latent_time"].astype(float) <= lt_hi)
ad_sub = ad_f3[mask_lt].copy()
ad_sub = ad_sub[ad_sub.obs["nmf4_bin"].isin(["low", "high"])].copy()
print(f"Cells in overlap (low+high only): {ad_sub.n_obs}")
print(ad_sub.obs["nmf4_bin"].value_counts())

ad_sub.obs["nmf4_bin"] = ad_sub.obs["nmf4_bin"].astype("category")
sc.tl.rank_genes_groups(
    ad_sub, groupby="nmf4_bin", groups=["high"], reference="low",
    method="wilcoxon", use_raw=False
)
df_hl = sc.get.rank_genes_groups_df(ad_sub, group="high")
out = os.path.join(OUT_DIR, "DE_fate3_high_vs_low__RAW_LTcontrolled.csv")
df_hl.to_csv(out, index=False)
n_total = df_hl.shape[0]
n_sig = (df_hl["pvals_adj"] < 0.05).sum()
n_sig_up = ((df_hl["pvals_adj"] < 0.05) & (df_hl["logfoldchanges"] > 0.5)).sum()
n_sig_dn = ((df_hl["pvals_adj"] < 0.05) & (df_hl["logfoldchanges"] < -0.5)).sum()
print(f"SAVED: {out}")
print(f"  total: {n_total}, sig (q<0.05): {n_sig}")
print(f"  sig UP   (logFC>0.5): {n_sig_up}")
print(f"  sig DOWN (logFC<-0.5): {n_sig_dn}")
print(f"\nTOP 15 UP (high vs low):")
top_up_hl = df_hl.nlargest(15, "scores")[["names", "scores", "logfoldchanges", "pvals_adj"]]
print(top_up_hl.to_string(index=False))

# ============================================================
# C. Spearman korelacija svih gena sa latent_time
# ============================================================
print("\n=== C. Spearman korelacija (RAW, 36,601 gen) ===")
ad_lt = ad_full[ad_full.obs["latent_time"].notna()].copy()
print(f"Cells: {ad_lt.n_obs}")
latent = ad_lt.obs["latent_time"].astype(float).to_numpy()
X = ad_lt.X
if not isinstance(X, np.ndarray):
    X = X.toarray()
print(f"Matrix: {X.shape}")
print("Computing Spearman gen po gen (~10-15 min)...")

results = []
for i in range(X.shape[1]):
    expr = X[:, i]
    if np.all(expr == 0):
        continue
    rho, p = spearmanr(expr, latent)
    if np.isnan(rho):
        continue
    results.append((ad_lt.var_names[i], rho, p))
    if (i+1) % 5000 == 0:
        print(f"  ...{i+1} processed, {len(results)} valid")

df_corr = pd.DataFrame(results, columns=["gene", "spearman_rho", "pval"])
df_corr["abs_rho"] = df_corr["spearman_rho"].abs()
df_corr["fdr"] = multipletests(df_corr["pval"].values, method="fdr_bh")[1]
df_corr = df_corr.sort_values(["fdr", "abs_rho"], ascending=[True, False])

out_all = os.path.join(OUT_DIR, "genes_correlated_with_latent_time__RAW_ALL_fdr.csv")
df_corr.to_csv(out_all, index=False)
print(f"SAVED: {out_all}")
print(f"  total valid: {df_corr.shape[0]}")
print(f"  rho range: {df_corr['spearman_rho'].min():.3f} to {df_corr['spearman_rho'].max():.3f}")

print("\n  Threshold breakdown:")
for label, (fdr_thr, rho_thr) in [
    ("STRICT_fdr01_rho03", (0.01, 0.30)),
    ("RELAXED_fdr05_rho02", (0.05, 0.20)),
    ("CASUAL_fdr05_rho015", (0.05, 0.15)),
]:
    mask = (df_corr["fdr"] < fdr_thr) & (df_corr["abs_rho"] > rho_thr)
    sub = df_corr[mask]
    n_pos = (sub["spearman_rho"] > 0).sum()
    n_neg = (sub["spearman_rho"] < 0).sum()
    print(f"    {label}: total={sub.shape[0]} pos={n_pos} neg={n_neg}")
    if sub.shape[0] >= 5:
        sub[sub["spearman_rho"] > 0].to_csv(
            os.path.join(OUT_DIR, f"latent_time_LATE__RAW_{label}.csv"), index=False
        )
        sub[sub["spearman_rho"] < 0].to_csv(
            os.path.join(OUT_DIR, f"latent_time_EARLY__RAW_{label}.csv"), index=False
        )

print("\n=== TOP 20 +rho (LATE genes, RAW) ===")
print(df_corr[df_corr["spearman_rho"] > 0].head(20)[["gene","spearman_rho","fdr"]].to_string(index=False))

print("\n=== TOP 20 -rho (EARLY genes, RAW) ===")
print(df_corr[df_corr["spearman_rho"] < 0].head(20)[["gene","spearman_rho","fdr"]].to_string(index=False))

# Provera DDR/EMT/CC gena
print("\n=== Provera kljucnih gena u RAW listi ===")
key_genes = {
    "DDR":  ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK1", "CHEK2", "PARP1", "FANCD2"],
    "EMT":  ["VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1"],
    "OXPHOS_examples": ["NDUFA1", "NDUFB1", "COX5A", "COX5B", "ATP5F1A", "ATP5F1B"],
    "Splicing": ["SF3B1", "U2AF1", "SRSF1", "HNRNPA1"],
}
for cat, glist in key_genes.items():
    print(f"\n  {cat}:")
    for g in glist:
        row = df_corr[df_corr["gene"] == g]
        if row.shape[0] > 0:
            r = row.iloc[0]
            print(f"    {g:10s}  rho={r['spearman_rho']:+.3f}  fdr={r['fdr']:.2e}")
        else:
            print(f"    {g:10s}  (nije u listi - mozda nije eksprimovan)")

print("\nDONE.")
