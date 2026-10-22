"""
Finalni enrichment na RAW-expanded listama.
Plus: gene-set test za Hallmark Myc Targets, OXPHOS, EMT da vidimo
da li su zaista u 'fate3 UP' i 'fate3 high vs low' listama.
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os, warnings, math
warnings.filterwarnings("ignore")
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt

BASE = DATA_ROOT + "/velocity/03_outputs"
TAB_OUT = os.path.join(BASE, "tabular", "enrichment_RAW")
FIG_OUT = os.path.join(BASE, "figures", "enrichment_RAW")
os.makedirs(TAB_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

# Filtriramo gen liste sa pragovima da izbegnemo housekeeping inflation
RAW_DIR = os.path.join(BASE, "tabular", "RAW_expanded")

# Top 200 gena, ali sa logFC pragom za fate3 vs other (da uklonimo housekeeping)
df_a = pd.read_csv(os.path.join(RAW_DIR, "DEG_fate3_high_vs_other__RAW_wilcoxon.csv"))
df_a_filt = df_a[(df_a["pvals_adj"] < 0.01) & (df_a["logfoldchanges"] > 1.5)].head(300)
df_a_filt.to_csv(os.path.join(RAW_DIR, "fate3_high_vs_other__top300_logFC15.csv"), index=False)

# fate3 high vs low - top 200 UP, top 200 DOWN
df_b = pd.read_csv(os.path.join(RAW_DIR, "DE_fate3_high_vs_low__RAW_LTcontrolled.csv"))
df_b_up = df_b[(df_b["pvals_adj"] < 0.05) & (df_b["logfoldchanges"] > 0.5)].head(200)
df_b_dn = df_b[(df_b["pvals_adj"] < 0.05) & (df_b["logfoldchanges"] < -0.5)].head(200)

# Spearman LATE/EARLY (CASUAL prag)
df_late = pd.read_csv(os.path.join(RAW_DIR, "latent_time_LATE__RAW_CASUAL_fdr05_rho015.csv"))
df_early = pd.read_csv(os.path.join(RAW_DIR, "latent_time_EARLY__RAW_CASUAL_fdr05_rho015.csv"))

INPUTS = {
    "fate3_vs_other_RAW_filtered":      list(df_a_filt["names"].astype(str).unique()),
    "fate3_high_vs_low_RAW_UP":         list(df_b_up["names"].astype(str).unique()),
    "fate3_high_vs_low_RAW_DOWN":       list(df_b_dn["names"].astype(str).unique()),
    "latent_time_LATE_RAW":             list(df_late["gene"].astype(str).unique()),
    "latent_time_EARLY_RAW":            list(df_early["gene"].astype(str).unique()),
}

LIBRARIES = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
    "MSigDB_Hallmark_2020",
    "Reactome_2022",
]

print("=" * 70)
print("INPUT GENE LISTS (RAW-expanded):")
print("=" * 70)
for label, genes in INPUTS.items():
    print(f"  {label}: {len(genes)} genes")
    print(f"    primer: {genes[:6]}")
    print()

all_summary = []
for label, genes in INPUTS.items():
    if len(genes) < 5:
        print(f"  {label}: premalo gena, preskace")
        continue
    print("=" * 70)
    print(f"INPUT: {label}  ({len(genes)} gena)")
    print("=" * 70)

    for lib in LIBRARIES:
        print(f"  [{lib}] ", end="", flush=True)
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib,
                             organism="human", outdir=None, no_plot=True)
            res = enr.results.copy().sort_values("Adjusted P-value")
            if res.shape[0] == 0:
                print("nema hitova")
                continue
            res["input_list"] = label
            res["library"] = lib
            res.to_csv(os.path.join(TAB_OUT, f"{label}__{lib}.csv"), index=False)
            res.head(15).to_csv(
                os.path.join(TAB_OUT, f"{label}__{lib}_top15.csv"), index=False
            )

            sig = res[res["Adjusted P-value"] < 0.05].head(15)
            if sig.shape[0] > 0:
                fig, ax = plt.subplots(figsize=(9, 0.4 * sig.shape[0] + 1))
                terms = sig["Term"].str[:60][::-1]
                neg_log_p = sig["Adjusted P-value"][::-1].apply(
                    lambda x: -math.log10(max(x, 1e-300))
                )
                ax.barh(terms, neg_log_p)
                ax.set_xlabel("-log10(adj P)")
                ax.set_title(f"{label}\n{lib}")
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_OUT, f"{label}__{lib}_barplot.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()

            n_sig = (res["Adjusted P-value"] < 0.05).sum()
            print(f"-> {res.shape[0]} terms, {n_sig} sig (q<0.05)")
            all_summary.append({
                "list": label, "library": lib, "n_genes": len(genes),
                "n_terms": res.shape[0], "n_sig_q05": n_sig,
                "top_term": res.iloc[0]["Term"][:70],
                "top_q": res.iloc[0]["Adjusted P-value"],
            })
        except Exception as e:
            print(f"GRESKA: {e}")

if all_summary:
    sdf = pd.DataFrame(all_summary)
    out = os.path.join(TAB_OUT, "ENRICHMENT_SUMMARY_RAW.csv")
    sdf.to_csv(out, index=False)
    print("\n" + "=" * 80)
    print("ENRICHMENT SUMMARY (RAW-expanded)")
    print("=" * 80)
    print(sdf.to_string(index=False))
    print(f"\nSAVED: {out}")

print("\nDONE.")
