#!/usr/bin/env python3
"""
Independent validation: GSE297522 (Hong, Lee, Neumann; Univ. Pittsburgh; public 31 Aug 2026)
Bulk RNA-seq (featureCounts, NovaSeq 6000) of PARPi-resistant TNBC lines.

Used here: MDA-MB-231 (M231), vehicle-treated (Vh) only, n = 3 per group:
    parental (Par) vs niraparib- (Nir), olaparib- (Ola), talazoparib-resistant (Tal).
NFA-treated samples (Tx) and the other cell lines (HCC1937, MDA-MB-436) are not used.

Method: PyDESeq2 (Wald test, design ~ condition, reference = Par), genes with
>= 10 counts in >= 3 of the 12 samples; duplicated gene symbols are summed.
Mean CPM per group is reported to judge detectability (CPM < 1 = not detected).

Outputs (validation_datasets/GSE297522/results/):
    GSE297522_M231_DESeq2_all_genes.csv   (all genes, 3 contrasts)
    GSE297522_M231_panel_genes.csv         (genes discussed in the manuscript)

Run:
    conda run --no-capture-output -n deseq python validation_GSE297522_deseq2.py
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

BASE = os.environ.get("GSE_BASE", DATA_ROOT + "/validation_datasets/GSE297522")
COUNTS = f"{BASE}/GSE297522_HumanTNBCPARPi_40826g66s_FeatureCount.txt.gz"
OUT = f"{BASE}/results"
os.makedirs(OUT, exist_ok=True)

GROUPS = ["Par", "Nir", "Ola", "Tal"]
PANEL = [
    # transporters
    "ABCB1", "ABCG2", "ABCC1", "ABCC2", "ATP7A", "ATP7B", "SLC31A1",
    # EMT
    "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "VIM", "CDH2", "FN1", "CDH1", "TGFB1", "MMP2", "MMP9", "HES1",
    # DDR
    "PARP1", "PARP2", "PARG", "BRCA1", "BRCA2", "PALB2", "RAD51", "RAD51C", "MRE11", "RAD50", "NBN", "RBBP8",
    "TP53BP1", "MAD2L2", "XRCC5", "XRCC6", "PRKDC", "LIG4", "ATM", "ATR", "CHEK1", "CHEK2", "WEE1", "SLFN11",
    "FANCD2", "XRCC1", "POLB", "MLH1", "MSH2",
    # drivers discussed
    "TOP2A", "FAM111B", "JUND",
]

print("=" * 70 + "\nLOAD\n" + "=" * 70)
raw = pd.read_csv(COUNTS, sep="\t", index_col=0)
raw.index = raw.index.astype(str)
ndup = raw.index.duplicated(keep=False).sum()
raw = raw.groupby(level=0).sum()
print(f"genes: {raw.shape[0]} (duplicated symbol rows summed: {ndup}) | samples: {raw.shape[1]}")

cols = [c for c in raw.columns if c.startswith("M231_") and "_Vh_" in c]
meta = pd.DataFrame({"sample": cols})
meta["condition"] = meta["sample"].str.split("_").str[1]
meta = meta.set_index("sample")
meta = meta[meta["condition"].isin(GROUPS)]
print(meta["condition"].value_counts().reindex(GROUPS).to_string())
counts = raw[meta.index].T.astype(int)          # samples x genes

lib = counts.sum(axis=1)
cpm = counts.div(lib, axis=0) * 1e6
mean_cpm = cpm.groupby(meta["condition"]).mean().T[GROUPS]
mean_cpm.columns = [f"CPM_{g}" for g in GROUPS]

keep = (counts >= 10).sum(axis=0) >= 3
counts_f = counts.loc[:, keep]
print(f"genes kept (>=10 counts in >=3 samples): {counts_f.shape[1]}")

print("=" * 70 + "\nDESeq2\n" + "=" * 70)
meta["condition"] = pd.Categorical(meta["condition"], categories=GROUPS)
try:
    dds = DeseqDataSet(counts=counts_f, metadata=meta, design="~condition", refit_cooks=True, quiet=True)
except TypeError:
    dds = DeseqDataSet(counts=counts_f, metadata=meta, design_factors="condition",
                       ref_level=["condition", "Par"], refit_cooks=True, quiet=True)
dds.deseq2()

res_all = []
for g in ["Nir", "Ola", "Tal"]:
    st = DeseqStats(dds, contrast=["condition", g, "Par"], quiet=True)
    st.summary()
    r = st.results_df[["baseMean", "log2FoldChange", "pvalue", "padj"]].copy()
    r.columns = [f"{c}_{g}vsPar" for c in r.columns]
    res_all.append(r)
    n = ((r[f"padj_{g}vsPar"] < 0.05) & (r[f"log2FoldChange_{g}vsPar"].abs() >= 1)).sum()
    print(f"  {g} vs Par: genes with padj<0.05 & |log2FC|>=1: {n}")
res = pd.concat(res_all, axis=1).join(mean_cpm, how="outer")
res.index.name = "gene"
res.to_csv(f"{OUT}/GSE297522_M231_DESeq2_all_genes.csv")

panel = pd.DataFrame(index=PANEL).join(mean_cpm.round(2)).join(
    res[[c for c in res.columns if c.startswith(("log2FoldChange", "padj"))]])
panel.index.name = "gene"
panel.to_csv(f"{OUT}/GSE297522_M231_panel_genes.csv")

print("=" * 70 + "\nPANEL GENES  (log2FC vs parental; padj; mean CPM)\n" + "=" * 70)
pd.set_option("display.width", 250)
show = pd.DataFrame(index=PANEL)
for g in ["Nir", "Ola", "Tal"]:
    lfc = panel[f"log2FoldChange_{g}vsPar"]; q = panel[f"padj_{g}vsPar"]
    show[f"{g} log2FC"] = lfc.round(2)
    show[f"{g} padj"] = q.map(lambda x: f"{x:.1e}" if pd.notna(x) else "NA")
show["CPM Par"] = panel["CPM_Par"]; show["CPM Nir"] = panel["CPM_Nir"]
print(show.to_string())
print(f"\nsaved: {OUT}/GSE297522_M231_DESeq2_all_genes.csv\nsaved: {OUT}/GSE297522_M231_panel_genes.csv\nDONE.")
