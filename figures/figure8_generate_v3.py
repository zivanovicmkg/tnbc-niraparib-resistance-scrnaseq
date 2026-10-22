#!/usr/bin/env python3
"""
Figure 6 generator - N436 cluster 6 (G2/M proliferative) vs cluster 0 (bulk).
Recomputed reproducibly from N436_phase2_processed.h5ad (seed=0).

Panels:
  6A  DE dotplot: top G2/M markers, cluster 0 vs cluster 6
  6B  ORA bar:    Hallmark 2020 + GO BP 2023 on up-in-c6 DE genes
  6C  Phase composition: stacked bar c0 vs c6 (Tirosh cell-cycle scoring)
  6D  Post-CC-regression UMAP: does c6 dissolve into c0 (honest visual)
  6E  Module scores: TNF/AP-1 and DDR, c0 vs c6 (pipeline 4-gene programs)

Program gene sets taken verbatim from fig1F_functional_compass.py.
Prints every key number vs the manuscript for reconciliation.

Run:
    conda run --no-capture-output -n b2bio python figure8_generate_v3.py

v3 (20.10.2026): layout fix only (B labels no longer overlap A; C legend
clear of title; E delta labels clear of bars; no suptitle). Output = Figure_8.*
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import textwrap
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # Arial if available; DejaVu is safe fallback
    "font.size": 7, "axes.linewidth": 0.5, "axes.titlesize": 8,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
RED = "#B2182B"; GREY = "#9E9E9E"
PHASE_COLORS = {"G1": "#4A7AB5", "S": "#E8A33D", "G2M": "#B2182B"}

PATH = (DATA_ROOT + "/scanpy_out_MDA231/"
        "objects/N436_phase2_processed.h5ad")
OUT = DATA_ROOT + "/figures"
for sub in ["pdf", "tiff", "png"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# Tirosh cell-cycle genes
S_GENES = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
           "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP",
           "RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2",
           "RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
           "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
G2M_GENES = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
             "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
             "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
             "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
             "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1",
             "ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]

# Pipeline programs (from fig1F_functional_compass.py)
PROG_TNF = ["JUND", "FOS", "JUN", "FOSL1"]
PROG_DDR = ["BRCA1", "RAD51", "ATM", "CHEK1"]

# canonical G2/M markers highlighted in manuscript 6A
MARK = ["CCNB1","CDC20","AURKA","PLK1","TPX2","CENPF","KIF20A","NEK2","ASPM","DLGAP5"]

def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
REC = {}

# ------------------------------------------------------------------- load
hdr("LOAD + cell-cycle scoring")
ad = sc.read_h5ad(PATH)
ad.obs["leiden"] = ad.obs["leiden"].astype(str)
s = [g for g in S_GENES if g in ad.var_names]
g2 = [g for g in G2M_GENES if g in ad.var_names]
sc.tl.score_genes_cell_cycle(ad, s_genes=s, g2m_genes=g2)
print("cells:", ad.n_obs, "| c0:", (ad.obs.leiden == "0").sum(),
      "| c6:", (ad.obs.leiden == "6").sum())

# ------------------------------------------------------------------- 6A DE
hdr("6A  DE cluster 6 vs 0")
sc.tl.rank_genes_groups(ad, "leiden", groups=["6"], reference="0",
                        method="wilcoxon", use_raw=False)
de = sc.get.rank_genes_groups_df(ad, group="6")
up = de[(de.pvals_adj < 0.05) & (de.logfoldchanges > 1)].copy()
up = up.sort_values("scores", ascending=False)
REC["6A DE gene count"] = (len(up), 70)
print(f"up in c6 (padj<0.05 & log2FC>1): {len(up)}  (manuscript 70)")
# top genes for dotplot: canonical markers first, then fill from top-scored
top_genes = [g for g in MARK if g in set(up.names)]
for g in up.names:
    if g not in top_genes:
        top_genes.append(g)
    if len(top_genes) >= 16:
        break

# ------------------------------------------------------------------- 6C phase
hdr("6C  phase composition")
def phase_frac(cl):
    sub = ad.obs[ad.obs.leiden == cl]
    f = (sub.phase.value_counts(normalize=True) * 100).reindex(["G1","S","G2M"]).fillna(0)
    return f, float(sub.G2M_score.median()), len(sub)
f6, med6, n6 = phase_frac("6")
f0, med0, n0 = phase_frac("0")
print(f"c6: G1={f6.G1:.1f} S={f6.S:.1f} G2M={f6.G2M:.1f} | medG2M={med6:.3f} (ms 97.8/0.525)")
print(f"c0: G1={f0.G1:.1f} S={f0.S:.1f} G2M={f0.G2M:.1f} | medG2M={med0:.3f} (ms 50/31/19/0.161)")
REC["6C c6 %G2M"] = (round(f6.G2M, 1), 97.8)
REC["6C c0 %G2M"] = (round(f0.G2M, 1), 50.0)

# ------------------------------------------------------------------- 6E modules
hdr("6E  module scores (4-gene pipeline programs)")
sc.tl.score_genes(ad, [g for g in PROG_TNF if g in ad.var_names],
                  score_name="TNF_AP1", random_state=0, use_raw=False)
sc.tl.score_genes(ad, [g for g in PROG_DDR if g in ad.var_names],
                  score_name="DDR", random_state=0, use_raw=False)
def mod(cl, k): return float(ad.obs.loc[ad.obs.leiden == cl, k].mean())
tnf6, tnf0 = mod("6", "TNF_AP1"), mod("0", "TNF_AP1")
ddr6, ddr0 = mod("6", "DDR"), mod("0", "DDR")
print(f"TNF/AP-1  c6={tnf6:.3f} c0={tnf0:.3f}  Δ={tnf6-tnf0:.3f}  (ms −0.164)")
print(f"DDR       c6={ddr6:.3f} c0={ddr0:.3f}  Δ={ddr6-ddr0:.3f}  (ms −0.072)")
REC["6E TNF/AP-1 Δ"] = (round(tnf6 - tnf0, 3), -0.164)
REC["6E DDR Δ"] = (round(ddr6 - ddr0, 3), -0.072)

# ------------------------------------------------------------------- 6B ORA
hdr("6B  ORA (Enrichr)")
ora = None
try:
    import gseapy as gp
    enr = gp.enrichr(gene_list=list(up.names),
                     gene_sets=["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"],
                     organism="human", outdir=None)
    ora = enr.results.copy()
    ora["mlog10"] = -np.log10(ora["Adjusted P-value"].clip(lower=1e-300))
    print(ora.sort_values("Adjusted P-value").head(6)[["Gene_set","Term","Adjusted P-value"]].to_string())
except Exception as e:
    print("ORA failed (offline?):", repr(e))

# ------------------------------------------------------------------- 6D regression
hdr("6D  CC regression + re-Leiden + UMAP")
rad = ad.copy()
sc.pp.highly_variable_genes(rad, n_top_genes=2000, flavor="seurat")
rad = rad[:, rad.var.highly_variable].copy()
sc.pp.regress_out(rad, ["S_score", "G2M_score"])
sc.pp.scale(rad, max_value=10)
sc.tl.pca(rad, n_comps=30, random_state=0)
sc.pp.neighbors(rad, n_neighbors=15, random_state=0)
sc.tl.umap(rad, random_state=0)
print("redistribution across resolutions (c6 vs c0 dominant post-cluster):")
for res in [0.1, 0.2, 0.3, 0.5]:
    sc.tl.leiden(rad, resolution=res, random_state=0, key_added="lp",
                 flavor="igraph", n_iterations=2, directed=False)
    ct = pd.crosstab(ad.obs.leiden, rad.obs.lp)
    r6 = (ct.loc["6"] / ct.loc["6"].sum() * 100)
    r0 = (ct.loc["0"] / ct.loc["0"].sum() * 100)
    d6, d0 = r6.idxmax(), r0.idxmax()
    shared = "SHARED" if d6 == d0 else "different"
    print(f"  res={res}: c6->{d6}:{r6.max():.0f}%  c0->{d0}:{r0.max():.0f}%  [{shared}]")
rad.obs["orig"] = ad.obs.leiden.values
rad.obs["phase"] = ad.obs.phase.values

# ==================================================================== FIGURE
hdr("RENDER composite Figure 6")
CM = 1 / 2.54
fig = plt.figure(figsize=(18 * CM, 13 * CM))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
              hspace=0.62, wspace=0.42,
              left=0.08, right=0.97, top=0.93, bottom=0.12)
gtop = gs[0, :].subgridspec(1, 2, width_ratios=[1.9, 1], wspace=0.95)  # room for B labels

# ---- 6A dotplot (span row0 col0-1)
axA = fig.add_subplot(gtop[0, 0])
cl2 = ["0", "6"]
meanmat = np.zeros((2, len(top_genes))); pctmat = np.zeros((2, len(top_genes)))
for yi, cl in enumerate(cl2):
    m = (ad.obs.leiden == cl).values
    for xi, g in enumerate(top_genes):
        x = ad[:, g].X
        x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
        v = x[m]
        meanmat[yi, xi] = v.mean(); pctmat[yi, xi] = (v > 0).mean() * 100
vmax = max(1e-9, meanmat.max())
for yi in range(2):
    for xi in range(len(top_genes)):
        axA.scatter(xi, yi, s=pctmat[yi, xi] * 3.2, c=[meanmat[yi, xi]],
                    cmap="Reds", vmin=0, vmax=vmax, edgecolors="#888", linewidths=0.3)
axA.set_xticks(range(len(top_genes)))
axA.set_xticklabels(top_genes, rotation=45, ha="right", style="italic", fontsize=6)
axA.set_yticks([0, 1]); axA.set_yticklabels(["cluster 0", "cluster 6"])
axA.set_ylim(-0.5, 1.5); axA.set_xlim(-0.6, len(top_genes) - 0.4)
axA.set_title(f"A   DE markers c6 vs c0 (n={len(up)} genes, log2FC>1)", loc="left", fontweight="bold")
axA.tick_params(length=0)
for sp in axA.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.5)

# ---- 6B ORA bar
axB = fig.add_subplot(gtop[0, 1])
if ora is not None:
    pick = (ora.sort_values("Adjusted P-value")
               .drop_duplicates("Term").head(6).iloc[::-1])
    cols = ["#B2182B" if gs_ == "MSigDB_Hallmark_2020" else "#2E5A8C"
            for gs_ in pick.Gene_set]
    axB.barh(range(len(pick)), pick.mlog10, color=cols, height=0.7)
    axB.set_yticks(range(len(pick)))
    axB.set_yticklabels([textwrap.fill(t.split(" (GO:")[0], 18) for t in pick.Term],
                        fontsize=5.5, linespacing=0.95)
    axB.axvline(-np.log10(0.05), ls="--", lw=0.5, color="#555")
    axB.set_xlabel("-log10 adj.p")
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color="#B2182B", label="Hallmark"),
                        Patch(color="#2E5A8C", label="GO BP")],
               loc="lower right", frameon=False, fontsize=5)
else:
    axB.text(0.5, 0.5, "ORA offline", ha="center")
axB.set_title("B   Over-representation", loc="left", fontweight="bold")
for sp in axB.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.5)

# ---- 6C phase stacked bar
axC = fig.add_subplot(gs[1, 0])
b0 = b6 = 0.0
for ph in ["G1", "S", "G2M"]:
    axC.bar(["cluster 0", "cluster 6"], [f0[ph], f6[ph]], bottom=[b0, b6],
            color=PHASE_COLORS[ph], label=ph, width=0.6)
    b0 += f0[ph]; b6 += f6[ph]
axC.set_ylabel("% cells"); axC.set_ylim(0, 100)
axC.legend(frameon=False, fontsize=5.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
           borderaxespad=0.2, handlelength=1, columnspacing=0.8)
axC.set_title("C   Cell-cycle phase", loc="left", fontweight="bold", pad=13)
for sp in axC.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.5)

# ---- 6D post-regression UMAP (orig cluster)
axD = fig.add_subplot(gs[1, 1])
U = rad.obsm["X_umap"]
m0 = (rad.obs.orig == "0").values; m6 = (rad.obs.orig == "6").values
other = ~(m0 | m6)
axD.scatter(U[other, 0], U[other, 1], s=1, c="#DDDDDD", linewidths=0, rasterized=True)
axD.scatter(U[m0, 0], U[m0, 1], s=1.5, c=GREY, linewidths=0, rasterized=True, label="c0")
axD.scatter(U[m6, 0], U[m6, 1], s=2.5, c=RED, linewidths=0, rasterized=True, label="c6")
axD.set_xticks([]); axD.set_yticks([]); axD.set_xlabel("UMAP1"); axD.set_ylabel("UMAP2")
axD.legend(frameon=False, markerscale=4, fontsize=5.5, loc="best")
axD.set_title("D   Post-CC-regression", loc="left", fontweight="bold")

# ---- 6E module score bars
axE = fig.add_subplot(gs[1, 2])
xlab = ["TNF/AP-1", "DDR"]
c0v = [tnf0, ddr0]; c6v = [tnf6, ddr6]
xx = np.arange(2)
axE.bar(xx - 0.2, c0v, 0.38, color=GREY, label="c0")
axE.bar(xx + 0.2, c6v, 0.38, color=RED, label="c6")
for i, (a, b) in enumerate(zip(c0v, c6v)):
    axE.text(i, max(a, b, 0) + 0.012, f"Δ={b-a:+.3f}", ha="center", va="bottom", fontsize=5.5)
axE.set_ylim(top=max(max(c0v), max(c6v), 0) + 0.07)
axE.set_xticks(xx); axE.set_xticklabels(xlab)
axE.set_ylabel("module score"); axE.axhline(0, lw=0.5, color="#555")
axE.legend(frameon=False, fontsize=5.5, loc="best")
axE.set_title("E   Module scores", loc="left", fontweight="bold")
for sp in axE.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.5)


for ext, kw in [("pdf", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                ("png", {"dpi": 200})]:
    p = os.path.join(OUT, ext, f"Figure_8.{ext}")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    print("saved", p)
plt.close()

hdr("SUMMARY computed vs manuscript")
for k, (c, m) in REC.items():
    print(f"  {k:22s} computed={str(c):12s} ms={m}")
print("\nDONE.")
