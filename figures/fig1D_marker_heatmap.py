"""
Figure 1D — Marker gene heatmap (top markers × Leiden clusters).

For each cluster, takes top 5 marker genes (by score from Stage 3 RAW DEG),
computes mean log-normalized expression per cluster, then z-scores per gene
across clusters, clipped at [-2, +2].

Manual block-diagonal grouping: cluster 0 markers first, cluster 1 markers
next, etc. — produces the "block-diagonal" pattern characteristic of marker
heatmaps.

Author: Marko Zivanovic
Output: PDF (vector) + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ============================================================
# GLOBAL STYLE
# ============================================================
plt.rcParams.update({
    'font.family':       'Arial',
    'font.size':         7,
    'axes.linewidth':    0.5,
    'axes.titlesize':    8,
    'axes.labelsize':    7,
    'xtick.labelsize':   7,
    'ytick.labelsize':   6.5,
    'legend.fontsize':   6.5,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'svg.fonttype':      'none',
})

# ============================================================
# CLUSTER PALETTE — same as Figure 1C
# ============================================================
CLUSTER_PALETTE = {
    '2': '#0000FF', '3': '#5B8DEF', '4': '#9DBBFF',      # NP231 family (2 dominant)
    '1': '#FF8000', '5': '#FFC266',                      # HP231 family (1 dominant)
    '0': '#00C000', '6': '#005A00',                      # LP231 family (0 dominant, 6 = G2/M)
    '7': '#8C564B', '9': '#D4A6C8',                      # mixed
    '10': '#E6007E', '11': '#E41A1C',                    # rare, highlighted
    '8': '#BDBDBD',                                      # QC
}

# ============================================================
# LOAD MARKER LIST (top N per cluster)
# ============================================================
MARKERS_CSV = DATA_ROOT + "/scanpy_out_MDA231/tables_stage3_RAW/markers_allclusters_top100_RAW.csv"
N_TOP = 5

print(f"Loading marker list: {MARKERS_CSV}")
m = pd.read_csv(MARKERS_CSV)
m['cluster'] = m['cluster'].astype(str)

top_markers = (
    m.sort_values(['cluster', 'scores'], ascending=[True, False])
     .groupby('cluster')
     .head(N_TOP)
)

cluster_order = [str(i) for i in range(12)]
gene_order = []
gene_to_cluster = {}
for c in cluster_order:
    genes = top_markers[top_markers['cluster']==c]['names'].tolist()
    for g in genes:
        if g not in gene_order:
            gene_order.append(g)
            gene_to_cluster[g] = c

print(f"Total unique genes: {len(gene_order)}")

# ============================================================
# LOAD STAGE 1 EXPRESSION DATA
# ============================================================
S1_PATH = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"
print(f"Loading Stage 1: {S1_PATH}")
ad = sc.read_h5ad(S1_PATH)
print(f"  Shape: {ad.shape}, raw: {ad.raw.shape if ad.raw is not None else 'None'}")
print(f"  Leiden clusters: {sorted(ad.obs['leiden'].unique().tolist())}")

# We use ad.raw (36,601 genes log-normalized — assumed). 
# If raw is in counts, we should log-normalize. Let's check.
# Standard scanpy raw is log-normalized counts. Let's verify.
print(f"  ad.X range: [{ad.X.min():.2f}, {ad.X.max():.2f}]")
if ad.raw is not None:
    # raw might have raw counts or log-norm; check by max value
    raw_max = ad.raw.X.max() if hasattr(ad.raw.X, 'max') else ad.raw.X.toarray().max()
    print(f"  ad.raw.X max: {raw_max:.2f}")

# ============================================================
# COMPUTE MEAN EXPRESSION PER CLUSTER PER GENE
# ============================================================
print("Computing mean expression per cluster...")

# Find gene indices in raw (or X if raw is None)
if ad.raw is not None:
    var_names = ad.raw.var_names
    expr_source = ad.raw.X
else:
    var_names = ad.var_names
    expr_source = ad.X

gene_idx = {}
for g in gene_order:
    if g in var_names:
        gene_idx[g] = var_names.get_loc(g)
    else:
        print(f"  WARNING: gene '{g}' not found in var_names")

# Compute mean per cluster
expr_matrix = pd.DataFrame(0.0, index=gene_order, columns=cluster_order)

leiden = ad.obs['leiden'].astype(str).values

for c in cluster_order:
    mask = leiden == c
    if mask.sum() == 0:
        continue
    cluster_cells = expr_source[mask, :]
    
    for g in gene_order:
        if g not in gene_idx:
            continue
        col_idx = gene_idx[g]
        # Get column
        if hasattr(cluster_cells, 'toarray'):
            vals = cluster_cells[:, col_idx].toarray().flatten()
        else:
            vals = np.asarray(cluster_cells[:, col_idx]).flatten()
        expr_matrix.loc[g, c] = float(vals.mean())

print(f"Expression matrix shape: {expr_matrix.shape}")
print(f"Expression range: [{expr_matrix.values.min():.2f}, {expr_matrix.values.max():.2f}]")

# Z-score per gene (across clusters)
zscore_matrix = expr_matrix.sub(expr_matrix.mean(axis=1), axis=0)
zscore_matrix = zscore_matrix.div(expr_matrix.std(axis=1) + 1e-9, axis=0)
zscore_matrix = zscore_matrix.clip(-2, 2)

print(f"Z-score range: [{zscore_matrix.values.min():.2f}, {zscore_matrix.values.max():.2f}]")

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 8.0
FIG_H_CM = 14.0
CM_TO_INCH = 1/2.54

fig = plt.figure(figsize=(FIG_W_CM*CM_TO_INCH, FIG_H_CM*CM_TO_INCH))

gs = GridSpec(2, 2, height_ratios=[0.8, 30], width_ratios=[20, 1],
              hspace=0.05, wspace=0.05,
              left=0.18, right=0.92, top=0.95, bottom=0.05)

ax_anno = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0])
ax_cbar = fig.add_subplot(gs[1, 1])

# === Annotation bar (cluster colors) ===
for i, c in enumerate(cluster_order):
    rect = plt.Rectangle((i, 0), 1, 1, facecolor=CLUSTER_PALETTE[c],
                          edgecolor='white', linewidth=0.5)
    ax_anno.add_patch(rect)

ax_anno.set_xlim(0, len(cluster_order))
ax_anno.set_ylim(0, 1)
ax_anno.set_xticks([i + 0.5 for i in range(len(cluster_order))])
ax_anno.set_xticklabels(cluster_order, fontsize=7)
ax_anno.xaxis.tick_top()
ax_anno.tick_params(axis='x', length=0, pad=2)
ax_anno.xaxis.set_label_position('top')
ax_anno.set_xlabel('Leiden cluster', fontsize=8, labelpad=4)
ax_anno.set_yticks([])
for spine in ax_anno.spines.values():
    spine.set_visible(False)

# === Main heatmap ===
im = ax_main.imshow(
    zscore_matrix.values,
    aspect='auto',
    cmap='RdBu_r',
    vmin=-2, vmax=2,
    interpolation='nearest',
)

ax_main.set_xticks([])

ax_main.set_yticks(range(len(gene_order)))
ax_main.set_yticklabels(gene_order, fontsize=5.5, style='italic')

ax_main.tick_params(axis='both', length=0, pad=2)

for spine in ax_main.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color('#888888')

# === Colorbar ===
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.set_label('Expression\n(z-score)', fontsize=7, labelpad=2)
cbar.ax.tick_params(labelsize=6.5, length=2)
cbar.outline.set_linewidth(0.5)
cbar.outline.set_edgecolor('#888888')

# Panel label
fig.text(0.02, 0.97, 'D', fontsize=11, fontweight='bold', va='top', ha='left')

# ============================================================
# SAVE
# ============================================================
OUTDIR_BASE = DATA_ROOT + '/figures'
OUTDIR_PDF  = os.path.join(OUTDIR_BASE, 'pdf')
OUTDIR_TIFF = os.path.join(OUTDIR_BASE, 'tiff')
OUTDIR_PNG  = os.path.join(OUTDIR_BASE, 'png')

os.makedirs(OUTDIR_PDF, exist_ok=True)
os.makedirs(OUTDIR_TIFF, exist_ok=True)
os.makedirs(OUTDIR_PNG, exist_ok=True)

pdf_path = os.path.join(OUTDIR_PDF, 'fig1D_marker_heatmap.pdf')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
print(f"Saved PDF: {pdf_path}")

tiff_path = os.path.join(OUTDIR_TIFF, 'fig1D_marker_heatmap.tiff')
fig.savefig(tiff_path, format='tiff', dpi=600,
             bbox_inches='tight', pad_inches=0.05,
             pil_kwargs={'compression': 'tiff_lzw'})
print(f"Saved TIFF: {tiff_path}")

png_path = os.path.join(OUTDIR_PNG, 'fig1D_marker_heatmap_preview.png')
fig.savefig(png_path, format='png', dpi=200,
             bbox_inches='tight', pad_inches=0.05)
print(f"Saved PNG preview: {png_path}")

for p in [pdf_path, tiff_path, png_path]:
    sz = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):45s}  {sz:6.2f} MB")

plt.close()
print("\nDONE.")
