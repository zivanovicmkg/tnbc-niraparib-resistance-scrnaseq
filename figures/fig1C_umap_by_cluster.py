"""
Figure 1C — UMAP visualization, colored by Leiden cluster (12 clusters).

Sample-driven palette: clusters grouped by their dominant sample of origin.
- C231 family (clusters 2, 3, 4): blue shades
- N231 family (clusters 1, 5, 10, 11): gold shades
- N436 family (clusters 0, 6): red shades
- Mixed/transitional (clusters 7, 9): purple
- QC artifact (cluster 8): grey, deprioritized

Author: Marko Zivanovic
Output: PDF (vector) + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# GLOBAL STYLE
# ============================================================
plt.rcParams.update({
    'font.family':       'Arial',
    'font.size':         8,
    'axes.linewidth':    0.6,
    'axes.titlesize':    9,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7.5,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'svg.fonttype':      'none',
})

# ============================================================
# CLUSTER PALETTE — SAMPLE-DRIVEN
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
# LOAD DATA
# ============================================================
CR_PATH = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
S1_PATH = DATA_ROOT + "/velocity/03_outputs/fate0_analysis/stage1_with_cellrank_fates.h5ad"

print(f"Loading CellRank merged: {CR_PATH}")
ad_cr = sc.read_h5ad(CR_PATH)

print(f"Loading Stage1 + fate annotations: {S1_PATH}")
ad_s1 = sc.read_h5ad(S1_PATH)

# Build mapping CellRank -> Stage1 by normalized barcode
def normalize_cr(bc):
    sample = bc.split('_')[0]
    barcode = bc.split(':')[1].rstrip('x')
    return f"{barcode}-{sample}"

def normalize_s1(bc):
    parts = bc.split('-')
    return f"{parts[0]}-{parts[-1]}"

cr_keys = pd.Index([normalize_cr(bc) for bc in ad_cr.obs_names])
s1_keys = pd.Index([normalize_s1(bc) for bc in ad_s1.obs_names])

s1_idx_map = pd.Series(index=s1_keys, data=range(len(s1_keys)))
mapped_s1_idx = np.array([s1_idx_map[k] for k in cr_keys])

# Transfer leiden annotation from Stage1 to CellRank order
leiden = pd.Series(ad_s1.obs['leiden'].values[mapped_s1_idx]).astype(str).values
umap = ad_cr.obsm['X_umap']

print(f"Cells: {len(leiden)}")
print(f"Clusters: {pd.Series(leiden).value_counts().sort_index().to_dict()}")

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 9.0
FIG_H_CM = 7.0
CM_TO_INCH = 1/2.54

fig, ax = plt.subplots(figsize=(FIG_W_CM*CM_TO_INCH, FIG_H_CM*CM_TO_INCH))

# Plot order: large clusters first (background), small clusters last (foreground)
# Sizes: 0=5092, 1=3866, 2=2692, 3=1541, 4=1317, 5=1161, 6=848, 7=696, 8=418, 9=244, 10=59, 11=19
plot_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

for cluster in plot_order:
    mask = leiden == cluster
    if mask.sum() == 0:
        continue
    n = mask.sum()
    # Cluster 8 (QC) — smaller alpha to deprioritize
    # Small clusters (10, 11) — bigger marker so they're visible
    alpha = 0.4 if cluster == '8' else 0.7
    s_size = 2.5 if cluster in ['10', '11'] else 0.8
    ax.scatter(
        umap[mask, 0], umap[mask, 1],
        c=CLUSTER_PALETTE[cluster],
        s=s_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        label=f"{cluster} (n={n:,})",
    )

# Style
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('UMAP 1', fontsize=8, labelpad=2)
ax.set_ylabel('UMAP 2', fontsize=8, labelpad=2)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color('#888888')

ax.set_aspect('equal', adjustable='datalim')
ax.set_title('Leiden cluster', fontsize=8.5, fontweight='bold', loc='left', pad=4)

# Legend — natural order 0-11
legend_handles = []
for cluster in [str(i) for i in range(12)]:
    n = (leiden == cluster).sum()
    label = f'{cluster}'
    if cluster == '8':
        label = f'{cluster} (QC)'
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='none',
                   markerfacecolor=CLUSTER_PALETTE[cluster],
                   markersize=5.5, markeredgewidth=0,
                   label=label)
    )

ax.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    handletextpad=0.3,
    labelspacing=0.35,
    borderaxespad=0,
    ncol=1,
    fontsize=7,
)

# Panel label
fig.text(0.02, 0.96, 'C', fontsize=11, fontweight='bold', va='top', ha='left')

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

pdf_path = os.path.join(OUTDIR_PDF, 'fig1C_umap_by_cluster.pdf')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
print(f"Saved PDF: {pdf_path}")

tiff_path = os.path.join(OUTDIR_TIFF, 'fig1C_umap_by_cluster.tiff')
fig.savefig(tiff_path, format='tiff', dpi=600,
             bbox_inches='tight', pad_inches=0.05,
             pil_kwargs={'compression': 'tiff_lzw'})
print(f"Saved TIFF: {tiff_path}")

png_path = os.path.join(OUTDIR_PNG, 'fig1C_umap_by_cluster_preview.png')
fig.savefig(png_path, format='png', dpi=200,
             bbox_inches='tight', pad_inches=0.05)
print(f"Saved PNG preview: {png_path}")

for p in [pdf_path, tiff_path, png_path]:
    sz = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):45s}  {sz:6.2f} MB")

plt.close()
print("\nDONE.")
