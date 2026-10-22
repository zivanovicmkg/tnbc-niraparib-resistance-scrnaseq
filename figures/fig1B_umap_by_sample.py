"""
Figure 1B — UMAP visualization of integrated MDA-MB-231 dataset, colored by sample.

Uses CellRank merged UMAP space (same as Figures 2, 3, 4, 5) for consistency.
Sample annotations transferred from Stage 1 by barcode mapping.

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
# COLORS — paleta consistent kroz ceo rad
# ============================================================
COLOR_C231 = '#0000FF'   # NP231 blue — parental
COLOR_N231 = '#FF8000'   # HP231 orange — high dose
COLOR_N436 = '#00C000'   # LP231 green — low dose, highest resistance

PALETTE = {
    'C231': COLOR_C231,
    'N231': COLOR_N231,
    'N436': COLOR_N436,
}
DISP = {'C231': 'NP231', 'N231': 'HP231', 'N436': 'LP231'}

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

# Transfer sample annotation from Stage1 to CellRank order
samples = ad_s1.obs['sample_id'].values[mapped_s1_idx]
umap = ad_cr.obsm['X_umap']

print(f"Cells: {len(samples)}")
print(f"Samples: {pd.Series(samples).value_counts().to_dict()}")

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 7.0
FIG_H_CM = 7.0
CM_TO_INCH = 1/2.54

fig, ax = plt.subplots(figsize=(FIG_W_CM*CM_TO_INCH, FIG_H_CM*CM_TO_INCH))

plot_order = ['C231', 'N231', 'N436']

for sample in plot_order:
    mask = samples == sample
    ax.scatter(
        umap[mask, 0], umap[mask, 1],
        c=PALETTE[sample],
        s=0.8,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        label=f"{DISP[sample]} (n={mask.sum():,})",
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

ax.set_title('Sample', fontsize=8.5, fontweight='bold', loc='left', pad=4)

# Legend
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=PALETTE[s],
               markersize=5.5, markeredgewidth=0,
               label=f'{DISP[s]} (n={(samples==s).sum():,})')
    for s in plot_order
]
ax.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    handletextpad=0.3,
    labelspacing=0.6,
    borderaxespad=0,
)

# Panel label
fig.text(0.02, 0.96, 'B', fontsize=11, fontweight='bold', va='top', ha='left')

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

pdf_path = os.path.join(OUTDIR_PDF, 'fig1B_umap_by_sample.pdf')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
print(f"Saved PDF: {pdf_path}")

tiff_path = os.path.join(OUTDIR_TIFF, 'fig1B_umap_by_sample.tiff')
fig.savefig(tiff_path, format='tiff', dpi=600,
             bbox_inches='tight', pad_inches=0.05,
             pil_kwargs={'compression': 'tiff_lzw'})
print(f"Saved TIFF: {tiff_path}")

png_path = os.path.join(OUTDIR_PNG, 'fig1B_umap_by_sample_preview.png')
fig.savefig(png_path, format='png', dpi=200,
             bbox_inches='tight', pad_inches=0.05)
print(f"Saved PNG preview: {png_path}")

for p in [pdf_path, tiff_path, png_path]:
    sz = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):45s}  {sz:6.2f} MB")

plt.close()
print("\nDONE.")
