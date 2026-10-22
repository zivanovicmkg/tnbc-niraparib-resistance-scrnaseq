"""
Figure 1E — Stacked bar plot of cluster composition per sample.

Shows proportion of each Leiden cluster within each sample (C231/N231/N436).
Bars sum to 100% (proportion). Cluster colors consistent with Figure 1C.

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
    'font.size':         7,
    'axes.linewidth':    0.5,
    'axes.titlesize':    8,
    'axes.labelsize':    7,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
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
# LOAD DATA
# ============================================================
S1_PATH = DATA_ROOT + "/velocity/03_outputs/fate0_analysis/stage1_with_cellrank_fates.h5ad"
print(f"Loading: {S1_PATH}")
ad = sc.read_h5ad(S1_PATH)
print(f"Cells: {ad.n_obs}")
print(f"Samples: {ad.obs['sample_id'].value_counts().to_dict()}")

# Make a small DataFrame with sample + leiden
df = pd.DataFrame({
    'sample': ad.obs['sample_id'].astype(str).values,
    'leiden': ad.obs['leiden'].astype(str).values,
})

# ============================================================
# COMPUTE PROPORTIONS
# ============================================================
ct = pd.crosstab(df['sample'], df['leiden'])
ct = ct.reindex(['C231', 'N231', 'N436'])

cluster_order = [str(i) for i in range(12)]
ct = ct[cluster_order]

prop = ct.div(ct.sum(axis=1), axis=0)
print("\nProportions per sample per cluster:")
print(prop.round(3))

# Sample size labels
sample_n = ct.sum(axis=1).to_dict()

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 6.5
FIG_H_CM = 7.0
CM_TO_INCH = 1/2.54

fig, ax = plt.subplots(figsize=(FIG_W_CM*CM_TO_INCH, FIG_H_CM*CM_TO_INCH))

bar_width = 0.6
x_positions = np.arange(len(prop.index))

# Stack from cluster 0 (bottom) to cluster 11 (top)
bottom = np.zeros(len(prop.index))
for cluster in cluster_order:
    values = prop[cluster].values
    ax.bar(
        x_positions, values, bar_width,
        bottom=bottom,
        color=CLUSTER_PALETTE[cluster],
        edgecolor='white',
        linewidth=0.4,
        label=cluster,
    )
    bottom += values

# Style
DISP = {'C231': 'NP231', 'N231': 'HP231', 'N436': 'LP231'}
xtick_labels = [f"{DISP[s]}\n(n={sample_n[s]:,})" for s in prop.index]
ax.set_xticks(x_positions)
ax.set_xticklabels(xtick_labels, fontsize=7)
ax.set_ylabel('Proportion of cells', fontsize=8)
ax.set_xlabel('')
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=7)

# Spines
for spine_name in ['top', 'right']:
    ax.spines[spine_name].set_visible(False)
for spine_name in ['left', 'bottom']:
    ax.spines[spine_name].set_linewidth(0.5)
    ax.spines[spine_name].set_color('#888888')

ax.tick_params(axis='both', length=2, width=0.5, color='#888888')

# Legend
legend_handles = [
    plt.Rectangle((0,0), 1, 1, facecolor=CLUSTER_PALETTE[c], edgecolor='none')
    for c in cluster_order
]
legend_labels = [f'{c}' if c != '8' else f'{c} (QC)' for c in cluster_order]

ax.legend(
    legend_handles, legend_labels,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.35,
    title='Cluster',
    title_fontsize=7,
    fontsize=6.5,
    handlelength=1.2,
    handleheight=1.0,
)

ax.set_title('Cluster composition', fontsize=8.5, fontweight='bold', loc='left', pad=4)

# Panel label
fig.text(0.02, 0.96, 'E', fontsize=11, fontweight='bold', va='top', ha='left')

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

pdf_path = os.path.join(OUTDIR_PDF, 'fig1E_stacked_bar.pdf')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
print(f"\nSaved PDF: {pdf_path}")

tiff_path = os.path.join(OUTDIR_TIFF, 'fig1E_stacked_bar.tiff')
fig.savefig(tiff_path, format='tiff', dpi=600,
             bbox_inches='tight', pad_inches=0.05,
             pil_kwargs={'compression': 'tiff_lzw'})
print(f"Saved TIFF: {tiff_path}")

png_path = os.path.join(OUTDIR_PNG, 'fig1E_stacked_bar_preview.png')
fig.savefig(png_path, format='png', dpi=200,
             bbox_inches='tight', pad_inches=0.05)
print(f"Saved PNG preview: {png_path}")

for p in [pdf_path, tiff_path, png_path]:
    sz = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):45s}  {sz:6.2f} MB")

plt.close()
print("\nDONE.")
