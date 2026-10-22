"""
Figure S2 — Independent NMF decomposition of velocity dynamics.

Multi-track heatmap visualizing the 4 NMF programs (k=4) across 300
latent-time bins, with sample / cluster / latent-time / fate_3 annotation
tracks. Same visual language as Figure 3A consensus driver heatmap.

Biological interpretation (independent of likelihood-based selection):
  - NMF1: housekeeping/metabolic (S100A6, S100A4, RPL12, TMSB4X, LDHA)
  - NMF2: stress/early TNF response (S100A6, ANKRD1, IER3, CCN1, MT2A)
  - NMF3: cell cycle/proliferation (UBE2C, TOP2A, CKS2, HMGB2, CCNB1)
  - NMF4: housekeeping (resistance-associated; S100A6, S100A4, IGFBP4, CTSD)

The TNF/AP-1 driver signature identified through likelihood-based
selection (main Figures 3-4) is distributed across multiple NMF programs
(primarily NMF2 and NMF3) rather than forming a single factor.

Author: Marko Zivanovic
Output: PDF + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import spearmanr

# ============================================================
# GLOBAL STYLE
# ============================================================
plt.rcParams.update({
    'font.family':       'Arial',
    'font.size':         7,
    'axes.linewidth':    0.5,
    'axes.titlesize':    8,
    'axes.labelsize':    7,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'legend.fontsize':   6.5,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'svg.fonttype':      'none',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'text.color':        'black',
    'axes.edgecolor':    'black',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
})

# ============================================================
# CONSTANTS
# ============================================================
H5AD_PATH = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca__PLUS_nmf4_bins.h5ad"
NMF_TSV   = DATA_ROOT + "/velocity/03_outputs/tabular/latent_time_programs_nmf/nmf_scores_per_cell_4comp_300pos_300neg.tsv"
TOP40_CSV = DATA_ROOT + "/velocity/03_outputs/tabular/latent_time_programs_nmf/nmf_top40_genes_per_program.csv"

N_BINS = 300

SAMPLE_PALETTE = {'C231': '#4575B4', 'N231': '#FDB863', 'N436': '#B2182B'}
CLUSTER_PALETTE = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
    '#8C564B', '#E377C2', '#9E9E9E', '#BCBD22', '#17BECF',
    '#AEC7E8', '#FFBB78',
]

# Program biological labels
PROGRAM_LABELS = {
    'lt_nmf1': 'NMF1\nHousekeeping/metabolic',
    'lt_nmf2': 'NMF2\nStress / early response',
    'lt_nmf3': 'NMF3\nCell cycle / proliferation',
    'lt_nmf4': 'NMF4\nHousekeeping (resistance-assoc.)',
}

# ============================================================
# LOAD DATA
# ============================================================
print("=== Loading h5ad ===")
ad = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {ad.shape}")
print(f"obs has: sample, latent_time, fate_0, fate_3, lt_nmf2, lt_nmf4, nmf4_bin")

print("\n=== Loading per-cell NMF TSV (all 4 programs) ===")
df_nmf = pd.read_csv(NMF_TSV, sep='\t')
print(f"NMF TSV rows: {len(df_nmf)}, columns: {list(df_nmf.columns)}")

if len(df_nmf) != ad.n_obs:
    raise ValueError(f"NMF TSV row count ({len(df_nmf)}) != h5ad cell count ({ad.n_obs})")

# Sanity: same sample/latent_time order?
mismatch_lt = np.abs(df_nmf['latent_time'].values - ad.obs['latent_time'].values).max()
print(f"Max latent_time mismatch (TSV vs h5ad): {mismatch_lt:.6e}")
if mismatch_lt > 1e-6:
    print("WARNING: TSV and h5ad latent_time do not align by index. Check ordering.")
else:
    print("OK: TSV row order matches h5ad cell order.")

# Build per-cell DataFrame
df_cells = pd.DataFrame({
    'sample':      ad.obs['sample'].values,
    'latent_time': ad.obs['latent_time'].values,
    'fate_0':      ad.obs['fate_0'].values,
    'fate_3':      ad.obs['fate_3'].values,
    'lt_nmf1':     df_nmf['lt_nmf1'].values,
    'lt_nmf2':     df_nmf['lt_nmf2'].values,
    'lt_nmf3':     df_nmf['lt_nmf3'].values,
    'lt_nmf4':     df_nmf['lt_nmf4'].values,
})

# Cluster column — h5ad doesn't have it directly; check leiden/louvain in obs
cluster_col = None
for cand in ['leiden', 'louvain', 'seurat_clusters', 'clusters', 'cluster']:
    if cand in ad.obs.columns:
        cluster_col = cand
        break
if cluster_col is not None:
    df_cells['cluster'] = ad.obs[cluster_col].values
    print(f"Cluster column found: '{cluster_col}'")
else:
    df_cells['cluster'] = 'na'
    print("WARNING: no cluster column in obs. Annotation track will be skipped.")

# ============================================================
# VALIDATE NMF VS LT ORDERING (Spearman)
# ============================================================
print("\n=== Spearman rho: NMF score vs latent_time ===")
nmf_lt_corrs = {}
for prog in ['lt_nmf1', 'lt_nmf2', 'lt_nmf3', 'lt_nmf4']:
    rho, p = spearmanr(df_cells[prog].values, df_cells['latent_time'].values)
    nmf_lt_corrs[prog] = rho
    print(f"  {prog}: rho = {rho:+.3f}  (p = {p:.2e})")

# Order programs by LT correlation (most positive at top, most negative at bottom)
# This puts late-LT programs (resistance-associated) at top, early-LT (QC) at bottom.
program_order = sorted(['lt_nmf1', 'lt_nmf2', 'lt_nmf3', 'lt_nmf4'],
                       key=lambda p: -nmf_lt_corrs[p])
print(f"\nProgram order (top -> bottom by LT rho): {program_order}")

# ============================================================
# BIN BY LATENT_TIME (300 bins)
# ============================================================
print(f"\n=== Binning {ad.n_obs} cells into {N_BINS} latent-time bins ===")
df_cells_sorted = df_cells.sort_values('latent_time').reset_index(drop=True)
df_cells_sorted['lt_bin'] = pd.cut(df_cells_sorted['latent_time'],
                                    bins=N_BINS, labels=False, include_lowest=True)

# Per-bin mean NMF scores (4 x 300 matrix)
bin_scores = np.zeros((4, N_BINS))
for i, prog in enumerate(program_order):
    grouped = df_cells_sorted.groupby('lt_bin')[prog].mean()
    # Reindex to ensure all 300 bins are present (NaN if empty)
    bin_scores[i, :] = grouped.reindex(range(N_BINS)).values

# Z-score each program (row) for visualization
bin_scores_z = np.zeros_like(bin_scores)
for i in range(4):
    mu = np.nanmean(bin_scores[i, :])
    sd = np.nanstd(bin_scores[i, :])
    bin_scores_z[i, :] = (bin_scores[i, :] - mu) / sd if sd > 0 else 0
print(f"Heatmap matrix shape: {bin_scores_z.shape} (4 programs x {N_BINS} bins)")
print(f"Z-score range: [{np.nanmin(bin_scores_z):.2f}, {np.nanmax(bin_scores_z):.2f}]")

# Per-bin annotation tracks: sample (mode), latent_time (mean), fate_3 (mean)
ann_sample = df_cells_sorted.groupby('lt_bin')['sample'].agg(
    lambda s: s.mode().iloc[0] if len(s) > 0 else 'C231'
).reindex(range(N_BINS)).values
ann_lt = df_cells_sorted.groupby('lt_bin')['latent_time'].mean().reindex(range(N_BINS)).values
ann_fate3 = df_cells_sorted.groupby('lt_bin')['fate_3'].mean().reindex(range(N_BINS)).values

if cluster_col is not None:
    ann_cluster = df_cells_sorted.groupby('lt_bin')['cluster'].agg(
        lambda s: s.mode().iloc[0] if len(s) > 0 else None
    ).reindex(range(N_BINS)).values
else:
    ann_cluster = None

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 17.5
FIG_H_CM = 8.0
CM_TO_INCH = 1 / 2.54

fig = plt.figure(figsize=(FIG_W_CM * CM_TO_INCH, FIG_H_CM * CM_TO_INCH))

# Layout: 4 annotation tracks (1 row each) + 1 heatmap row
# height_ratios: small for annotation, large for heatmap
n_tracks = 4 if ann_cluster is not None else 3
track_h = 0.4
heatmap_h = 5.0
height_ratios = [track_h] * n_tracks + [heatmap_h]

gs = fig.add_gridspec(
    nrows=n_tracks + 1, ncols=2,
    width_ratios=[20, 0.6],
    height_ratios=height_ratios,
    hspace=0.10, wspace=0.04,
    left=0.20, right=0.94, top=0.92, bottom=0.10,
)

# ============================================================
# ANNOTATION TRACKS (above heatmap)
# ============================================================
# Track 1: sample
ax_s = fig.add_subplot(gs[0, 0])
sample_codes = np.array([list(SAMPLE_PALETTE.keys()).index(s) if s in SAMPLE_PALETTE else 0
                         for s in ann_sample])
sample_cmap = ListedColormap([SAMPLE_PALETTE[s] for s in SAMPLE_PALETTE.keys()])
ax_s.imshow(sample_codes.reshape(1, -1), aspect='auto',
            cmap=sample_cmap, vmin=0, vmax=len(SAMPLE_PALETTE)-1,
            interpolation='nearest')
ax_s.set_yticks([0]); ax_s.set_yticklabels(['Sample'], fontsize=6.0)
ax_s.set_xticks([])
for sp in ax_s.spines.values():
    sp.set_linewidth(0.4)

# Track 2: cluster (if present)
row_idx = 1
if ann_cluster is not None:
    ax_c = fig.add_subplot(gs[row_idx, 0])
    unique_clusters = sorted([c for c in pd.unique(ann_cluster) if c is not None],
                             key=lambda x: int(x) if str(x).isdigit() else 99)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_codes = np.array([cluster_to_idx[c] if c in cluster_to_idx else -1
                              for c in ann_cluster])
    cluster_cmap_used = ListedColormap(CLUSTER_PALETTE[:len(unique_clusters)])
    ax_c.imshow(cluster_codes.reshape(1, -1), aspect='auto',
                cmap=cluster_cmap_used, vmin=0, vmax=len(unique_clusters)-1,
                interpolation='nearest')
    ax_c.set_yticks([0]); ax_c.set_yticklabels(['Cluster'], fontsize=6.0)
    ax_c.set_xticks([])
    for sp in ax_c.spines.values():
        sp.set_linewidth(0.4)
    row_idx += 1

# Track 3: latent_time (continuous viridis)
ax_lt = fig.add_subplot(gs[row_idx, 0])
ax_lt.imshow(ann_lt.reshape(1, -1), aspect='auto',
             cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
ax_lt.set_yticks([0]); ax_lt.set_yticklabels(['Latent time'], fontsize=6.0)
ax_lt.set_xticks([])
for sp in ax_lt.spines.values():
    sp.set_linewidth(0.4)
row_idx += 1

# Track 4: fate_3 (continuous Reds)
ax_f3 = fig.add_subplot(gs[row_idx, 0])
ax_f3.imshow(ann_fate3.reshape(1, -1), aspect='auto',
             cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
ax_f3.set_yticks([0]); ax_f3.set_yticklabels(['Fate 3'], fontsize=6.0)
ax_f3.set_xticks([])
for sp in ax_f3.spines.values():
    sp.set_linewidth(0.4)
row_idx += 1

# ============================================================
# HEATMAP — 4 programs x 300 bins
# ============================================================
ax_hm = fig.add_subplot(gs[row_idx, 0])
im = ax_hm.imshow(bin_scores_z, aspect='auto',
                  cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest',
                  rasterized=True)

# Y-axis: program labels with biological annotation
ax_hm.set_yticks(range(4))
ax_hm.set_yticklabels([PROGRAM_LABELS[p] for p in program_order],
                      fontsize=6.5)
# X-axis: bin index, label only at start/middle/end
ax_hm.set_xticks([0, N_BINS//2, N_BINS-1])
ax_hm.set_xticklabels(['early\n(LT=0)', 'mid', 'late\n(LT=1)'], fontsize=6.0)
ax_hm.set_xlabel('Latent-time bin', fontsize=7, labelpad=2)
for sp in ax_hm.spines.values():
    sp.set_linewidth(0.4)

# Colorbar (right column)
ax_cbar = fig.add_subplot(gs[row_idx, 1])
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.set_label('z-score (per program)', fontsize=6.5, labelpad=2)
cbar.ax.tick_params(labelsize=6, length=2, width=0.4)
cbar.outline.set_linewidth(0.4)

# Panel label
fig.text(0.005, 0.97, '', fontsize=11, fontweight='bold',
         va='top', ha='left', color='black')  # No panel label for supplementary

# ============================================================
# SAVE
# ============================================================
OUTDIR_BASE = DATA_ROOT + '/figures'
OUTDIR_PDF  = os.path.join(OUTDIR_BASE, 'pdf')
OUTDIR_TIFF = os.path.join(OUTDIR_BASE, 'tiff')
OUTDIR_PNG  = os.path.join(OUTDIR_BASE, 'png')
os.makedirs(OUTDIR_PDF,  exist_ok=True)
os.makedirs(OUTDIR_TIFF, exist_ok=True)
os.makedirs(OUTDIR_PNG,  exist_ok=True)

pdf_path  = os.path.join(OUTDIR_PDF,  'Figure_S2_nmf.pdf')
tiff_path = os.path.join(OUTDIR_TIFF, 'Figure_S2_nmf.tiff')
png_path  = os.path.join(OUTDIR_PNG,  'Figure_S2_nmf_preview.png')

saved = []
try:
    fig.savefig(png_path, format='png', dpi=200, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    print(f"\nSaved PNG: {png_path}")
    saved.append(png_path)
except Exception as e:
    print(f"\nPNG save FAILED: {e}")

try:
    fig.savefig(tiff_path, format='tiff', dpi=600, bbox_inches='tight',
                pad_inches=0.05, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f"Saved TIFF: {tiff_path}")
    saved.append(tiff_path)
except Exception as e:
    print(f"TIFF save FAILED: {e}")

try:
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    print(f"Saved PDF: {pdf_path}")
    saved.append(pdf_path)
except Exception as e:
    print(f"PDF save FAILED: {e}")

print()
for p in saved:
    sz = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):50s}  {sz:6.2f} MB")

plt.close()
print("\nDONE.")
