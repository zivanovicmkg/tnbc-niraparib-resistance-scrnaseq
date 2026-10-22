"""
Figure 2 master composite — RNA velocity & fate analysis.
(script numbering; = manuscript Figure 4)

v4 (20.10.2026): NP231/HP231/LP231 names + colors, new cluster palette,
Cliff's delta printed for E/F. Outputs Figure_2_v4.* (old Figure_2.* untouched).

Layout A (clean 2-column alignment, ~17.5 x 21 cm):

  Row 1:  A (velocity stream)        |  C (latent_time + macrostates)
          ~8.5 x 7 cm                |  ~8.5 x 7 cm

  Row 2:  B (per-sample, 3 sub-panels horizontal, full width)
          ~17 x 5.5 cm

  Row 3:  D (fate prob, 2 sub-panels)|  E (length boxplot) | F (conf. boxplot)
          ~8.5 x 6.5 cm              |  ~4.25 x 6.5 cm     | ~4.25 x 6.5 cm

Key technical decisions:
  - All scatter / stream / fate-probability layers rasterized=True for
    smaller TIFF; text, axes, brackets, boxplots remain vector.
  - velocity_length computed as np.linalg.norm(layers['velocity'], axis=1)
    on raw velocity layer — NOT scv.tl.velocity_confidence (which gives
    different smoothed values).
  - velocity_confidence computed via scv.tl.velocity_confidence(ad).
  - Cluster 8 (QC artifact) grayed out (alpha 0.25) in A-D, EXCLUDED from
    boxplot statistics in E-F.
  - No scv.set_figure_params (would inject dark stylesheet).
  - rcParams white background forced explicitly.

Author: Marko Zivanovic
Output: PDF + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from scipy.stats import mannwhitneyu
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
# PALETTES
# ============================================================
SAMPLE_PALETTE = {'C231': '#0000FF', 'N231': '#FF8000', 'N436': '#00C000'}
DISP = {'C231': 'NP231', 'N231': 'HP231', 'N436': 'LP231'}
SAMPLES = ['C231', 'N231', 'N436']
SAMPLE_TITLES = {
    'C231': 'NP231 (parental)',
    'N231': 'HP231 (high dose)',
    'N436': 'LP231 (low dose)',
}

CLUSTER_PALETTE = {
    '2': '#0000FF', '3': '#5B8DEF', '4': '#9DBBFF',      # NP231 family (2 dominant)
    '1': '#FF8000', '5': '#FFC266',                      # HP231 family (1 dominant)
    '0': '#00C000', '6': '#005A00',                      # LP231 family (0 dominant, 6 = G2/M)
    '7': '#8C564B', '9': '#D4A6C8',                      # mixed
    '10': '#E6007E', '11': '#E41A1C',                    # rare, highlighted
    '8': '#BDBDBD',                                      # QC
}
CLUSTER_ORDER = [str(i) for i in range(12)]
PLOT_ORDER = ['8', '0', '2', '1', '3', '4', '5', '6', '9', '7', '11', '10']

BG_GREY = '#E0E0E0'

# ============================================================
# LOAD DATA
# ============================================================
CR_PATH = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
S1_FATE_PATH = DATA_ROOT + "/velocity/03_outputs/fate0_analysis/stage1_with_cellrank_fates.h5ad"

print(f"Loading CellRank: {CR_PATH}")
ad = sc.read_h5ad(CR_PATH)

print(f"Loading Stage1 fate: {S1_FATE_PATH}")
ad_fate = sc.read_h5ad(S1_FATE_PATH)

# Barcode normalization to transfer leiden
def normalize_cr(bc):
    sample = bc.split('_')[0]
    barcode = bc.split(':')[1].rstrip('x')
    return f"{barcode}-{sample}"

def normalize_s1(bc):
    parts = bc.split('-')
    return f"{parts[0]}-{parts[-1]}"

cr_keys = pd.Index([normalize_cr(bc) for bc in ad.obs_names])
s1_keys = pd.Index([normalize_s1(bc) for bc in ad_fate.obs_names])
s1_idx_map = pd.Series(index=s1_keys, data=range(len(s1_keys)))
mapped_s1_idx = np.array([s1_idx_map[k] for k in cr_keys])

ad.obs['leiden'] = pd.Categorical(
    ad_fate.obs['leiden'].values[mapped_s1_idx].astype(str),
    categories=CLUSTER_ORDER,
)
ad.obs['sample'] = pd.Categorical(
    [bc.split('_')[0] for bc in ad.obs_names],
    categories=SAMPLES,
)

leiden_arr = ad.obs['leiden'].astype(str).values
sample_arr = ad.obs['sample'].astype(str).values
mask_c8 = leiden_arr == '8'
mask_other = ~mask_c8

print(f"  Cells: {ad.n_obs}")
print(f"  Cluster 8: {mask_c8.sum()}, others: {mask_other.sum()}")
print(f"  Sample counts: {ad.obs['sample'].value_counts().to_dict()}")

# ============================================================
# COMPUTE: velocity_length (raw norm), velocity_confidence (scvelo)
# ============================================================
print("\nComputing velocity_length = norm(layers['velocity'])...")
vel = ad.layers['velocity']
if hasattr(vel, 'toarray'):
    vel = vel.toarray()
vel = np.nan_to_num(vel)
vel_length = np.linalg.norm(vel, axis=1)
print(f"  velocity_length median: {np.median(vel_length):.3f}")

print("Computing velocity_confidence via scvelo...")
if 'velocity_confidence' not in ad.obs.columns:
    scv.tl.velocity_confidence(ad)
vel_conf = ad.obs['velocity_confidence'].values
print(f"  velocity_confidence median: {np.median(vel_conf):.3f}")

# Ensure velocity_graph and velocity_umap exist for stream plots
if 'velocity_graph' not in ad.uns:
    print("Computing velocity_graph...")
    scv.tl.velocity_graph(ad)
if 'velocity_umap' not in ad.obsm:
    print("Computing velocity_umap embedding...")
    scv.tl.velocity_embedding(ad, basis='umap')

umap = ad.obsm['X_umap']
xpad = (umap[:, 0].max() - umap[:, 0].min()) * 0.03
ypad = (umap[:, 1].max() - umap[:, 1].min()) * 0.03
xlim = (umap[:, 0].min() - xpad, umap[:, 0].max() + xpad)
ylim = (umap[:, 1].min() - ypad, umap[:, 1].max() + ypad)

# ============================================================
# MACROSTATE CENTROIDS
# ============================================================
ms = ad.obs['macrostates_fwd']
ms_cats = list(ms.cat.categories)
centroids = {}
for cat in ms_cats:
    mk = (ms == cat).values
    if mk.sum() > 0:
        centroids[cat] = umap[mk].mean(axis=0)
term_cats = list(ad.obs['term_states_fwd'].cat.categories)
print(f"  Macrostates: {ms_cats}, terminal: {term_cats}")

# ============================================================
# HELPER: rasterize stream artists (for PDF safety + smaller TIFF)
# ============================================================
def rasterize_stream_artists(ax):
    n = 0
    for artist in ax.get_children():
        if isinstance(artist, FancyArrowPatch):
            artist.set_rasterized(True); n += 1
        elif isinstance(artist, LineCollection):
            try:
                if len(artist.get_segments()) > 5:
                    artist.set_rasterized(True); n += 1
            except Exception:
                pass
    return n

# ============================================================
# FIGURE — 17.5 x 21 cm, 3 rows
# ============================================================
FIG_W_CM = 17.5
FIG_H_CM = 21.0
CM_TO_INCH = 1 / 2.54

fig = plt.figure(figsize=(FIG_W_CM * CM_TO_INCH, FIG_H_CM * CM_TO_INCH))

# Master gridspec: 3 rows (top=A+C, mid=B, bot=D+E+F)
# Heights: A/C row ~7cm, B row ~5.5cm, D/E/F row ~6.5cm
# Plus small gaps between rows
gs_master = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[7.0, 5.5, 6.5],
    hspace=0.32,
    left=0.05, right=0.98, top=0.96, bottom=0.04,
)

# Row 1: A (left half) | C (right half)
gs_row1 = gs_master[0].subgridspec(
    nrows=1, ncols=2,
    width_ratios=[1, 1],
    wspace=0.20,
)

# Row 2: B (3 sub-panels horizontal, full width) + slim legend column on right
# We use 4 sub-cols: 3 for panels, 1 narrow for shared legend
gs_row2 = gs_master[1].subgridspec(
    nrows=1, ncols=4,
    width_ratios=[1, 1, 1, 0.18],
    wspace=0.12,
)

# Row 3: D (left half) | E (1/4) | F (1/4)
# D itself splits into 2 sub-panels (fate3, fate0) + 2 sub-colorbars
gs_row3 = gs_master[2].subgridspec(
    nrows=1, ncols=3,
    width_ratios=[2.0, 1.0, 1.0],
    wspace=0.30,
)

# ============================================================
# PANEL A — UMAP velocity stream (overall)
# ============================================================
print("\n=== Panel A: velocity stream overall ===")
# A has its own internal split: scatter+stream main area + small legend column
gs_A = gs_row1[0, 0].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.18], wspace=0.05)
ax_A = fig.add_subplot(gs_A[0, 0])
ax_A_leg = fig.add_subplot(gs_A[0, 1])
ax_A_leg.axis('off')

# Layer 1: scatter by cluster
for cluster in PLOT_ORDER:
    mk = leiden_arr == cluster
    if mk.sum() == 0: continue
    alpha = 0.3 if cluster == '8' else 0.55
    s_size = 1.2 if cluster in ['10', '11'] else 0.6
    ax_A.scatter(umap[mk, 0], umap[mk, 1],
                 c=CLUSTER_PALETTE[cluster],
                 s=s_size, alpha=alpha, linewidths=0,
                 rasterized=True, zorder=1)

# Layer 2: stream
try:
    scv.pl.velocity_embedding_stream(
        ad, basis='umap', color='leiden',
        ax=ax_A, show=False, size=0, alpha=0,
        legend_loc='none', title='',
        arrow_size=0.8, arrow_color='black',
        linewidth=0.6, density=1.2, min_mass=1.0,
    )
except Exception:
    scv.pl.velocity_embedding_stream(
        ad, basis='umap', color=None,
        ax=ax_A, show=False, size=0, alpha=0,
        legend_loc='none', title='',
        arrow_size=0.8, arrow_color='black',
        linewidth=0.6, density=1.2, min_mass=1.0,
    )
rasterize_stream_artists(ax_A)
ax_A.set_facecolor('white')

ax_A.set_xlim(xlim); ax_A.set_ylim(ylim)
ax_A.set_xticks([]); ax_A.set_yticks([])
ax_A.set_xlabel('UMAP 1', fontsize=7, labelpad=2)
ax_A.set_ylabel('UMAP 2', fontsize=7, labelpad=2)
for sp in ax_A.spines.values():
    sp.set_linewidth(0.5); sp.set_color('#888888')
ax_A.set_title('RNA velocity', fontsize=8, fontweight='bold', loc='left', pad=3)

# Cluster legend in side column
leg_handles = []
for cluster in CLUSTER_ORDER:
    label = f'{cluster}*' if cluster == '8' else f'{cluster}'
    leg_handles.append(plt.Line2D(
        [0], [0], marker='o', color='none',
        markerfacecolor=CLUSTER_PALETTE[cluster],
        markersize=4, markeredgewidth=0, label=label,
    ))
leg = ax_A_leg.legend(
    handles=leg_handles, loc='center left',
    bbox_to_anchor=(0, 0.5),
    frameon=False, handletextpad=0.2, labelspacing=0.25,
    borderaxespad=0, fontsize=5.8, ncol=2, columnspacing=0.5,
    title='Cluster', title_fontsize=6.5,
)
for t in leg.get_texts(): t.set_color('black')
leg.get_title().set_color('black')

# ============================================================
# PANEL C — UMAP latent_time + macrostates
# ============================================================
print("=== Panel C: latent_time ===")
# C internal split: main UMAP + horizontal colorbar below
gs_C = gs_row1[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[20, 1], hspace=0.22)
ax_C = fig.add_subplot(gs_C[0, 0])
cax_C = fig.add_subplot(gs_C[1, 0])

lt = ad.obs['latent_time'].values

# Layer 1a: cluster 8 grey background
ax_C.scatter(umap[mask_c8, 0], umap[mask_c8, 1],
             c='#9E9E9E', s=0.7, alpha=0.25, linewidths=0,
             rasterized=True, zorder=1)

# Layer 1b: non-cluster-8 by latent_time
norm_lt = Normalize(vmin=0.0, vmax=1.0)
idx_other = np.where(mask_other)[0]
order_within = np.argsort(lt[idx_other])
order = idx_other[order_within]
ax_C.scatter(umap[order, 0], umap[order, 1],
             c=lt[order], cmap='viridis', norm=norm_lt,
             s=0.7, alpha=0.7, linewidths=0,
             rasterized=True, zorder=2)

# Layer 2: macrostate centroids
for cat in ms_cats:
    if cat not in centroids: continue
    cx, cy = centroids[cat]
    is_terminal = cat in term_cats
    is_qc = (cat == '0')
    if is_qc:
        ax_C.scatter([cx], [cy], s=120, c='black', edgecolors='white',
                     linewidths=1.8, marker='X', zorder=10)
        ax_C.text(cx, cy - 0.55, f'M{cat}', fontsize=6, fontweight='bold',
                  color='black', ha='center', va='top', zorder=11,
                  bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                            edgecolor='none', alpha=0.85))
    elif is_terminal:
        ax_C.scatter([cx], [cy], s=120, c='black', edgecolors='white',
                     linewidths=1.0, marker='o', zorder=10)
        ax_C.text(cx, cy, f'M{cat}*', fontsize=6, fontweight='bold',
                  color='white', ha='center', va='center', zorder=11)
    else:
        ax_C.scatter([cx], [cy], s=95, c='white', edgecolors='black',
                     linewidths=0.9, marker='o', zorder=10)
        ax_C.text(cx, cy, f'M{cat}', fontsize=6, fontweight='bold',
                  color='black', ha='center', va='center', zorder=11)

ax_C.set_xlim(xlim); ax_C.set_ylim(ylim)
ax_C.set_xticks([]); ax_C.set_yticks([])
ax_C.set_xlabel('UMAP 1', fontsize=7, labelpad=2)
ax_C.set_ylabel('UMAP 2', fontsize=7, labelpad=2)
for sp in ax_C.spines.values():
    sp.set_linewidth(0.5); sp.set_color('#888888')
ax_C.set_title('Latent time + macrostates', fontsize=8, fontweight='bold', loc='left', pad=3)

# Mini-legend inside upper right
leg_C_handles = [
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor='black', markeredgecolor='white',
               markeredgewidth=0.9, markersize=5.5, label='Terminal'),
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor='white', markeredgecolor='black',
               markeredgewidth=0.9, markersize=5.5, label='Transient'),
    plt.Line2D([0], [0], marker='X', color='none',
               markerfacecolor='black', markeredgecolor='white',
               markeredgewidth=0.7, markersize=6.5, label='QC'),
]
leg_C = ax_C.legend(
    handles=leg_C_handles, loc='upper right',
    bbox_to_anchor=(0.99, 0.99),
    frameon=True, facecolor='white', edgecolor='#888888',
    framealpha=0.92, handletextpad=0.3, labelspacing=0.2,
    borderpad=0.3, fontsize=5.5,
)
leg_C.get_frame().set_linewidth(0.4)
for t in leg_C.get_texts(): t.set_color('black')

# Colorbar
sm_lt = ScalarMappable(norm=norm_lt, cmap='viridis'); sm_lt.set_array([])
cb_C = fig.colorbar(sm_lt, cax=cax_C, orientation='horizontal')
cb_C.set_label('Latent time', fontsize=6.5, labelpad=1.5)
cb_C.ax.tick_params(labelsize=6, length=2, pad=1)
cb_C.outline.set_linewidth(0.4)
cb_C.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

# ============================================================
# PANEL B — per-sample velocity stream
# ============================================================
print("=== Panel B: per-sample stream ===")
ax_B_list = [fig.add_subplot(gs_row2[0, i]) for i in range(3)]
ax_B_leg = fig.add_subplot(gs_row2[0, 3])
ax_B_leg.axis('off')

for ax_b, sample in zip(ax_B_list, SAMPLES):
    print(f"  Panel B — {sample}")
    mask_s = (sample_arr == sample)
    n_cells = mask_s.sum()

    # Layer 0: full UMAP grey background
    ax_b.scatter(umap[~mask_s, 0], umap[~mask_s, 1],
                 c=BG_GREY, s=0.4, alpha=0.35, linewidths=0,
                 rasterized=True, zorder=0)
    # Layer 1: this sample by cluster
    for cluster in PLOT_ORDER:
        mk = mask_s & (leiden_arr == cluster)
        if mk.sum() == 0: continue
        alpha = 0.3 if cluster == '8' else 0.65
        s_size = 1.2 if cluster in ['10', '11'] else 0.7
        ax_b.scatter(umap[mk, 0], umap[mk, 1],
                     c=CLUSTER_PALETTE[cluster],
                     s=s_size, alpha=alpha, linewidths=0,
                     rasterized=True, zorder=1)

    # Layer 2: per-sample stream
    ad_sub = ad[mask_s].copy()
    try:
        scv.pl.velocity_embedding_stream(
            ad_sub, basis='umap', color='leiden',
            ax=ax_b, show=False, size=0, alpha=0,
            legend_loc='none', title='',
            arrow_size=0.8, arrow_color='black',
            linewidth=0.55, density=1.2, min_mass=1.0,
        )
    except Exception:
        scv.pl.velocity_embedding_stream(
            ad_sub, basis='umap', color=None,
            ax=ax_b, show=False, size=0, alpha=0,
            legend_loc='none', title='',
            arrow_size=0.8, arrow_color='black',
            linewidth=0.55, density=1.2, min_mass=1.0,
        )
    rasterize_stream_artists(ax_b)
    ax_b.set_facecolor('white')

    ax_b.set_xlim(xlim); ax_b.set_ylim(ylim)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#888888')
    ax_b.set_title(f"{SAMPLE_TITLES[sample]}\nn = {n_cells:,}",
                   fontsize=7.5, fontweight='bold', loc='left', pad=3)

ax_B_list[0].set_xlabel('UMAP 1', fontsize=7, labelpad=2)
ax_B_list[0].set_ylabel('UMAP 2', fontsize=7, labelpad=2)

# Shared legend for panel B
leg_B = ax_B_leg.legend(
    handles=leg_handles, loc='center left',
    bbox_to_anchor=(0, 0.5),
    frameon=False, handletextpad=0.2, labelspacing=0.25,
    borderaxespad=0, fontsize=5.8, ncol=1,
    title='Cluster', title_fontsize=6.5,
)
for t in leg_B.get_texts(): t.set_color('black')
leg_B.get_title().set_color('black')

# ============================================================
# PANEL D — fate probability side-by-side
# ============================================================
print("=== Panel D: fate probability ===")
# D split into 2 sub-panels horizontally + 2 sub-colorbars below
gs_D = gs_row3[0, 0].subgridspec(
    nrows=2, ncols=2,
    height_ratios=[20, 1], width_ratios=[1, 1],
    wspace=0.20, hspace=0.18,
)
ax_D_left  = fig.add_subplot(gs_D[0, 0])
ax_D_right = fig.add_subplot(gs_D[0, 1])
cax_D_left  = fig.add_subplot(gs_D[1, 0])
cax_D_right = fig.add_subplot(gs_D[1, 1])

fate3 = ad.obs['fate_3'].values
fate0 = ad.obs['fate_0'].values

def draw_fate_subpanel(ax, cax, fate_arr, cmap, title, centroid_xy, marker_kind):
    ax.scatter(umap[mask_c8, 0], umap[mask_c8, 1],
               c='#9E9E9E', s=0.7, alpha=0.25, linewidths=0,
               rasterized=True, zorder=1)
    norm = Normalize(vmin=0.0, vmax=1.0)
    idx_o = np.where(mask_other)[0]
    ow = np.argsort(fate_arr[idx_o])
    od = idx_o[ow]
    ax.scatter(umap[od, 0], umap[od, 1],
               c=fate_arr[od], cmap=cmap, norm=norm,
               s=0.7, alpha=0.75, linewidths=0,
               rasterized=True, zorder=2)
    cx, cy = centroid_xy
    if marker_kind == 'star':
        ax.scatter([cx], [cy], s=140, c='black', edgecolors='white',
                   linewidths=1.0, marker='*', zorder=10)
        ax.text(cx, cy - 0.6, 'M3', fontsize=6, fontweight='bold',
                color='black', ha='center', va='top', zorder=11,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none', alpha=0.85))
    else:
        ax.scatter([cx], [cy], s=110, c='black', edgecolors='white',
                   linewidths=1.7, marker='X', zorder=10)
        ax.text(cx, cy - 0.5, 'M0', fontsize=6, fontweight='bold',
                color='black', ha='center', va='top', zorder=11,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none', alpha=0.85))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#888888')
    ax.set_title(title, fontsize=7.5, fontweight='bold', loc='left', pad=3)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0.0, 0.5, 1.0])
    return cb

cb_D_l = draw_fate_subpanel(ax_D_left, cax_D_left, fate3, plt.cm.Purples,
                             'Fate 3 (resistant terminal)',
                             centroids['3'], 'star')
cb_D_l.set_label('fate_3 prob.', fontsize=6.5, labelpad=1.5)

cb_D_r = draw_fate_subpanel(ax_D_right, cax_D_right, fate0, plt.cm.RdPu,
                             'Fate 0 (QC artifact)',
                             centroids['0'], 'cross')
cb_D_r.set_label('fate_0 prob.', fontsize=6.5, labelpad=1.5)

ax_D_left.set_ylabel('UMAP 2', fontsize=7, labelpad=2)
ax_D_left.set_xlabel('UMAP 1', fontsize=7, labelpad=2)

# ============================================================
# PANEL E — boxplot velocity_length per sample
# ============================================================
print("=== Panel E: velocity_length boxplot ===")
ax_E = fig.add_subplot(gs_row3[0, 1])

mask_keep = leiden_arr != '8'
data_E = {}
stats_E = {}
for s in SAMPLES:
    mk = mask_keep & (sample_arr == s)
    arr = vel_length[mk]
    data_E[s] = arr
    q25, q75 = np.percentile(arr, [25, 75])
    stats_E[s] = {'n': len(arr), 'median': np.median(arr),
                  'mean': np.mean(arr), 'iqr_low': q25, 'iqr_high': q75}
    print(f"  {s}: n={len(arr)} median={np.median(arr):.3f} mean={np.mean(arr):.3f}")

def cliffs_delta(a, b):
    u, _ = mannwhitneyu(a, b, alternative='two-sided')
    return 2.0 * u / (len(a) * len(b)) - 1.0   # +: a > b ; |d|<0.147 negligible, <0.33 small, <0.474 medium

def fmt_p(p):
    if p < 0.0001:   stars = '****'
    elif p < 0.001:  stars = '***'
    elif p < 0.01:   stars = '**'
    elif p < 0.05:   stars = '*'
    else:            stars = 'ns'
    if p < 1e-30:    txt = 'p < 1e-30'
    elif p < 1e-4:   txt = f'p = {p:.1e}'
    elif p < 0.001:  txt = f'p = {p:.2e}'
    elif p < 0.05:   txt = f'p = {p:.3f}'
    else:            txt = f'p = {p:.2f}'
    return f'{txt} {stars}'

pairs = [('C231', 'N231'), ('N231', 'N436'), ('C231', 'N436')]
pvals_E = {}
for a, b in pairs:
    _, p = mannwhitneyu(data_E[a], data_E[b], alternative='two-sided')
    pvals_E[(a, b)] = p
    print(f"  E {DISP[a]} vs {DISP[b]}: p={p:.2e}  Cliff's delta={cliffs_delta(data_E[a], data_E[b]):+.3f}")

def draw_boxplot(ax, data_per_s, stats, pvals, ylabel, label_decimals=2):
    x_pos = [0, 1, 2]
    rng = np.random.default_rng(42)
    SUB = 600
    for xp, s in zip(x_pos, SAMPLES):
        arr = data_per_s[s]
        n_sub = min(SUB, len(arr))
        idx_sub = rng.choice(len(arr), n_sub, replace=False)
        jitter = rng.uniform(-0.18, 0.18, n_sub)
        ax.scatter(np.full(n_sub, xp) + jitter, arr[idx_sub],
                   s=1.2, alpha=0.25, c=SAMPLE_PALETTE[s],
                   linewidths=0, rasterized=True, zorder=1)
    box_data = [data_per_s[s] for s in SAMPLES]
    bp = ax.boxplot(box_data, positions=x_pos, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.0),
                    whiskerprops=dict(color='black', linewidth=0.6),
                    capprops=dict(color='black', linewidth=0.6),
                    boxprops=dict(linewidth=0.6), zorder=3)
    for patch, s in zip(bp['boxes'], SAMPLES):
        patch.set_facecolor(SAMPLE_PALETTE[s]); patch.set_alpha(0.85)
        patch.set_edgecolor('black')

    def whisker_top(a):
        q1, q3 = np.percentile(a, [25, 75]); lim = q3 + 1.5 * (q3 - q1)
        return a[a <= lim].max()
    ymax_data = max(max(np.percentile(a, 99.5) for a in box_data),
                    max(whisker_top(a) for a in box_data))
    ymin_data = min(arr.min() for arr in box_data)
    yrange = ymax_data - ymin_data
    bracket_levels = {
        ('C231', 'N231'): ymax_data + 0.05 * yrange,
        ('N231', 'N436'): ymax_data + 0.13 * yrange,
        ('C231', 'N436'): ymax_data + 0.23 * yrange,
    }
    bracket_xs = {('C231','N231'):(0,1),('N231','N436'):(1,2),('C231','N436'):(0,2)}
    tick_h = 0.018 * yrange
    for pair in pairs:
        y = bracket_levels[pair]; x1, x2 = bracket_xs[pair]
        ax.plot([x1, x1, x2, x2], [y - tick_h, y, y, y - tick_h],
                color='black', linewidth=0.5, zorder=5)
        ax.text((x1 + x2) / 2, y + 0.005 * yrange, fmt_p(pvals[pair]),
                fontsize=5.5, ha='center', va='bottom', color='black', zorder=6)
    ax.set_ylim(ymin_data - 0.04 * yrange, ymax_data + 0.32 * yrange)
    ax.set_xlim(-0.7, 2.7)
    ax.set_xticks(x_pos)
    fmt_str = f"{{:.{label_decimals}f}}"
    short_med = 'med' if label_decimals >= 3 else 'median'
    xtick_labels = [
        f"{DISP[s]}\nn={stats[s]['n']:,}"
        for s in SAMPLES
    ]
    ax.set_xticklabels(xtick_labels, fontsize=6)
    for tl, s in zip(ax.get_xticklabels(), SAMPLES):
        tl.set_color(SAMPLE_PALETTE[s]); tl.set_fontweight('bold')
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    for sp_n in ['top', 'right']: ax.spines[sp_n].set_visible(False)
    for sp_n in ['left', 'bottom']:
        ax.spines[sp_n].set_linewidth(0.5)
        ax.spines[sp_n].set_color('black')
    ax.tick_params(axis='both', length=2, width=0.5)
    ax.grid(False)

draw_boxplot(ax_E, data_E, stats_E, pvals_E,
             ylabel='Velocity length (a.u.)', label_decimals=2)

# ============================================================
# PANEL F — boxplot velocity_confidence per sample
# ============================================================
print("=== Panel F: velocity_confidence boxplot ===")
ax_F = fig.add_subplot(gs_row3[0, 2])

data_F = {}
stats_F = {}
for s in SAMPLES:
    mk = mask_keep & (sample_arr == s)
    arr = vel_conf[mk]
    data_F[s] = arr
    q25, q75 = np.percentile(arr, [25, 75])
    stats_F[s] = {'n': len(arr), 'median': np.median(arr),
                  'mean': np.mean(arr), 'iqr_low': q25, 'iqr_high': q75}
    print(f"  {s}: n={len(arr)} median={np.median(arr):.3f} mean={np.mean(arr):.3f}")

pvals_F = {}
for a, b in pairs:
    _, p = mannwhitneyu(data_F[a], data_F[b], alternative='two-sided')
    pvals_F[(a, b)] = p
    print(f"  F {DISP[a]} vs {DISP[b]}: p={p:.2e}  Cliff's delta={cliffs_delta(data_F[a], data_F[b]):+.3f}")

draw_boxplot(ax_F, data_F, stats_F, pvals_F,
             ylabel='Velocity confidence (a.u.)', label_decimals=3)

# ============================================================
# PANEL LABELS A-F (top-left of each panel, 11pt bold)
# ============================================================
# Use figure-level coords from the gridspec
def panel_label(ax, letter, dx=-0.06, dy=0.04):
    ax.text(dx, 1.0 + dy, letter,
            transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            va='bottom', ha='left', color='black')

panel_label(ax_A, 'A', dx=-0.08, dy=0.02)
panel_label(ax_B_list[0], 'B', dx=-0.08, dy=0.02)
panel_label(ax_C, 'C', dx=-0.06, dy=0.02)
panel_label(ax_D_left, 'D', dx=-0.10, dy=0.02)
panel_label(ax_E, 'E', dx=-0.22, dy=0.02)
panel_label(ax_F, 'F', dx=-0.22, dy=0.02)

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

pdf_path  = os.path.join(OUTDIR_PDF,  'Figure_2_v5.pdf')
tiff_path = os.path.join(OUTDIR_TIFF, 'Figure_2_v5.tiff')
png_path  = os.path.join(OUTDIR_PNG,  'Figure_2_v5_preview.png')

saved = []
print("\nSaving outputs...")

try:
    fig.savefig(png_path, format='png', dpi=200, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    print(f"  PNG: {png_path}")
    saved.append(png_path)
except Exception as e:
    print(f"  PNG FAILED: {e}")

try:
    fig.savefig(tiff_path, format='tiff', dpi=600, bbox_inches='tight',
                pad_inches=0.05, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f"  TIFF: {tiff_path}")
    saved.append(tiff_path)
except Exception as e:
    print(f"  TIFF FAILED: {e}")

try:
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    print(f"  PDF: {pdf_path}")
    saved.append(pdf_path)
except Exception as e:
    print(f"  PDF FAILED: {e}")

print()
for p in saved:
    sz = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):30s}  {sz:6.2f} MB")

plt.close()
print("\nDONE.")
