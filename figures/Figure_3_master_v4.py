"""
Figure 3 master composite v3  (script numbering; = manuscript Figure 5).

v3 (20.10.2026): NP231/HP231/LP231 names + colors, new cluster palette,
panel D = TOP2A/JUND/FAM111B (genes discussed in Discussion), B colorbar label
no longer bleeds into C, roomier A tracks, latent-time medians per sample
printed. Outputs Figure_3_v3.* (old Figure_3.* untouched).

Changes from v1:
  - Panel labels A, B, C, D moved to figure-level fig.text using gridspec
    cell positions (axes-relative dx was unreliable across nested gridspecs;
    A/B/D were overlapping panel titles, C was off-canvas).
  - Panel B colorbar moved closer to dot plot (labelpad reduced) and label
    text shortened so it doesn't bleed into panel C area.
  - Panel C: explicit rho ticks set to [-0.25, 0, 0.25] for clarity.
  - Top margin slightly increased (top=0.965 -> 0.955) to leave room for
    panel labels above panel titles.

Layout (~17.5 x 22 cm total):

  Row 1:  A (multi-track heatmap, 17 drivers x 300 LT bins, FULL WIDTH)
          17.5 x 7 cm

  Row 2:  B (dot plot 17 x 11)        |  C (GSEA stack, 3 vertical sub-panels)
          ~10 x 11 cm                 |  ~7 x 11 cm

  Row 3:  D (violin 1x4 row of 4 genes, FULL WIDTH)
          17.5 x 4 cm

Key technical decisions:
  - Cluster 8 (QC artifact) EXCLUDED from all quantitative panels.
  - velocity_length / latent_time data flows pulled from CellRank +
    Stage1 RAW (.raw slot, full 36,601 genes).
  - All scatter / heatmap / GSEA fill layers rasterized=True for smaller TIFF.
  - Boxplot/violin elements remain vector for clean rendering.
  - No scv.set_figure_params (would inject dark stylesheet).
  - rcParams white background forced explicitly.
  - GSEA computed once on the master ranking (Spearman rho vs latent_time)
    via gseapy.prerank.

Author: Marko Zivanovic
Output: PDF + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import os
import shutil
import re
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import zscore, mannwhitneyu

import scanpy as sc
import gseapy
from gseapy import prerank

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
# PALETTES & CONSTANTS
# ============================================================
SAMPLE_PALETTE = {'C231': '#0000FF', 'N231': '#FF8000', 'N436': '#00C000'}
DISP = {'C231': 'NP231', 'N231': 'HP231', 'N436': 'LP231'}
SAMPLES = ['C231', 'N231', 'N436']

CLUSTER_PALETTE = {
    '2': '#0000FF', '3': '#5B8DEF', '4': '#9DBBFF',      # NP231 family (2 dominant)
    '1': '#FF8000', '5': '#FFC266',                      # HP231 family (1 dominant)
    '0': '#00C000', '6': '#005A00',                      # LP231 family (0 dominant, 6 = G2/M)
    '7': '#8C564B', '9': '#D4A6C8',                      # mixed
    '10': '#E6007E', '11': '#E41A1C',                    # rare, highlighted
    '8': '#BDBDBD',                                      # QC
}
CLUSTER_ORDER = [str(i) for i in range(12)]

# 17 consensus drivers in the data-driven order from 3A
CONSENSUS_GENES_ORDER = [
    'ANKRD1', 'F3', 'CDA', 'SLAMF7', 'KPNA2', 'IL24', 'JUND',
    'HSP90AA1', 'TOP2A', 'AREG', 'KIFC3', 'KRT81', 'CLIP1', 'RPL24',
    'FAM111B', 'C19orf33', 'FTH1',
]
N_NEGATIVE_RHO = 14

# Sample-grouped cluster ordering for B (cluster 8 excluded)
CLUSTER_GROUPS = [
    ('NP231', ['2', '3', '4']),
    ('Mixed', ['7', '9']),
    ('HP231', ['1', '5', '11', '10']),
    ('LP231', ['0', '6']),
]
GROUP_COLORS = {
    'NP231': '#0000FF',
    'Mixed': '#8C564B',
    'HP231': '#FF8000',
    'LP231': '#00C000',
}
CLUSTERS_X = [c for _, lst in CLUSTER_GROUPS for c in lst]

# 4 violin genes (D)
VIOLIN_GENES = ['TOP2A', 'JUND', 'FAM111B']   # genes discussed in Discussion

PAIRS = [('C231', 'N231'), ('N231', 'N436'), ('C231', 'N436')]

N_BINS = 300

# GSEA targets (C)
GENE_SETS = [
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
    'HALLMARK_E2F_TARGETS',
]
QUERY_KEYWORDS = {
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB':
        ['tnfalphasignalingvianfkb', 'tnfasignalingvianfkb',
         'tnfsignalingnfkb', 'tnfalphasignaling', 'tnfsignaling'],
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION':
        ['epithelialmesenchymaltransition', 'emttransition'],
    'HALLMARK_E2F_TARGETS':
        ['e2ftargets'],
}

COLOR_POS_NES = '#B2182B'
COLOR_NEG_NES = '#4575B4'
COLOR_NS      = '#888888'

# ============================================================
# PATHS
# ============================================================
CR_PATH      = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
S1_FATE_PATH = DATA_ROOT + "/velocity/03_outputs/fate0_analysis/stage1_with_cellrank_fates.h5ad"
S1_RAW_PATH  = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"
LT_CSV       = DATA_ROOT + "/velocity/03_outputs/tabular/latent_time_programs/genes_correlated_with_latent_time_ALL.csv"

# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading CellRank: {CR_PATH}")
ad_cr = sc.read_h5ad(CR_PATH)

print(f"Loading Stage1 fate: {S1_FATE_PATH}")
ad_fate = sc.read_h5ad(S1_FATE_PATH)

print(f"Loading Stage1 RAW (with .raw): {S1_RAW_PATH}")
ad_raw = sc.read_h5ad(S1_RAW_PATH)
if ad_raw.raw is None:
    raise RuntimeError("ad_raw.raw is None — full transcriptome unavailable!")

# ============================================================
# BARCODE NORMALIZATION
# ============================================================
def normalize_cr(bc):
    sample = bc.split('_')[0]
    barcode = bc.split(':')[1].rstrip('x')
    return f"{barcode}-{sample}"

def normalize_s1(bc):
    parts = bc.split('-')
    return f"{parts[0]}-{parts[-1]}"

cr_keys = pd.Index([normalize_cr(bc) for bc in ad_cr.obs_names])

def build_s1_idx(adata):
    keys = pd.Index([normalize_s1(bc) for bc in adata.obs_names])
    return pd.Series(index=keys, data=range(len(keys)))

s1_fate_idx = build_s1_idx(ad_fate)
s1_raw_idx  = build_s1_idx(ad_raw)
mapped_fate_idx = np.array([s1_fate_idx[k] for k in cr_keys])
mapped_raw_idx  = np.array([s1_raw_idx[k]  for k in cr_keys])

leiden_arr = ad_fate.obs['leiden'].values[mapped_fate_idx].astype(str)
sample_arr = np.array([bc.split('_')[0] for bc in ad_cr.obs_names])
lt_arr     = ad_cr.obs['latent_time'].values
fate3_arr  = ad_cr.obs['fate_3'].values

mask_keep = leiden_arr != '8'
print(f"  Cells: {ad_cr.n_obs}, cluster 8 excluded: {(~mask_keep).sum()}, "
      f"kept: {mask_keep.sum()}")
print("  Latent time per sample (cluster 8 excluded):")
for _s in SAMPLES:
    _m = mask_keep & (sample_arr == _s)
    _q = np.percentile(lt_arr[_m], [25, 50, 75])
    print(f"    {DISP[_s]}: n={_m.sum()}  median={_q[1]:.3f}  IQR=[{_q[0]:.3f}, {_q[2]:.3f}]")

# ============================================================
# EXTRACT EXPRESSION FOR ALL NEEDED GENES
# ============================================================
all_genes_needed = sorted(set(CONSENSUS_GENES_ORDER) | set(VIOLIN_GENES))
raw_var_names = ad_raw.raw.var_names

gene_idx_raw = {}
missing = []
for g in all_genes_needed:
    if g in raw_var_names:
        gene_idx_raw[g] = raw_var_names.get_loc(g)
    else:
        missing.append(g)
if missing:
    raise RuntimeError(f"Missing genes in ad_raw.raw: {missing}")

raw_X = ad_raw.raw.X
col_idx = [gene_idx_raw[g] for g in all_genes_needed]
expr_log_full = raw_X[np.ix_(mapped_raw_idx, col_idx)].toarray()
gene_to_col = {g: i for i, g in enumerate(all_genes_needed)}
print(f"  Expression matrix: {expr_log_full.shape}  "
      f"(min={expr_log_full.min():.2f}, max={expr_log_full.max():.2f})")

# ============================================================
# RUN GSEA PRERANK (for C)
# ============================================================
print(f"\nLoading ranking: {LT_CSV}")
df_rho = pd.read_csv(LT_CSV)
df_sorted = df_rho.sort_values('spearman_rho', ascending=False).reset_index(drop=True)
ranked_genes = df_sorted['gene'].values
ranked_metric = df_sorted['spearman_rho'].values
n_total = len(ranked_genes)
print(f"  Ranking: {n_total} genes, rho range "
      f"[{ranked_metric.min():.3f}, {ranked_metric.max():.3f}]")

rnk = pd.DataFrame({0: ranked_genes, 1: ranked_metric})
tmp_outdir = tempfile.mkdtemp(prefix='gsea_master_')
print(f"\nRunning gseapy.prerank (1000 perm, seed=42)...")

pre_res = prerank(
    rnk=rnk, gene_sets='MSigDB_Hallmark_2020',
    permutation_num=1000, min_size=15, max_size=500,
    seed=42, threads=1, no_plot=True,
    outdir=tmp_outdir, format='png', verbose=False,
)
res2d = pre_res.res2d
results_dict = pre_res.results

# Robust matching for our 3 sets
def norm_str(s):
    s = str(s).lower()
    s = re.sub(r'^hallmark[_\s-]*', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def find_term(query):
    keywords = QUERY_KEYWORDS.get(query, []) + [norm_str(query)]
    res2d_norm = res2d['Term'].apply(norm_str)
    for kw in keywords:
        if not kw: continue
        m = res2d[res2d_norm.str.contains(kw, regex=False)
                  | res2d_norm.apply(lambda t: kw in t or t in kw)]
        if len(m) > 0:
            return m.iloc[0]
    return None

extracted = {}
matched_term_names = {}
for gs_name in GENE_SETS:
    row = find_term(gs_name)
    if row is None:
        print(f"  WARN: '{gs_name}' not found")
        extracted[gs_name] = None
        matched_term_names[gs_name] = None
    else:
        extracted[gs_name] = row
        matched_term_names[gs_name] = row['Term']
        print(f"  Found: {gs_name} -> '{row['Term']}'  "
              f"NES={row['NES']:.3f}  p={row['NOM p-val']:.3g}")

def get_es_and_hits(matched_term):
    if matched_term is None:
        return None, None
    if matched_term not in results_dict:
        nq = norm_str(matched_term)
        candidates = [k for k in results_dict.keys() if norm_str(k) == nq]
        if not candidates:
            return None, None
        rec = results_dict[candidates[0]]
    else:
        rec = results_dict[matched_term]
    res_curve = np.asarray(rec['RES']) if 'RES' in rec else None
    if 'hits' in rec: hits = np.asarray(rec['hits'])
    elif 'hit_indices' in rec: hits = np.asarray(rec['hit_indices'])
    else: hits = None
    return res_curve, hits

es_curves = {}
hit_lists = {}
for gs_name in GENE_SETS:
    es, hits = get_es_and_hits(matched_term_names[gs_name])
    es_curves[gs_name] = es
    hit_lists[gs_name] = hits

# ============================================================
# HELPER: format p-value with stars
# ============================================================
def fmt_p(p):
    if pd.isna(p): return 'p = NA'
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

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 17.5
FIG_H_CM = 24.0
CM_TO_INCH = 1 / 2.54

fig = plt.figure(figsize=(FIG_W_CM * CM_TO_INCH, FIG_H_CM * CM_TO_INCH))

# Master gridspec: 3 rows
#   Row 1 (A): heatmap full width, ~7 cm
#   Row 2 (B|C): dot plot + GSEA stack, ~11 cm
#   Row 3 (D): violin 1x4, ~4 cm
gs_master = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[9.0, 11.0, 4.5],
    hspace=0.30,
    left=0.05, right=0.97, top=0.945, bottom=0.04,
)

# ============================================================
# PANEL A — Multi-track heatmap (full width)
# ============================================================
print("\n=== Panel A: heatmap ===")

# Filter and rank cells by latent_time
expr_keep    = expr_log_full[mask_keep]
leiden_keep  = leiden_arr[mask_keep]
sample_keep  = sample_arr[mask_keep]
lt_keep      = lt_arr[mask_keep]
fate3_keep   = fate3_arr[mask_keep]

order_lt = np.argsort(lt_keep)
expr_sorted    = expr_keep[order_lt]
leiden_sorted  = leiden_keep[order_lt]
sample_sorted  = sample_keep[order_lt]
lt_sorted      = lt_keep[order_lt]
fate3_sorted   = fate3_keep[order_lt]

n_kept = len(lt_sorted)
bin_edges = np.linspace(0, n_kept, N_BINS + 1).astype(int)
print(f"  Binning {n_kept} cells into {N_BINS} bins")

# Build sub-matrix only for the 17 consensus genes (in 3A order)
cons_cols = [gene_to_col[g] for g in CONSENSUS_GENES_ORDER]
expr_cons_sorted = expr_sorted[:, cons_cols]

expr_bin    = np.zeros((N_BINS, len(CONSENSUS_GENES_ORDER)))
lt_bin      = np.zeros(N_BINS)
fate3_bin   = np.zeros(N_BINS)
sample_bin  = np.empty(N_BINS, dtype=object)
cluster_bin = np.empty(N_BINS, dtype=object)
for i in range(N_BINS):
    s, e = bin_edges[i], bin_edges[i+1]
    if e <= s: e = s + 1
    expr_bin[i]  = expr_cons_sorted[s:e].mean(axis=0)
    lt_bin[i]    = lt_sorted[s:e].mean()
    fate3_bin[i] = fate3_sorted[s:e].mean()
    vs, cs = np.unique(sample_sorted[s:e], return_counts=True)
    sample_bin[i] = vs[np.argmax(cs)]
    vc, cc = np.unique(leiden_sorted[s:e], return_counts=True)
    cluster_bin[i] = vc[np.argmax(cc)]

expr_z_A = zscore(expr_bin, axis=0, nan_policy='omit')
expr_z_A = np.clip(expr_z_A, -2.0, 2.0)
expr_z_A_for_imshow = expr_z_A.T  # rows=genes, cols=bins

# Nested gridspec for A: 5 rows (4 tracks + heatmap), 2 cols (main + cb)
gs_A = gs_master[0].subgridspec(
    nrows=5, ncols=2,
    height_ratios=[1.2, 1.2, 1.2, 1.2, 25],
    width_ratios=[40, 1.0],
    wspace=0.025, hspace=0.30,
)
ax_A_t1 = fig.add_subplot(gs_A[0, 0])
ax_A_t2 = fig.add_subplot(gs_A[1, 0])
ax_A_t3 = fig.add_subplot(gs_A[2, 0])
ax_A_t4 = fig.add_subplot(gs_A[3, 0])
ax_A_h  = fig.add_subplot(gs_A[4, 0])
ax_A_cb = fig.add_subplot(gs_A[4, 1])

def draw_categorical_track(ax, values, palette, label):
    colors = np.zeros((1, N_BINS, 3))
    for i, v in enumerate(values):
        hex_c = palette.get(v, '#FFFFFF')
        rgb = tuple(int(hex_c.lstrip('#')[j:j+2], 16) / 255 for j in (0, 2, 4))
        colors[0, i] = rgb
    ax.imshow(colors, aspect='auto', interpolation='nearest', rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888888')
    ax.set_ylabel(label, fontsize=6, rotation=0, ha='right', va='center', labelpad=3)

def draw_continuous_track(ax, values, cmap, vmin, vmax, label):
    arr = np.asarray(values).reshape(1, -1)
    ax.imshow(arr, aspect='auto', interpolation='nearest',
              cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888888')
    ax.set_ylabel(label, fontsize=6, rotation=0, ha='right', va='center', labelpad=3)

draw_categorical_track(ax_A_t1, sample_bin,  SAMPLE_PALETTE,  'Sample')
draw_categorical_track(ax_A_t2, cluster_bin, CLUSTER_PALETTE, 'Cluster')
draw_continuous_track (ax_A_t3, lt_bin,    'viridis', 0.0, 1.0, 'Latent time')
draw_continuous_track (ax_A_t4, fate3_bin, 'Purples', 0.0, 1.0, 'fate_3')

im_A = ax_A_h.imshow(
    expr_z_A_for_imshow, aspect='auto', interpolation='nearest',
    cmap='RdBu_r', vmin=-2.0, vmax=2.0, rasterized=True,
)
ax_A_h.set_xticks([])
ax_A_h.set_yticks(range(len(CONSENSUS_GENES_ORDER)))
ax_A_h.set_yticklabels(CONSENSUS_GENES_ORDER, fontsize=6.5, fontstyle='italic')
ax_A_h.set_xlabel(f'Latent time (rank-binned, {N_BINS} bins)',
                  fontsize=7, labelpad=2)
for sp in ax_A_h.spines.values():
    sp.set_linewidth(0.4); sp.set_color('#888888')
ax_A_h.axhline(N_NEGATIVE_RHO - 0.5, color='black',
               linewidth=0.7, linestyle='--', zorder=5)
ax_A_h.text(0, len(CONSENSUS_GENES_ORDER) - 0.4, 'low LT',
            fontsize=5.5, color='#444444', ha='left', va='top')
ax_A_h.text(N_BINS - 1, len(CONSENSUS_GENES_ORDER) - 0.4, 'high LT',
            fontsize=5.5, color='#444444', ha='right', va='top')

cb_A = fig.colorbar(im_A, cax=ax_A_cb, orientation='vertical')
cb_A.set_label('Expression\n(z-score)', fontsize=6.5, labelpad=2)
cb_A.ax.tick_params(labelsize=6, length=2, pad=1)
cb_A.outline.set_linewidth(0.4)
cb_A.set_ticks([-2, -1, 0, 1, 2])

ax_A_t1.set_title('Consensus drivers (17) along latent time',
                   fontsize=8, fontweight='bold', loc='left', pad=4)

# ============================================================
# PANEL B (left in row 2) — Dot plot 17 x 11
# ============================================================
print("\n=== Panel B: dot plot ===")

gs_BC = gs_master[1].subgridspec(
    nrows=1, ncols=2,
    width_ratios=[10, 7],
    wspace=0.18,
)

# Internal grid for B: main dot plot + cb on right + size legend below
gs_B = gs_BC[0, 0].subgridspec(
    nrows=2, ncols=2,
    height_ratios=[20, 2.0],
    width_ratios=[20, 1.0],
    wspace=0.05, hspace=0.30,
)
ax_B    = fig.add_subplot(gs_B[0, 0])
ax_B_cb = fig.add_subplot(gs_B[0, 1])
ax_B_sz = fig.add_subplot(gs_B[1, :])

# Compute per-cluster pct + mean for the 17 consensus genes
n_genes_B = len(CONSENSUS_GENES_ORDER)
n_clusters_B = len(CLUSTERS_X)
pct_expr  = np.zeros((n_genes_B, n_clusters_B))
mean_expr = np.zeros((n_genes_B, n_clusters_B))

cons_cols_B = [gene_to_col[g] for g in CONSENSUS_GENES_ORDER]
expr_cons_full = expr_log_full[:, cons_cols_B]

for j, c in enumerate(CLUSTERS_X):
    mc = mask_keep & (leiden_arr == c)
    if mc.sum() == 0:
        continue
    sub = expr_cons_full[mc]
    pct_expr[:, j]  = (sub > 0).mean(axis=0) * 100
    mean_expr[:, j] = sub.mean(axis=0)

expr_z_B = zscore(mean_expr, axis=1, nan_policy='omit')
expr_z_B = np.clip(expr_z_B, -2.0, 2.0)

S_MIN, S_MAX = 5, 70
def pct_to_size(pct):
    return S_MIN + (S_MAX - S_MIN) * (pct / 100.0)

cmap_B = plt.cm.Reds
norm_B = Normalize(vmin=-2.0, vmax=2.0)

for i in range(n_genes_B):
    for j in range(n_clusters_B):
        s = pct_to_size(pct_expr[i, j])
        c = cmap_B(norm_B(expr_z_B[i, j]))
        ax_B.scatter(j, i, s=s, color=c, edgecolors='black',
                     linewidths=0.3, zorder=3, rasterized=True)

# Vertical separators between sample family groups
group_sizes_B = [len(lst) for _, lst in CLUSTER_GROUPS]
sep_positions = []
acc = 0
for g_size in group_sizes_B[:-1]:
    acc += g_size
    sep_positions.append(acc - 0.5)
for sx in sep_positions:
    ax_B.axvline(sx, color='#CCCCCC', linewidth=0.4, zorder=1)

ax_B.axhline(N_NEGATIVE_RHO - 0.5, color='black',
             linewidth=0.7, linestyle='--', zorder=2)

ax_B.set_xticks(range(n_clusters_B))
ax_B.set_xticklabels(CLUSTERS_X, fontsize=6.5)
ax_B.set_yticks(range(n_genes_B))
ax_B.set_yticklabels(CONSENSUS_GENES_ORDER, fontsize=6.5, fontstyle='italic')
ax_B.invert_yaxis()
ax_B.set_xlim(-0.6, n_clusters_B - 0.4)
ax_B.set_ylim(n_genes_B - 0.5, -0.5)
for sp in ax_B.spines.values():
    sp.set_linewidth(0.4); sp.set_color('#888888')
ax_B.tick_params(axis='both', length=2, width=0.4)

# Group labels below cluster numbers
trans_B = mtrans.blended_transform_factory(ax_B.transData, ax_B.transAxes)
group_centers_B = []
acc = 0
for g_size in group_sizes_B:
    group_centers_B.append(acc + (g_size - 1) / 2.0)
    acc += g_size
for (gname, _), cx in zip(CLUSTER_GROUPS, group_centers_B):
    ax_B.text(cx, -0.10, gname, fontsize=7, fontweight='bold',
              color=GROUP_COLORS[gname],
              ha='center', va='top', transform=trans_B, clip_on=False)

# Colorbar
sm_B = ScalarMappable(norm=norm_B, cmap=cmap_B); sm_B.set_array([])
cb_B = fig.colorbar(sm_B, cax=ax_B_cb, orientation='vertical')
cb_B.ax.set_title('z-score', fontsize=6, pad=3)
cb_B.ax.tick_params(labelsize=5.8, length=2, pad=1)
cb_B.outline.set_linewidth(0.4)
cb_B.set_ticks([-2, -1, 0, 1, 2])

# Dot size legend (horizontal, below)
ax_B_sz.set_xlim(0, 1); ax_B_sz.set_ylim(0, 1)
ax_B_sz.axis('off')
legend_pcts = [25, 50, 75, 100]
x_legend = np.linspace(0.30, 0.70, len(legend_pcts))
ax_B_sz.text(0.18, 0.55, '% cells\nexpressing',
             fontsize=6, ha='right', va='center', transform=ax_B_sz.transAxes)
for pct, x in zip(legend_pcts, x_legend):
    ax_B_sz.scatter(x, 0.55, s=pct_to_size(pct), color='#CCCCCC',
                    edgecolors='black', linewidths=0.3,
                    transform=ax_B_sz.transAxes, clip_on=False)
    ax_B_sz.text(x, 0.10, f'{pct}%',
                 fontsize=5.8, ha='center', va='top',
                 transform=ax_B_sz.transAxes)

ax_B.set_title('Drivers x clusters', fontsize=8, fontweight='bold',
               loc='left', pad=4)

# ============================================================
# PANEL C (right in row 2) — GSEA stack
# ============================================================
print("\n=== Panel C: GSEA stack ===")

gs_C = gs_BC[0, 1].subgridspec(
    nrows=3, ncols=1,
    hspace=0.55,
)

def draw_gsea_subpanel(gs_outer_cell, gs_name, row_data, es_curve, hits,
                       show_xlabel=False, panel_title=True):
    gs_inner = gs_outer_cell.subgridspec(
        nrows=3, ncols=1,
        height_ratios=[4, 0.6, 1.5],
        hspace=0.05,
    )
    ax_es   = fig.add_subplot(gs_inner[0, 0])
    ax_tick = fig.add_subplot(gs_inner[1, 0], sharex=ax_es)
    ax_rm   = fig.add_subplot(gs_inner[2, 0], sharex=ax_es)

    nes = float(row_data['NES']) if row_data is not None else 0.0
    p   = float(row_data['NOM p-val']) if row_data is not None else 1.0
    if abs(nes) < 1.0 or p > 0.05:
        line_color = COLOR_NS
    elif nes > 0:
        line_color = COLOR_POS_NES
    else:
        line_color = COLOR_NEG_NES

    if es_curve is not None:
        x = np.arange(len(es_curve))
        ax_es.plot(x, es_curve, color=line_color, linewidth=1.0,
                   rasterized=True)
        ax_es.fill_between(x, 0, es_curve, color=line_color, alpha=0.15,
                           rasterized=True)
        ax_es.axhline(0, color='#888888', linewidth=0.4, linestyle='-')
    ax_es.set_ylabel('ES', fontsize=6, labelpad=2)
    for sp in ax_es.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888888')
    ax_es.tick_params(axis='both', length=2, width=0.4, labelsize=5.5)
    plt.setp(ax_es.get_xticklabels(), visible=False)

    ax_tick.set_ylim(0, 1)
    if hits is not None and len(hits) > 0:
        for h in hits:
            ax_tick.plot([h, h], [0, 1], color='black',
                         linewidth=0.3, alpha=0.8, rasterized=True)
    ax_tick.set_yticks([])
    for sp_n in ['top', 'bottom']:
        ax_tick.spines[sp_n].set_visible(False)
    for sp_n in ['left', 'right']:
        ax_tick.spines[sp_n].set_linewidth(0.4)
        ax_tick.spines[sp_n].set_color('#888888')
    ax_tick.tick_params(axis='x', length=0)
    plt.setp(ax_tick.get_xticklabels(), visible=False)

    pos_mask = ranked_metric >= 0
    neg_mask = ranked_metric < 0
    x_rm = np.arange(n_total)
    ax_rm.fill_between(x_rm, 0, ranked_metric, where=pos_mask,
                       color=COLOR_POS_NES, linewidth=0, alpha=0.85,
                       interpolate=True, rasterized=True)
    ax_rm.fill_between(x_rm, 0, ranked_metric, where=neg_mask,
                       color=COLOR_NEG_NES, linewidth=0, alpha=0.85,
                       interpolate=True, rasterized=True)
    ax_rm.axhline(0, color='#888888', linewidth=0.4)
    ax_rm.set_ylabel(r'$\rho$', fontsize=6, labelpad=2)
    ax_rm.set_yticks([-0.25, 0.25])
    if show_xlabel:
        ax_rm.set_xlabel('Rank', fontsize=6.5, labelpad=2)
    else:
        plt.setp(ax_rm.get_xticklabels(), visible=False)
    ax_rm.set_xlim(0, n_total)
    for sp in ax_rm.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888888')
    ax_rm.tick_params(axis='both', length=2, width=0.4, labelsize=5.5)

    if row_data is not None:
        nes_txt = f"NES={nes:+.2f}"
        # Compact p text for narrow panel
        if p < 1e-30: p_txt = 'p<1e-30'
        elif p < 1e-4: p_txt = f'p={p:.0e}'
        elif p < 0.05: p_txt = f'p={p:.3f}'
        else: p_txt = f'p={p:.2f}'
        if p < 0.0001:   stars = '****'
        elif p < 0.001:  stars = '***'
        elif p < 0.01:   stars = '**'
        elif p < 0.05:   stars = '*'
        else:            stars = 'ns'
        display_name = row_data['Term']
        title = f"{display_name}  {nes_txt}  {p_txt} {stars}"
    else:
        title = f"{gs_name}  (NOT FOUND)"
    if panel_title:
        ax_es.set_title(title, fontsize=6.8, fontweight='bold',
                        loc='left', pad=2)

for i, gs_name in enumerate(GENE_SETS):
    is_last = (i == len(GENE_SETS) - 1)
    draw_gsea_subpanel(
        gs_C[i],
        gs_name,
        extracted[gs_name],
        es_curves[gs_name],
        hit_lists[gs_name],
        show_xlabel=is_last,
    )

# ============================================================
# PANEL D — Violin 1x4 row (full width)
# ============================================================
print("\n=== Panel D: violin 1x4 ===")

gs_D = gs_master[2].subgridspec(
    nrows=1, ncols=len(VIOLIN_GENES),
    wspace=0.30,
)

# Pre-compute per-gene per-sample arrays (cluster 8 excluded)
violin_data = {g: {} for g in VIOLIN_GENES}
for g in VIOLIN_GENES:
    j = gene_to_col[g]
    for s in SAMPLES:
        m = mask_keep & (sample_arr == s)
        violin_data[g][s] = expr_log_full[m, j]

def draw_violin_subpanel(ax, gene, show_ylabel=True):
    arrs = [violin_data[gene][s] for s in SAMPLES]
    x_pos = [0, 1, 2]

    parts = ax.violinplot(
        arrs, positions=x_pos, widths=0.8,
        showmeans=False, showmedians=False, showextrema=False,
        bw_method=0.25,
    )
    for body, s in zip(parts['bodies'], SAMPLES):
        body.set_facecolor(SAMPLE_PALETTE[s])
        body.set_edgecolor('black')
        body.set_linewidth(0.5)
        body.set_alpha(0.65)

    box_width = 0.18
    for xp, arr, s in zip(x_pos, arrs, SAMPLES):
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        iqr = q75 - q25
        lo = max(arr.min(), q25 - 1.5 * iqr)
        hi = min(arr.max(), q75 + 1.5 * iqr)
        ax.plot([xp, xp], [lo, hi], color='black', linewidth=0.5, zorder=4)
        cap_w = box_width * 0.4
        ax.plot([xp - cap_w/2, xp + cap_w/2], [lo, lo],
                color='black', linewidth=0.5, zorder=4)
        ax.plot([xp - cap_w/2, xp + cap_w/2], [hi, hi],
                color='black', linewidth=0.5, zorder=4)
        rect_x0 = xp - box_width / 2
        ax.add_patch(plt.Rectangle(
            (rect_x0, q25), box_width, q75 - q25,
            facecolor='white', edgecolor='black', linewidth=0.6, zorder=5,
        ))
        ax.plot([rect_x0, rect_x0 + box_width], [med, med],
                color='black', linewidth=1.0, zorder=6)

    all_data = np.concatenate(arrs)
    ymax_data = all_data.max()
    ymin_data = all_data.min()
    yrange = ymax_data - ymin_data
    if yrange < 0.01:
        yrange = max(ymax_data * 0.1, 0.01)

    bracket_levels = {
        ('C231','N231'): ymax_data + 0.06 * yrange,
        ('N231','N436'): ymax_data + 0.16 * yrange,
        ('C231','N436'): ymax_data + 0.28 * yrange,
    }
    bracket_xs = {('C231','N231'):(0,1),('N231','N436'):(1,2),('C231','N436'):(0,2)}
    tick_h = 0.018 * yrange
    for pair in PAIRS:
        _, p = mannwhitneyu(violin_data[gene][pair[0]],
                            violin_data[gene][pair[1]],
                            alternative='two-sided')
        y = bracket_levels[pair]
        x1, x2 = bracket_xs[pair]
        ax.plot([x1, x1, x2, x2], [y - tick_h, y, y, y - tick_h],
                color='black', linewidth=0.5, zorder=7)
        # Compact p formatting for narrow panel
        if p < 1e-30: pt = 'p<1e-30'
        elif p < 1e-4: pt = f'p={p:.0e}'
        elif p < 0.05: pt = f'p={p:.3f}'
        else: pt = f'p={p:.2f}'
        if p < 0.0001:   stars = '****'
        elif p < 0.001:  stars = '***'
        elif p < 0.01:   stars = '**'
        elif p < 0.05:   stars = '*'
        else:            stars = 'ns'
        ax.text((x1 + x2) / 2, y + 0.005 * yrange, f'{pt} {stars}',
                fontsize=5.0, ha='center', va='bottom', zorder=8)

    ax.set_ylim(ymin_data - 0.03 * yrange, ymax_data + 0.42 * yrange)
    ax.set_xlim(-0.7, 2.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([DISP[s] for s in SAMPLES], fontsize=6)
    for tl, s in zip(ax.get_xticklabels(), SAMPLES):
        tl.set_color(SAMPLE_PALETTE[s])
        tl.set_fontweight('bold')
    if show_ylabel:
        ax.set_ylabel('Expression\n(log-norm)', fontsize=6.5, labelpad=2)
    for sp_n in ['top', 'right']:
        ax.spines[sp_n].set_visible(False)
    for sp_n in ['left', 'bottom']:
        ax.spines[sp_n].set_linewidth(0.5)
        ax.spines[sp_n].set_color('black')
    ax.tick_params(axis='both', length=2, width=0.5)
    ax.grid(False)
    ax.set_title(gene, fontsize=8, fontweight='bold', fontstyle='italic',
                 loc='left', pad=3, color='black')

ax_D_list = []
for k, gene in enumerate(VIOLIN_GENES):
    ax = fig.add_subplot(gs_D[0, k])
    draw_violin_subpanel(ax, gene, show_ylabel=(k == 0))
    ax_D_list.append(ax)

# ============================================================
# PANEL LABELS A-D (figure-level, positioned via gridspec cells)
# ============================================================
# Force draw to settle gridspec positions before reading them
fig.canvas.draw()

# Each panel gets its label at the top-left of its row block in figure coords.
# We use the master gridspec cell positions for A, D (full-width rows),
# and the inner BC split for B and C.
def add_label(x, y, letter):
    fig.text(x, y, letter,
             fontsize=11, fontweight='bold',
             va='top', ha='left', color='black')

pos_A = gs_master[0].get_position(fig)   # full-width row 1
pos_BC = gs_master[1].get_position(fig)  # row 2 (B + C)
pos_D = gs_master[2].get_position(fig)   # full-width row 3
pos_B = gs_BC[0, 0].get_position(fig)    # B (left of row 2)
pos_C = gs_BC[0, 1].get_position(fig)    # C (right of row 2)

# A — top-left of row 1 (slightly above its top edge)
add_label(0.005, pos_A.y1 + 0.030, 'A')
# B — top-left of B cell
add_label(0.005, pos_B.y1 + 0.012, 'B')
# C — top-left of C cell
add_label(pos_C.x0 - 0.045, pos_C.y1 + 0.030, 'C')
# D — top-left of row 3
add_label(0.005, pos_D.y1 + 0.012, 'D')

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

pdf_path  = os.path.join(OUTDIR_PDF,  'Figure_3_v4.pdf')
tiff_path = os.path.join(OUTDIR_TIFF, 'Figure_3_v4.tiff')
png_path  = os.path.join(OUTDIR_PNG,  'Figure_3_v4_preview.png')

print(f"\nSaving outputs (figure dims: {FIG_W_CM} x {FIG_H_CM} cm)...")

saved = []
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
    note = '  WARN: > 10 MB!' if sz > 10 else ''
    print(f"  {os.path.basename(p):30s}  {sz:6.2f} MB{note}")

# Cleanup tmp dir
try:
    shutil.rmtree(tmp_outdir)
except Exception:
    pass

plt.close()
print("\nDONE.")
