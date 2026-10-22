"""
Figure 4 master composite — Parameter sweep robustness.
(script numbering; = manuscript Figure 6)

v2 (20.10.2026): GO terms wrapped to 2 lines instead of truncated (duplicate
labels in B), rho labels in E moved outside bars. Outputs Figure_4_v2.*

Layout (~17.5 x 22 cm total):

  Row 1:  A (velocity_genes count per variant, FULL WIDTH)
          17.5 x 4 cm

  Row 2:  B (GO enrichment 3 sub-panels)  |  C (consensus drivers frac_variants)
          ~10 x 11 cm                     |  ~7 x 11 cm

  Row 3:  D (Jaccard 7x7 heatmap)         |  E (rho_TNF per variant)
          ~7 x 7 cm                       |  ~10 x 7 cm

Optimizations vs standalone panels:
  - 4A: NO colorbar (counts annotated above each bar)
  - 4B: hspace reduced to 0.65, top 6 GO terms per sub-panel (was 6 already)
  - 4C: NO colorbar (threshold line + bar lengths self-explanatory)
  - 4D: KEEPS colorbar (heatmap requires it)
  - 4E: no colorbar (never had one)

Rasterization:
  - 4A bars: vector (only 7)
  - 4B bars: rasterized (18 bars total)
  - 4C bars: rasterized (27 bars)
  - 4D heatmap: rasterized (imshow rasterized)
  - 4E bars: vector (only 7)

Author: Marko Zivanovic
Output: PDF + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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
# PATHS
# ============================================================
TAB = DATA_ROOT + "/velocity/03_outputs/tabular"
PATH_VEL_METRICS  = f"{TAB}/parameter_sweep_v2/velocity_metrics_per_param.csv"
PATH_JACCARD      = f"{TAB}/parameter_sweep_v2/top50_likelihood_jaccard.csv"
PATH_CONSENSUS    = f"{TAB}/parameter_consensus/consensus_top_likelihood_genes.csv"
PATH_BIO_CONS     = f"{TAB}/parameter_consensus/biology_consistency_per_variant.csv"
PATH_GO_DRIVERS   = f"{TAB}/enrichment/velocity_drivers_top50__GO_Biological_Process_2023_top10.csv"
PATH_GO_EARLY     = f"{TAB}/enrichment/latent_time_EARLY_RELAXED__GO_Biological_Process_2023_top10.csv"
PATH_GO_LATE      = f"{TAB}/enrichment/latent_time_LATE_RELAXED__GO_Biological_Process_2023_top10.csv"

# ============================================================
# CONSTANTS
# ============================================================
N_TOP_GO_PER_PANEL = 6
MAX_GO_TERM_LEN = 38   # tighter for narrow B panel inside master

# 4C
MIN_N_VARIANTS = 5
THRESHOLD_FRAC = 6/7

# ============================================================
# HELPERS
# ============================================================
def short_variant_label(v):
    if v.startswith('ref_'):
        return 'ref'
    return v.split('_')[0]

def short_variant_full(v):
    """Compact one-line: 'ref (n=10, pcs=30, g=2000)'"""
    if v.startswith('ref_'):
        prefix = 'ref'
        rest = v[4:]
    else:
        prefix = v.split('_')[0]
        rest = '_'.join(v.split('_')[1:])
    parts = rest.split('_')
    bits = []
    for p in parts:
        if p.startswith('n'): bits.append(f"n={p[1:]}")
        elif p.startswith('pcs'): bits.append(f"pcs={p[3:]}")
        elif p.startswith('g'): bits.append(f"g={p[1:]}")
    return f"{prefix} ({', '.join(bits)})"

def clean_go_term(t):
    t = re.sub(r'\s*\(GO:\d+\)\s*$', '', str(t))
    lines = textwrap.wrap(t, 46)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][:45] + '\u2026'
    return '\n'.join(lines)

def fmt_p_compact(p):
    if pd.isna(p): return 'p=NA'
    if p < 1e-30: return 'p<1e-30'
    if p < 1e-4:  return f'p={p:.0e}'
    if p < 0.05:  return f'p={p:.3f}'
    return f'p={p:.2f}'

# ============================================================
# LOAD ALL DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# 4A: velocity metrics
df_vel = pd.read_csv(PATH_VEL_METRICS)
df_vel = df_vel.rename(columns={df_vel.columns[0]: 'variant'})
print(f"\n4A: vel metrics shape={df_vel.shape}, "
      f"n_velocity_genes={df_vel['n_velocity_genes'].tolist()}")

# 4B: GO csvs
panels_B = [
    ('Velocity drivers (top50)', PATH_GO_DRIVERS),
    ('Early latent-time genes',  PATH_GO_EARLY),
    ('Late latent-time genes',   PATH_GO_LATE),
]
B_data = []
for title, path in panels_B:
    df_go = pd.read_csv(path)
    df_go = df_go.sort_values('Adjusted P-value', ascending=True).reset_index(drop=True)
    df_top = df_go.head(N_TOP_GO_PER_PANEL).copy()
    df_top['neg_log10_adjP'] = -np.log10(df_top['Adjusted P-value'])
    df_top['term_clean'] = df_top['Term'].apply(clean_go_term)
    B_data.append({'title': title, 'df': df_top})
    print(f"\n4B [{title}]: top {N_TOP_GO_PER_PANEL} terms loaded")

# 4C: consensus drivers
df_cons = pd.read_csv(PATH_CONSENSUS)
df_cons = df_cons[df_cons['n_variants'] >= MIN_N_VARIANTS].copy()
df_cons = df_cons.sort_values(['frac_variants', 'gene'],
                              ascending=[False, True]).reset_index(drop=True)
print(f"\n4C: {len(df_cons)} consensus genes (n_variants >= {MIN_N_VARIANTS})")

# 4D: jaccard
df_jac = pd.read_csv(PATH_JACCARD, index_col=0)
J = df_jac.values
n_var = len(df_jac)
print(f"\n4D: jaccard shape={J.shape}, "
      f"off-diag mean={J[~np.eye(n_var, dtype=bool)].mean():.3f}")

# 4E: biology consistency
df_bio = pd.read_csv(PATH_BIO_CONS)
df_bio['short'] = df_bio['variant'].apply(short_variant_label)
print(f"\n4E: biology consistency loaded, "
      f"rho_tnf range=[{df_bio['rho_tnf_vs_LT'].min():.3f}, "
      f"{df_bio['rho_tnf_vs_LT'].max():.3f}]")

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 17.5
FIG_H_CM = 25.0
CM_TO_INCH = 1 / 2.54

fig = plt.figure(figsize=(FIG_W_CM * CM_TO_INCH, FIG_H_CM * CM_TO_INCH))

# Master gridspec: 3 rows (A | B+C | D+E)
gs_master = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[4.0, 14.0, 7.0],
    hspace=0.42,
    left=0.06, right=0.97, top=0.955, bottom=0.045,
)

# ============================================================
# PANEL A — bar plot of velocity_genes per variant (FULL WIDTH)
# ============================================================
print("\n" + "=" * 60)
print("PANEL A — velocity genes per variant")
print("=" * 60)

ax_A = fig.add_subplot(gs_master[0])

variants_A = df_vel['variant'].tolist()
n_genes_A = df_vel['n_velocity_genes'].astype(int).values
xtick_labels_A = [short_variant_full(v) for v in variants_A]
ref_idx_A = next((i for i, v in enumerate(variants_A) if v.startswith('ref_')), 0)
ref_n_A = n_genes_A[ref_idx_A]

n_bars_A = len(variants_A)
x_pos_A = np.arange(n_bars_A)
vmin_A, vmax_A = n_genes_A.min(), n_genes_A.max()
norm_A = Normalize(vmin=vmin_A, vmax=vmax_A)
cmap_A = plt.cm.viridis
bar_colors_A = [cmap_A(norm_A(n)) for n in n_genes_A]

bars_A = ax_A.bar(x_pos_A, n_genes_A, width=0.7,
                  color=bar_colors_A, edgecolor='black', linewidth=0.5,
                  zorder=3)
bars_A[ref_idx_A].set_edgecolor('#B2182B')
bars_A[ref_idx_A].set_linewidth(1.5)

ax_A.axhline(ref_n_A, color='#888888', linewidth=0.6, linestyle='--',
             alpha=0.7, zorder=1, label=f'ref level (n={ref_n_A})')

for xp, n in zip(x_pos_A, n_genes_A):
    ax_A.text(xp, n + (vmax_A - vmin_A) * 0.04, str(n),
              fontsize=6.5, ha='center', va='bottom',
              fontweight='bold', color='black', zorder=4)

ax_A.set_xticks(x_pos_A)
ax_A.set_xticklabels(xtick_labels_A, fontsize=6, rotation=25, ha='right',
                     rotation_mode='anchor')
for tl, v in zip(ax_A.get_xticklabels(), variants_A):
    if v.startswith('ref_'):
        tl.set_color('#B2182B'); tl.set_fontweight('bold')

ax_A.set_ylabel('Velocity genes\n(count)', fontsize=7, labelpad=3)
ax_A.set_ylim(0, vmax_A * 1.25)
ax_A.set_xlim(-0.6, n_bars_A - 0.4)
for sp_n in ['top', 'right']:
    ax_A.spines[sp_n].set_visible(False)
for sp_n in ['left', 'bottom']:
    ax_A.spines[sp_n].set_linewidth(0.5)
    ax_A.spines[sp_n].set_color('black')
ax_A.tick_params(axis='y', length=2, width=0.5)
ax_A.tick_params(axis='x', length=2, width=0.5)
ax_A.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98),
            frameon=False, fontsize=6, handletextpad=0.5)
ax_A.set_title('Velocity genes per parameter variant',
               fontsize=8, fontweight='bold', loc='left', pad=4)

# ============================================================
# ROW 2 — B (left) + C (right)
# ============================================================
gs_BC = gs_master[1].subgridspec(
    nrows=1, ncols=2,
    width_ratios=[10, 7],
    wspace=0.28,
)

# ============================================================
# PANEL B — GO enrichment 3 stacked sub-panels
# ============================================================
print("\n" + "=" * 60)
print("PANEL B — GO enrichment (3 sub-panels)")
print("=" * 60)

gs_B = gs_BC[0, 0].subgridspec(
    nrows=3, ncols=1,
    hspace=0.42,
)

def draw_GO_subpanel(ax_pos_cell, panel, show_xlabel=False):
    ax = fig.add_subplot(ax_pos_cell)
    df = panel['df'].iloc[::-1].reset_index(drop=True)  # reverse for top-on-top
    n = len(df)
    y_pos = np.arange(n)
    values = df['neg_log10_adjP'].values
    terms = df['term_clean'].values

    vmin = values.min(); vmax = values.max()
    if vmax - vmin < 0.5:
        vmin = max(0, vmin - 0.5)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.magma_r
    colors = [cmap(norm(v)) for v in values]

    ax.barh(y_pos, values, height=0.7, color=colors,
            edgecolor='black', linewidth=0.4, zorder=3, rasterized=True)

    overlaps = df['Overlap'].values
    xmax = values.max()
    for yp, val, ov in zip(y_pos, values, overlaps):
        ax.text(val + xmax * 0.015, yp, f'{ov}',
                fontsize=5.5, va='center', ha='left', color='black', zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=5.4, linespacing=0.85)
    ax.set_xlim(0, xmax * 1.30)
    ax.set_ylim(-0.7, n - 0.3)
    if show_xlabel:
        ax.set_xlabel(r'$-\log_{10}$ (adjusted p-value)',
                      fontsize=6.5, labelpad=2)
    for sp_n in ['top', 'right']:
        ax.spines[sp_n].set_visible(False)
    for sp_n in ['left', 'bottom']:
        ax.spines[sp_n].set_linewidth(0.5)
        ax.spines[sp_n].set_color('black')
    ax.tick_params(axis='x', length=2, width=0.5, labelsize=5.8)
    ax.tick_params(axis='y', length=0)
    ax.set_title(panel['title'], fontsize=7, fontweight='bold', loc='left', pad=2)

for i, panel in enumerate(B_data):
    is_last = (i == len(B_data) - 1)
    draw_GO_subpanel(gs_B[i], panel, show_xlabel=is_last)

# ============================================================
# PANEL C — consensus drivers frac_variants barchart
# ============================================================
print("\n" + "=" * 60)
print("PANEL C — consensus drivers frac_variants")
print("=" * 60)

ax_C = fig.add_subplot(gs_BC[0, 1])

genes_C = df_cons['gene'].values[::-1]
fracs_C = df_cons['frac_variants'].values[::-1]
n_vars_C = df_cons['n_variants'].values[::-1]
n_genes_C = len(genes_C)
y_pos_C = np.arange(n_genes_C)

vmin_C = fracs_C.min() - 0.05
vmax_C = fracs_C.max()
norm_C = Normalize(vmin=vmin_C, vmax=vmax_C)
cmap_C = plt.cm.YlOrBr
colors_C = [cmap_C(norm_C(f)) for f in fracs_C]

ax_C.barh(y_pos_C, fracs_C, height=0.78, color=colors_C,
          edgecolor='black', linewidth=0.4, zorder=3, rasterized=True)
ax_C.axvline(THRESHOLD_FRAC, color='black', linewidth=0.6, linestyle='--',
             alpha=0.7, zorder=2)

for yp, frac, n in zip(y_pos_C, fracs_C, n_vars_C):
    ax_C.text(frac + 0.012, yp, f'{n}/7',
              fontsize=5.5, va='center', ha='left',
              color='black', zorder=4)

ax_C.set_yticks(y_pos_C)
ax_C.set_yticklabels(genes_C, fontsize=5.8, fontstyle='italic')
ax_C.set_ylim(-0.7, n_genes_C - 0.3)
ax_C.set_xlim(0, 1.10)
ax_C.set_xticks([0, 0.5, 1.0])
ax_C.set_xticklabels(['0', '0.5', '1.0'])
ax_C.set_xlabel('Fraction of variants', fontsize=7, labelpad=2)
for sp_n in ['top', 'right']:
    ax_C.spines[sp_n].set_visible(False)
for sp_n in ['left', 'bottom']:
    ax_C.spines[sp_n].set_linewidth(0.5)
    ax_C.spines[sp_n].set_color('black')
ax_C.tick_params(axis='x', length=2, width=0.5, labelsize=6)
ax_C.tick_params(axis='y', length=0)
ax_C.set_title('Driver consensus across 7 variants',
               fontsize=8, fontweight='bold', loc='left', pad=4)
# Inline note for threshold (small)
ax_C.text(THRESHOLD_FRAC, -0.6, '6/7',
          fontsize=5.5, color='#444444', ha='center', va='top')

# ============================================================
# ROW 3 — D (left) + E (right)
# ============================================================
gs_DE = gs_master[2].subgridspec(
    nrows=1, ncols=2,
    width_ratios=[7, 10],
    wspace=0.22,
)

# ============================================================
# PANEL D — Jaccard 7x7 heatmap
# ============================================================
print("\n" + "=" * 60)
print("PANEL D — Jaccard heatmap")
print("=" * 60)

# Internal grid: heatmap + colorbar
gs_D = gs_DE[0, 0].subgridspec(
    nrows=1, ncols=2,
    width_ratios=[20, 1],
    wspace=0.06,
)
ax_D = fig.add_subplot(gs_D[0, 0])
ax_D_cb = fig.add_subplot(gs_D[0, 1])

variants_D = list(df_jac.columns)
labels_D = [short_variant_label(v) for v in variants_D]

off_D = J[~np.eye(n_var, dtype=bool)]
vmin_D = max(0, off_D.min() - 0.05)
vmax_D = 1.0
norm_D = Normalize(vmin=vmin_D, vmax=vmax_D)
cmap_D = plt.cm.viridis

im_D = ax_D.imshow(J, aspect='equal', cmap=cmap_D,
                   vmin=vmin_D, vmax=vmax_D,
                   interpolation='nearest', rasterized=True)

for i in range(n_var):
    for j in range(n_var):
        v = J[i, j]
        rgb = cmap_D(norm_D(v))[:3]
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        txt_color = 'white' if luminance < 0.5 else 'black'
        ax_D.text(j, i, f'{v:.2f}', ha='center', va='center',
                  fontsize=5.8, color=txt_color, zorder=4)

ax_D.set_xticks(range(n_var))
ax_D.set_yticks(range(n_var))
ax_D.set_xticklabels(labels_D, fontsize=6.5)
ax_D.set_yticklabels(labels_D, fontsize=6.5)
for tl_x, tl_y, lab in zip(ax_D.get_xticklabels(), ax_D.get_yticklabels(), labels_D):
    if lab == 'ref':
        tl_x.set_color('#B2182B'); tl_x.set_fontweight('bold')
        tl_y.set_color('#B2182B'); tl_y.set_fontweight('bold')

for sp in ax_D.spines.values():
    sp.set_linewidth(0.4); sp.set_color('#888888')
ax_D.tick_params(axis='both', length=2, width=0.4)
ax_D.set_title('Top-50 likelihood overlap (Jaccard)',
               fontsize=8, fontweight='bold', loc='left', pad=4)

cb_D = fig.colorbar(im_D, cax=ax_D_cb, orientation='vertical')
cb_D.set_label('Jaccard\nindex', fontsize=6.5, labelpad=2)
cb_D.ax.tick_params(labelsize=6, length=2, pad=1)
cb_D.outline.set_linewidth(0.4)

# ============================================================
# PANEL E — TNF/AP-1 rho per variant
# ============================================================
print("\n" + "=" * 60)
print("PANEL E — rho TNF per variant")
print("=" * 60)

ax_E = fig.add_subplot(gs_DE[0, 1])

# Sort by rho ascending so most negative is at TOP after invert_yaxis
df_E = df_bio[['short', 'rho_tnf_vs_LT']].copy()
df_E.columns = ['variant', 'rho']
df_E_sorted = df_E.sort_values('rho', ascending=True).reset_index(drop=True)
plot_variants = df_E_sorted['variant'].values
plot_rhos = df_E_sorted['rho'].values
n_E = len(plot_variants)

def rho_to_color(r):
    if r > -0.10: return '#BBBBBB'
    elif r > -0.20: return '#D1E5F0'
    elif r > -0.35: return '#67A9CF'
    else: return '#2166AC'

bar_colors_E = [rho_to_color(r) for r in plot_rhos]

y_pos_E = np.arange(n_E)
bars_E = ax_E.barh(y_pos_E, plot_rhos, height=0.7,
                   color=bar_colors_E, edgecolor='black',
                   linewidth=0.5, zorder=3)
for i, v in enumerate(plot_variants):
    if v == 'v1':
        bars_E[i].set_edgecolor('#B2182B')
        bars_E[i].set_linewidth(1.2)

ax_E.axvline(0, color='black', linewidth=0.7, zorder=2)
median_rho = np.median(plot_rhos)
ax_E.axvline(median_rho, color='#888888', linewidth=0.6,
             linestyle='--', alpha=0.85, zorder=2)
ax_E.text(median_rho, n_E - 0.2, f' median = {median_rho:.2f}',
          fontsize=5.8, color='#444444', ha='left', va='bottom', zorder=4)

for yp, r, v in zip(y_pos_E, plot_rhos, plot_variants):
    rho_txt = f'\u03c1 = {r:+.3f}'
    ax_E.text(r - 0.012, yp, rho_txt,
              fontsize=5.8, va='center', ha='right',
              color='black', zorder=4)
    if v == 'v1':
        ax_E.text(0.012, yp, 'outlier',
                  fontsize=5.8, va='center', ha='left',
                  color='#B2182B', fontstyle='italic', zorder=4)

ax_E.set_yticks(y_pos_E)
ax_E.set_yticklabels(plot_variants, fontsize=7)
for tl, v in zip(ax_E.get_yticklabels(), plot_variants):
    if v == 'ref':
        tl.set_color('#B2182B'); tl.set_fontweight('bold')
ax_E.invert_yaxis()
ax_E.set_xlim(plot_rhos.min() - 0.17, 0.10)
ax_E.set_xticks([-0.6, -0.4, -0.2, 0.0])
ax_E.set_xlabel(r'Spearman $\rho$ (TNF/AP-1 vs LT)',
                fontsize=7, labelpad=2)
for sp_n in ['top', 'right']:
    ax_E.spines[sp_n].set_visible(False)
for sp_n in ['left', 'bottom']:
    ax_E.spines[sp_n].set_linewidth(0.5)
    ax_E.spines[sp_n].set_color('black')
ax_E.tick_params(axis='x', length=2, width=0.5)
ax_E.tick_params(axis='y', length=0)
ax_E.set_title('TNF/AP-1 \u2193 with LT across variants',
               fontsize=8, fontweight='bold', loc='left', pad=4)

n_below_02 = (plot_rhos < -0.2).sum()
summary_E = (f'{n_below_02}/7 variants: \u03c1 < \u22120.20  '
             f'(median = {median_rho:+.2f})')
ax_E.text(0.5, -0.20, summary_E,
          transform=ax_E.transAxes,
          fontsize=5.8, ha='center', va='top', color='#222222')

# ============================================================
# PANEL LABELS (figure-level via gs.get_position)
# ============================================================
fig.canvas.draw()

def add_label(x, y, letter):
    fig.text(x, y, letter, fontsize=11, fontweight='bold',
             va='top', ha='left', color='black')

pos_A  = gs_master[0].get_position(fig)
pos_B  = gs_BC[0, 0].get_position(fig)
pos_C  = gs_BC[0, 1].get_position(fig)
pos_D  = gs_DE[0, 0].get_position(fig)
pos_E  = gs_DE[0, 1].get_position(fig)

print("\n=== Panel positions (left, bottom, width, height) in figure fraction ===")
for letter, p in zip('ABCDE', [pos_A, pos_B, pos_C, pos_D, pos_E]):
    print(f"  {letter}: ({p.x0:.3f}, {p.y0:.3f}, {p.width:.3f}, {p.height:.3f})")

add_label(0.005, pos_A.y1 + 0.012, 'A')
add_label(0.005, pos_B.y1 + 0.012, 'B')
add_label(pos_C.x0 - 0.040, pos_C.y1 + 0.012, 'C')
add_label(0.005, pos_D.y1 + 0.012, 'D')
add_label(pos_E.x0 - 0.020, pos_E.y1 + 0.012, 'E')

# ============================================================
# SAVE
# ============================================================
print(f"\nFigure dims: {FIG_W_CM} x {FIG_H_CM} cm")

OUTDIR_BASE = DATA_ROOT + '/figures'
OUTDIR_PDF  = os.path.join(OUTDIR_BASE, 'pdf')
OUTDIR_TIFF = os.path.join(OUTDIR_BASE, 'tiff')
OUTDIR_PNG  = os.path.join(OUTDIR_BASE, 'png')
os.makedirs(OUTDIR_PDF,  exist_ok=True)
os.makedirs(OUTDIR_TIFF, exist_ok=True)
os.makedirs(OUTDIR_PNG,  exist_ok=True)

pdf_path  = os.path.join(OUTDIR_PDF,  'Figure_4_v2.pdf')
tiff_path = os.path.join(OUTDIR_TIFF, 'Figure_4_v2.tiff')
png_path  = os.path.join(OUTDIR_PNG,  'Figure_4_v2_preview.png')

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

plt.close()
print("\nDONE.")
