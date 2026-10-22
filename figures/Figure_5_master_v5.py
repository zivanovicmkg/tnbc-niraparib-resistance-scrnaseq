"""
Figure 5 master composite v5  (script numbering; = manuscript Figure 7).

v5 (20.10.2026): panel C (hand-drawn model schematic with manually assigned
statuses) removed per team decision; Hallmark terms in A wrapped instead of
truncated, wider gaps between A sub-panels. Outputs Figure_5_v5.*

(v4 notes below kept for history)

Changes from v3:
  - v3 introduced aspect-corrected Ellipse for circles, but the side
    effect was that status badge circles became too small to fit
    arrow symbols. v4 enlarges status circles and legend dots to
    compensate.
  - Status badge height: 6.5 -> 8.0 (more vertical room).
  - Status circle radius: h*0.16 -> h*0.28 (~75% larger).
  - Legend dot radius: 1.0 -> 1.5 in y, scaled by aspect in x.
  - Status arrow font: 6.5pt -> 7.5pt for better legibility.

Layout (17.5 x 22 cm total, layout beta):
  Panel A (top, h=8cm)    — MSigDB Hallmark ORA, 3 horizontal sub-panels
                              (drivers / early LT / late LT)
  Panel B (middle, h=7cm) — CC check (horizontal bars + 17 x 2 heatmap)
  Panel C (bottom, h=7cm) — biological model schematic
                              (3 cell pellets, niraparib trajectory,
                               status badges, adaptive quiescence,
                               key finding box)

Panel logic is duplicated (not refactored) to keep this script
self-contained and predictable, mirroring Figure 3 / Figure 4 masters.

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
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch, FancyArrowPatch, FancyBboxPatch
import gseapy

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
ENRICH_DIR = DATA_ROOT + "/velocity/03_outputs/tabular/enrichment"
RANKED_PATH = DATA_ROOT + "/velocity/03_outputs/tabular/velocity_genes_ranked_by_likelihood.csv"
CONSENSUS_PATH = DATA_ROOT + "/velocity/03_outputs/tabular/parameter_consensus/consensus_top_likelihood_genes.csv"
S1_RAW_PATH  = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"

# 5A constants
PANELS_5A = [
    ('Velocity drivers (top50)',
     'velocity_drivers_top50__MSigDB_Hallmark_2020_top10.csv',
     'velocity_drivers_top50__MSigDB_Hallmark_2020.csv'),
    ('Early latent-time genes',
     'latent_time_EARLY_RELAXED__MSigDB_Hallmark_2020_top10.csv',
     'latent_time_EARLY_RELAXED__MSigDB_Hallmark_2020.csv'),
    ('Late latent-time genes',
     'latent_time_LATE_RELAXED__MSigDB_Hallmark_2020_top10.csv',
     'latent_time_LATE_RELAXED__MSigDB_Hallmark_2020.csv'),
]
N_TOP_PER_PANEL = 8
MAX_TERM_LEN = 24

# 5B constants — 17 consensus drivers (3A order)
CONSENSUS_GENES_ORDER = [
    'ANKRD1', 'F3', 'CDA', 'SLAMF7', 'KPNA2', 'IL24', 'JUND',
    'HSP90AA1', 'TOP2A', 'AREG', 'KIFC3', 'KRT81', 'CLIP1', 'RPL24',
    'FAM111B', 'C19orf33', 'FTH1',
]
N_NEGATIVE_RHO = 14

# 5C palette (consistent with fig1A)
COLOR_C231 = '#4575B4'
COLOR_N231 = '#FDB863'
COLOR_N436 = '#B2182B'
COLOR_BOX  = '#E8E8E8'
COLOR_TXT  = '#222222'
COLOR_ACC  = '#555555'
COLOR_UP_STRONG   = '#B2182B'
COLOR_UP_MILD     = '#F4A582'
COLOR_NEUTRAL     = '#CCCCCC'
COLOR_DOWN_MILD   = '#92C5DE'
COLOR_DOWN_STRONG = '#2166AC'

# ============================================================
# HELPERS — 5A
# ============================================================
def clean_hallmark_term(t):
    t = str(t).strip()
    t = re.sub(r'^HALLMARK[_\s]+', '', t, flags=re.IGNORECASE)
    lines = textwrap.wrap(t, 22)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][:21] + '\u2026'
    return '\n'.join(lines)

# ============================================================
# HELPERS — 5C (copied from fig1A schematic for consistency)
# ============================================================
def draw_cell_pellet(ax, x, y, color, label, sublabel=None, radius=3.5):
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor='black',
                        linewidth=0.8, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y - radius - 2.0, label, ha='center', va='top',
            fontsize=8, fontweight='bold', color=COLOR_TXT, zorder=4)
    if sublabel:
        ax.text(x, y - radius - 5.0, sublabel, ha='center', va='top',
                fontsize=6.5, color=COLOR_TXT, style='italic', zorder=4)

def draw_arrow_5c(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0),
                  color=COLOR_ACC, lw=1.0, mutation_scale=12):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>', mutation_scale=mutation_scale,
                            color=color, linewidth=lw, zorder=2)
    ax.add_patch(arrow)
    if label:
        midx = (x1 + x2) / 2 + label_offset[0]
        midy = (y1 + y2) / 2 + label_offset[1]
        ax.text(midx, midy, label, ha='center', va='center',
                fontsize=6.5, color=COLOR_TXT, zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.9))

def draw_status_badge(ax, x, y, w, h, program_label, status_color, status_arrow):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15,rounding_size=0.4",
                         facecolor='white', edgecolor='black',
                         linewidth=0.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + h*0.22, program_label, ha='center', va='center',
            fontsize=6.0, color=COLOR_TXT, zorder=4)
    circle = plt.Circle((x, y - h*0.22), h*0.16,
                        facecolor=status_color, edgecolor='black',
                        linewidth=0.4, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y - h*0.22, status_arrow, ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='white', zorder=5)

# ============================================================
# LOAD DATA — 5A
# ============================================================
print("=== Loading 5A data ===")
panel_data_5a = []
for title, fname_top, fname_full in PANELS_5A:
    path_top  = os.path.join(ENRICH_DIR, fname_top)
    path_full = os.path.join(ENRICH_DIR, fname_full)
    path = path_top if os.path.exists(path_top) else path_full
    df = pd.read_csv(path)
    df = df.sort_values('Adjusted P-value', ascending=True).reset_index(drop=True)
    df_top = df.head(N_TOP_PER_PANEL).copy()
    df_top['neg_log10_adjP'] = -np.log10(df_top['Adjusted P-value'])
    df_top['term_clean'] = df_top['Term'].apply(clean_hallmark_term)
    panel_data_5a.append({'title': title, 'df': df_top})
    print(f"  {title}: {len(df_top)} terms loaded")

# ============================================================
# LOAD DATA — 5B
# ============================================================
print("\n=== Loading 5B data ===")
libs = gseapy.get_library('MSigDB_Hallmark_2020', organism='human')
e2f_genes = set(libs['E2F Targets'])
g2m_genes = set(libs['G2-M Checkpoint'])
cc_genes = e2f_genes | g2m_genes

df_ranked = pd.read_csv(RANKED_PATH)
df_ranked = df_ranked.sort_values('fit_likelihood', ascending=False).reset_index(drop=True)

df_cons = pd.read_csv(CONSENSUS_PATH)
df_cons_top17 = df_cons[df_cons['n_variants'] >= 6].copy()
top17_set = set(df_cons_top17['gene'].tolist())

categories_5b = []

n_cc = len(top17_set & cc_genes)
categories_5b.append({
    'label': 'Top 17 consensus',
    'n_total': len(top17_set), 'n_cc': n_cc,
    'pct_cc': n_cc / len(top17_set) * 100,
})

top50_set = set(df_ranked['gene'].head(50).tolist())
n_cc = len(top50_set & cc_genes)
categories_5b.append({
    'label': 'Top 50 likelihood',
    'n_total': len(top50_set), 'n_cc': n_cc,
    'pct_cc': n_cc / len(top50_set) * 100,
})

all_vel_set = set(df_ranked['gene'].tolist())
n_cc = len(all_vel_set & cc_genes)
n_all_vel = len(all_vel_set)
categories_5b.append({
    'label': f'All velocity genes (n={n_all_vel})',
    'n_total': n_all_vel, 'n_cc': n_cc,
    'pct_cc': n_cc / n_all_vel * 100,
})

import scanpy as sc
ad = sc.read_h5ad(S1_RAW_PATH)
all_genes_universe = set(ad.raw.var_names) if ad.raw is not None else set(ad.var_names)
baseline_pct = len(all_genes_universe & cc_genes) / len(all_genes_universe) * 100

for cat in categories_5b:
    cat['fold_enrichment'] = cat['pct_cc'] / baseline_pct if baseline_pct > 0 else float('inf')

heatmap_data_5b = np.zeros((len(CONSENSUS_GENES_ORDER), 2), dtype=int)
for i, g in enumerate(CONSENSUS_GENES_ORDER):
    heatmap_data_5b[i, 0] = int(g in e2f_genes)
    heatmap_data_5b[i, 1] = int(g in g2m_genes)

print(f"  Top 17 consensus: {categories_5b[0]['n_cc']}/{categories_5b[0]['n_total']} = {categories_5b[0]['pct_cc']:.1f}% CC ({categories_5b[0]['fold_enrichment']:.1f}x)")
print(f"  Top 50 likelihood: {categories_5b[1]['n_cc']}/{categories_5b[1]['n_total']} = {categories_5b[1]['pct_cc']:.1f}% CC ({categories_5b[1]['fold_enrichment']:.1f}x)")
print(f"  All velocity genes: {categories_5b[2]['n_cc']}/{categories_5b[2]['n_total']} = {categories_5b[2]['pct_cc']:.1f}% CC ({categories_5b[2]['fold_enrichment']:.1f}x)")
print(f"  Baseline: {baseline_pct:.2f}%")

# ============================================================
# FIGURE — full master, 17.5 x 22 cm
# ============================================================
FIG_W_CM = 17.5
FIG_H_CM = 16.0
CM_TO_INCH = 1 / 2.54

fig = plt.figure(figsize=(FIG_W_CM * CM_TO_INCH, FIG_H_CM * CM_TO_INCH))

# Outer gridspec: 3 rows (panel A | panel B | panel C)
# Sub-gridspecs inherit position from cell; margins controlled here.
gs_outer = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=[8.0, 7.0],
    hspace=0.45,
    left=0.13, right=0.95, top=0.93, bottom=0.04,
)

# ============================================================
# PANEL A — Hallmark ORA (3 horizontal sub-panels)
# ============================================================
gs_A = gs_outer[0].subgridspec(
    nrows=1, ncols=3,
    wspace=0.95,
)

def draw_panel_5a(ax, panel, show_xlabel=True):
    df = panel['df'].iloc[::-1].reset_index(drop=True)
    n = len(df)
    y_pos = np.arange(n)
    values = df['neg_log10_adjP'].values
    terms = df['term_clean'].values
    overlaps = df['Overlap'].values
    adj_pvals = df['Adjusted P-value'].values

    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 0.5:
        vmin = max(0, vmin - 0.5)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.magma_r
    colors = [cmap(norm(v)) for v in values]

    ax.barh(y_pos, values, height=0.7,
            color=colors, edgecolor='black', linewidth=0.4,
            zorder=3, rasterized=True)

    xmax = values.max()
    has_ns = (adj_pvals > 0.05).any()
    xlim_max = max(xmax * (1.30 if has_ns else 1.10), 1.7)
    ax.set_xlim(0, xlim_max)

    THRESH = -np.log10(0.05)
    if THRESH < xlim_max:
        ax.axvline(THRESH, color='#888888', linewidth=0.5,
                   linestyle='--', alpha=0.7, zorder=1)

    for yp, val, ov, p, color in zip(y_pos, values, overlaps, adj_pvals, colors):
        rgb = color[:3]
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        ov_txt_color = 'white' if luminance < 0.5 else 'black'
        ax.text(val - xlim_max * 0.008, yp, f'{ov}',
                fontsize=5.5, va='center', ha='right',
                color=ov_txt_color, zorder=4)
        if p > 0.05:
            ax.text(val + xlim_max * 0.020, yp, 'ns',
                    fontsize=5.5, va='center', ha='left',
                    color='#888888', fontstyle='italic', zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=5.6, linespacing=0.85)
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
    ax.grid(False)

    ax.set_title(panel['title'], fontsize=7.5, fontweight='bold',
                 loc='left', pad=3)

axes_5a = []
for i, panel in enumerate(panel_data_5a):
    ax = fig.add_subplot(gs_A[0, i])
    draw_panel_5a(ax, panel, show_xlabel=True)
    axes_5a.append(ax)

# ============================================================
# PANEL B — CC check (bars left, heatmap right)
# ============================================================
gs_B = gs_outer[1].subgridspec(
    nrows=1, ncols=2,
    width_ratios=[8, 8],
    wspace=0.45,
)

ax_bar = fig.add_subplot(gs_B[0, 0])
ax_hm = fig.add_subplot(gs_B[0, 1])

# Bar plot — 5B left
n_cat = len(categories_5b)
y_pos = np.arange(n_cat)
pcts  = [c['pct_cc'] for c in categories_5b]
labels = [c['label'] for c in categories_5b]
n_cc_list = [c['n_cc'] for c in categories_5b]
n_tot_list = [c['n_total'] for c in categories_5b]
folds = [c['fold_enrichment'] for c in categories_5b]

bar_colors_5b = ['#B2182B', '#D6604D', '#F4A582']
ax_bar.barh(y_pos, pcts, height=0.65,
            color=bar_colors_5b, edgecolor='black', linewidth=0.5,
            zorder=3, rasterized=True)
ax_bar.axvline(baseline_pct, color='#888888', linewidth=0.6,
               linestyle='--', alpha=0.85, zorder=1,
               label=f'Baseline ({baseline_pct:.2f}%)')

xmax_5b = max(pcts) * 1.55
ax_bar.set_xlim(0, xmax_5b)
ax_bar.set_ylim(-0.6, n_cat - 0.4)

for yp, pct, n_cc, n_tot, fold in zip(y_pos, pcts, n_cc_list, n_tot_list, folds):
    txt = f'{n_cc}/{n_tot}  ({pct:.1f}%, {fold:.1f}\u00d7)'
    ax_bar.text(pct + xmax_5b * 0.015, yp, txt,
                fontsize=6.0, va='center', ha='left',
                color='black', zorder=4)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(labels, fontsize=6.5)
ax_bar.invert_yaxis()
ax_bar.set_xlabel('% genes in CC (E2F U G2M)', fontsize=7, labelpad=2)
for sp_n in ['top', 'right']:
    ax_bar.spines[sp_n].set_visible(False)
for sp_n in ['left', 'bottom']:
    ax_bar.spines[sp_n].set_linewidth(0.5)
    ax_bar.spines[sp_n].set_color('black')
ax_bar.tick_params(axis='x', length=2, width=0.5, labelsize=6.2)
ax_bar.tick_params(axis='y', length=0)
ax_bar.legend(loc='lower right', frameon=False, fontsize=6,
              handletextpad=0.5)
ax_bar.set_title('CC enrichment across driver categories',
                 fontsize=7.5, fontweight='bold', loc='left', pad=4)

# Heatmap — 5B right
discrete_cmap = ListedColormap(['#F0F0F0', '#B2182B'])
ax_hm.imshow(heatmap_data_5b, aspect='auto', cmap=discrete_cmap,
             vmin=0, vmax=1, interpolation='nearest')
for i in range(len(CONSENSUS_GENES_ORDER)):
    for j in range(2):
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor='#888888', linewidth=0.4)
        ax_hm.add_patch(rect)
ax_hm.set_yticks(range(len(CONSENSUS_GENES_ORDER)))
ax_hm.set_yticklabels(CONSENSUS_GENES_ORDER, fontsize=6.5, fontstyle='italic')
ax_hm.set_xticks([0, 1])
ax_hm.set_xticklabels(['E2F\nTargets', 'G2-M\nCheckpoint'], fontsize=6.5)
ax_hm.axhline(N_NEGATIVE_RHO - 0.5, color='black', linewidth=0.7,
              linestyle='--', zorder=5)
ax_hm.set_xlim(-0.5, 1.5)
ax_hm.set_ylim(len(CONSENSUS_GENES_ORDER) - 0.5, -0.5)
for sp in ax_hm.spines.values():
    sp.set_linewidth(0.4); sp.set_color('#888888')
ax_hm.tick_params(axis='both', length=2, width=0.4)
legend_elements = [
    Patch(facecolor='#F0F0F0', edgecolor='black', linewidth=0.4, label='Not in set'),
    Patch(facecolor='#B2182B', edgecolor='black', linewidth=0.4, label='In set'),
]
ax_hm.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=6,
             handletextpad=0.5)
ax_hm.set_title('CC membership of 17 consensus drivers',
                fontsize=7.5, fontweight='bold', loc='left', pad=4)

# ============================================================
# PANEL LABELS A / B / C using gs.get_position()
# ============================================================
fig.canvas.draw()  # ensure positions are computed

posA = gs_outer[0].get_position(fig)
posB = gs_outer[1].get_position(fig)

label_x = 0.005
fig.text(label_x, posA.y1 + 0.050, 'A', fontsize=11, fontweight='bold',
         va='top', ha='left', color='black')
fig.text(label_x, posB.y1 + 0.050, 'B', fontsize=11, fontweight='bold',
         va='top', ha='left', color='black')

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

pdf_path  = os.path.join(OUTDIR_PDF,  'Figure_5_v5.pdf')
tiff_path = os.path.join(OUTDIR_TIFF, 'Figure_5_v5.tiff')
png_path  = os.path.join(OUTDIR_PNG,  'Figure_5_v5_preview.png')

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
