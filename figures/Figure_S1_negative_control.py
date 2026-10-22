"""
Figure S1 — Negative control for RNA velocity analysis.

Compares the actual RNA velocity model output against two randomization
controls to show that the velocity signal is not a methodological artefact:

  1. Original              — real spliced/unspliced data, dynamical model
  2. Permuted unspliced    — unspliced counts shuffled across cells
                              (destroys cell-specific kinetic information,
                              keeps the unspliced expression DISTRIBUTION)
  3. Random velocity field — per-cell velocity vectors replaced with
                              random unit vectors of same magnitude

Two metrics shown side-by-side as bar plots:
  - vel_confidence_median: how directionally consistent each cell's
                           velocity is with its k-nearest neighbours
  - vel_length_median:     median absolute magnitude of velocity vectors

Source: negative_control_summary.csv

Author: Marko Zivanovic
Output: PDF + TIFF (600 dpi LZW) + PNG preview
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
CSV_PATH = DATA_ROOT + "/velocity/03_outputs/tabular/negative_control/negative_control_summary.csv"

# Bar colors: original = signature blue (real signal),
# permuted = mid-grey (partial control), random = light red (null)
SCENARIO_COLORS = {
    'Original':                   '#2166AC',  # deep blue — real
    'Permuted unspliced (cells)': '#999999',  # mid grey — partial null
    'Random velocity field':      '#D6604D',  # warm red-orange — null
}
# Short labels for x-axis (full names too long for narrow panel)
SCENARIO_SHORT = {
    'Original':                   'Original',
    'Permuted unspliced (cells)': 'Permuted\nunspliced',
    'Random velocity field':      'Random\nvelocity field',
}

# ============================================================
# LOAD
# ============================================================
print(f"Loading {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(df.to_string(index=False))

# Maintain CSV order (Original first)
scenarios = df['scenario'].tolist()
conf_vals  = df['vel_confidence_median'].values
len_vals   = df['vel_length_median'].values

print("\n=== Numeric summary ===")
for s, c, l in zip(scenarios, conf_vals, len_vals):
    print(f"  {s:35s}: confidence={c:.3f}, length={l:.2f}")

# Compute fold-change vs Random (the most aggressive null) for confidence
random_conf = df.loc[df['scenario'] == 'Random velocity field',
                     'vel_confidence_median'].iloc[0]
orig_conf = df.loc[df['scenario'] == 'Original',
                   'vel_confidence_median'].iloc[0]
fold_orig_vs_random = orig_conf / random_conf
print(f"\nFold-change in confidence: Original / Random = {fold_orig_vs_random:.2f}x")

# ============================================================
# FIGURE
# ============================================================
FIG_W_CM = 12.0
FIG_H_CM = 6.0
CM_TO_INCH = 1 / 2.54

fig = plt.figure(figsize=(FIG_W_CM * CM_TO_INCH, FIG_H_CM * CM_TO_INCH))

# Two sub-panels side-by-side
gs = fig.add_gridspec(
    nrows=1, ncols=2,
    wspace=0.45,
    left=0.10, right=0.97, top=0.86, bottom=0.22,
)

# ============================================================
# Sub-panel A: velocity confidence
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])

x_pos = np.arange(len(scenarios))
bar_colors = [SCENARIO_COLORS[s] for s in scenarios]
ax1.bar(x_pos, conf_vals, width=0.65,
        color=bar_colors, edgecolor='black', linewidth=0.5,
        zorder=3)

# Annotations on top of bars
ymax_conf = max(conf_vals) * 1.20
for xp, v in zip(x_pos, conf_vals):
    ax1.text(xp, v + ymax_conf * 0.025, f'{v:.2f}',
             ha='center', va='bottom', fontsize=6.2,
             fontweight='bold', color='black', zorder=4)

ax1.set_xticks(x_pos)
ax1.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios], fontsize=6.2)
ax1.set_ylabel('Median velocity\nconfidence', fontsize=7, labelpad=2)
ax1.set_ylim(0, ymax_conf)
ax1.set_xlim(-0.6, len(scenarios) - 0.4)
for sp_n in ['top', 'right']:
    ax1.spines[sp_n].set_visible(False)
for sp_n in ['left', 'bottom']:
    ax1.spines[sp_n].set_linewidth(0.5)
    ax1.spines[sp_n].set_color('black')
ax1.tick_params(axis='y', length=2, width=0.5)
ax1.tick_params(axis='x', length=2, width=0.5)
ax1.set_title('Velocity confidence', fontsize=7.5, fontweight='bold',
              loc='left', pad=3)

# ============================================================
# Sub-panel B: velocity length
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])

ax2.bar(x_pos, len_vals, width=0.65,
        color=bar_colors, edgecolor='black', linewidth=0.5,
        zorder=3)

ymax_len = max(len_vals) * 1.20
for xp, v in zip(x_pos, len_vals):
    ax2.text(xp, v + ymax_len * 0.025, f'{v:.2f}',
             ha='center', va='bottom', fontsize=6.2,
             fontweight='bold', color='black', zorder=4)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios], fontsize=6.2)
ax2.set_ylabel('Median velocity\nlength', fontsize=7, labelpad=2)
ax2.set_ylim(0, ymax_len)
ax2.set_xlim(-0.6, len(scenarios) - 0.4)
for sp_n in ['top', 'right']:
    ax2.spines[sp_n].set_visible(False)
for sp_n in ['left', 'bottom']:
    ax2.spines[sp_n].set_linewidth(0.5)
    ax2.spines[sp_n].set_color('black')
ax2.tick_params(axis='y', length=2, width=0.5)
ax2.tick_params(axis='x', length=2, width=0.5)
ax2.set_title('Velocity length', fontsize=7.5, fontweight='bold',
              loc='left', pad=3)

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

pdf_path  = os.path.join(OUTDIR_PDF,  'Figure_S1_negative_control.pdf')
tiff_path = os.path.join(OUTDIR_TIFF, 'Figure_S1_negative_control.tiff')
png_path  = os.path.join(OUTDIR_PNG,  'Figure_S1_negative_control_preview.png')

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
