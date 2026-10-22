#!/usr/bin/env python3
"""
Figure 9 (manuscript) — Targeted DGE across niraparib sublines.

Panels
  A  Drug transporters (efflux / Cu-Pt efflux / influx)
  B  EMT genes (EMT signalling-pathway panel dropped per team decision)
  C  DNA-damage response (HDR widened to DDR)

Each panel: dot plot per subline (size = % cells expressing, colour = mean
log-norm expression) + heatmap of log2FC for four fixed contrasts:
  HP231 vs NP231 | LP231 vs NP231 | HP231+LP231 vs NP231 | LP231 vs HP231

Statistics
  Wilcoxon rank-sum per cell (scanpy rank_genes_groups), BH-FDR over all
  genes detected in >= 10 cells. Each subline is a single culture, so cells
  are pseudo-replicates and p-values are inflated; a gene is therefore marked
  as changed only if FDR < 0.05 AND |log2FC| >= 0.25. Stars: * <0.05,
  ** <0.01, *** <0.001 (FDR), shown only for genes passing both criteria.
  Genes expressed in < 10% of cells in every subline are not plotted (listed
  in the figure footnote and the log).
  Cluster 8 (QC) excluded, as in all quantitative figures.

Also prints, for every panel gene, whether it is a velocity gene / consensus
driver and its Spearman rho vs latent time (team question: can any of these
be a driver?).

Outputs
  figures/{pdf,tiff,png}/Figure_9_DGE_v2.*
  figures/dge/dge_panel_stats_v2.csv   (all numbers behind the figure)

Run
  conda run --no-capture-output -n b2bio python figure9_DGE_v1.py
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import textwrap
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

BASE = os.environ.get("DGE_BASE", DATA_ROOT + "")
S1_PATH = f"{BASE}/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"
TAB = f"{BASE}/velocity/03_outputs/tabular"
CONS_PATH = f"{TAB}/parameter_consensus/consensus_top_likelihood_genes.csv"
RANKED_PATH = f"{TAB}/velocity_genes_ranked_by_likelihood.csv"
LT_PATH = f"{TAB}/latent_time_programs/genes_correlated_with_latent_time_ALL.csv"
OUT = f"{BASE}/figures"

plt.rcParams.update({
    'font.family': os.environ.get("DGE_FONT", 'Arial'), 'font.size': 6.5,
    'axes.linewidth': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

SAMPLES = ['C231', 'N231', 'N436']
DISP = {'C231': 'NP231', 'N231': 'HP231', 'N436': 'LP231'}
SAMPLE_COLORS = {'C231': '#0000FF', 'N231': '#FF8000', 'N436': '#00C000'}

PANELS = [
    ('A', 'Drug transporters', [
        ('Efflux pumps', ['ABCB1', 'ABCG2', 'ABCC1', 'ABCC2']),
        ('Cu/Pt', ['ATP7A', 'ATP7B']),
        ('Influx', ['SLC31A1']),
    ]),
    ('B', 'EMT genes', [
        ('Master TFs', ['SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2']),
        ('Mesenchymal', ['VIM', 'CDH2', 'FN1']),
        ('Epith.', ['CDH1']),
        ('Effectors', ['TGFB1', 'MMP2', 'MMP9']),
    ]),
    ('C', 'DNA-damage response', [
        ('PARP', ['PARP1', 'PARP2', 'PARG']),
        ('Homologous recombination', ['BRCA1', 'BRCA2', 'PALB2', 'RAD51', 'RAD51C',
                                      'MRE11', 'RAD50', 'NBN', 'RBBP8']),
        ('NHEJ / 53BP1 axis', ['TP53BP1', 'MAD2L2', 'XRCC5', 'XRCC6', 'PRKDC', 'LIG4']),
        ('Checkpoint', ['ATM', 'ATR', 'CHEK1', 'CHEK2', 'WEE1']),
        ('PARPi sens.', ['SLFN11']),
        ('FA / BER / MMR', ['FANCD2', 'XRCC1', 'POLB', 'MLH1', 'MSH2']),
    ]),
]

CONTRASTS = [  # (label, group_col, test group, reference)
    ('HP231 vs NP231', 'sample', 'N231', 'C231'),
    ('LP231 vs NP231', 'sample', 'N436', 'C231'),
    ('HP+LP vs NP231', 'treat', 'treated', 'control'),
    ('LP231 vs HP231', 'sample', 'N436', 'N231'),
]
MIN_PCT = 10.0
FDR_T = 0.05
LFC_T = 0.5


def hdr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ============================================================ LOAD
hdr("LOAD")
ad = sc.read_h5ad(S1_PATH)
adr = ad.raw.to_adata() if ad.raw is not None else ad.copy()
adr.obs = ad.obs.copy()
adr = adr[adr.obs['leiden'].astype(str) != '8'].copy()
adr.obs['sample'] = adr.obs['sample_id'].astype(str)
adr.obs['treat'] = np.where(adr.obs['sample'] == 'C231', 'control', 'treated')
print(f"cells (cluster 8 excluded): {adr.n_obs}  |  " +
      ", ".join(f"{DISP[s]}={int((adr.obs['sample'] == s).sum())}" for s in SAMPLES))
sc.pp.filter_genes(adr, min_cells=10)
print(f"genes tested (detected in >=10 cells): {adr.n_vars}")

all_panel_genes = [g for _, _, groups in PANELS for _, gl in groups for g in gl]
missing = [g for g in all_panel_genes if g not in adr.var_names]
if missing:
    print(f"NOT IN MATRIX (never detected): {missing}")

# ============================================================ PER-SAMPLE MEAN / PCT
X = adr.X


def col(g):
    v = X[:, adr.var_names.get_loc(g)]
    return v.toarray().ravel() if hasattr(v, 'toarray') else np.asarray(v).ravel()


mean = {}; pct = {}
for g in all_panel_genes:
    if g not in adr.var_names:
        mean[g] = {s: 0.0 for s in SAMPLES}; pct[g] = {s: 0.0 for s in SAMPLES}
        continue
    v = col(g)
    mean[g] = {s: float(v[(adr.obs['sample'] == s).values].mean()) for s in SAMPLES}
    pct[g] = {s: float((v[(adr.obs['sample'] == s).values] > 0).mean() * 100) for s in SAMPLES}

# ============================================================ DGE (4 contrasts)
hdr("DGE  Wilcoxon + BH-FDR")
stats = {}
for lab, gcol, grp, ref in CONTRASTS:
    sub = adr[adr.obs[gcol].isin([grp, ref])].copy()
    sc.tl.rank_genes_groups(sub, gcol, groups=[grp], reference=ref,
                            method='wilcoxon', use_raw=False, tie_correct=True)
    df = sc.get.rank_genes_groups_df(sub, group=grp).set_index('names')
    stats[lab] = df
    n_pass = ((df['pvals_adj'] < FDR_T) & (df['logfoldchanges'].abs() >= LFC_T)).sum()
    cnt = {t: int(((df['pvals_adj'] < FDR_T) & (df['logfoldchanges'].abs() >= t)).sum()) for t in (0.25, 0.5, 1.0)}
    print(f"  {lab:16s}: genome-wide passing FDR<{FDR_T} at |log2FC| >=0.25: {cnt[0.25]}  >=0.5: {cnt[0.5]}  >=1.0: {cnt[1.0]}  (of {len(df)})")

# ============================================================ TABLE
rows = []
for pl, ptitle, groups in PANELS:
    for gname, gl in groups:
        for g in gl:
            r = {'panel': pl, 'panel_title': ptitle, 'group': gname, 'gene': g}
            for s in SAMPLES:
                r[f'pct_{DISP[s]}'] = round(pct[g][s], 1)
                r[f'mean_{DISP[s]}'] = round(mean[g][s], 4)
            r['plotted'] = max(pct[g].values()) >= MIN_PCT
            for lab, *_ in CONTRASTS:
                df = stats[lab]
                if g in df.index:
                    lfc = float(df.loc[g, 'logfoldchanges']); q = float(df.loc[g, 'pvals_adj'])
                else:
                    lfc, q = np.nan, np.nan
                r[f'log2FC | {lab}'] = lfc
                r[f'FDR | {lab}'] = q
                r[f'pass | {lab}'] = bool(q < FDR_T and abs(lfc) >= LFC_T) if not np.isnan(q) else False
            rows.append(r)
tab = pd.DataFrame(rows)
os.makedirs(f"{OUT}/dge", exist_ok=True)
tab_path = f"{OUT}/dge/dge_panel_stats_v2.csv"
tab.to_csv(tab_path, index=False)
print(f"\nsaved table: {tab_path}")

dropped = tab.loc[~tab['plotted'], 'gene'].tolist()
print(f"not plotted (<{MIN_PCT:.0f}% cells in every subline): {dropped}")

hdr("PANEL GENES PASSING (FDR<0.05 & |log2FC|>=0.25)")
for lab, *_ in CONTRASTS:
    p = tab[tab['plotted'] & tab[f'pass | {lab}']]
    up = p[p[f'log2FC | {lab}'] > 0]['gene'].tolist()
    dn = p[p[f'log2FC | {lab}'] < 0]['gene'].tolist()
    print(f"  {lab:16s}  UP: {up}\n  {'':16s}  DOWN: {dn}")

# ============================================================ DRIVER CHECK
hdr("DRIVER CHECK  (velocity genes / consensus / rho vs latent time)")
try:
    cons = pd.read_csv(CONS_PATH).set_index('gene')
except Exception as e:
    cons = pd.DataFrame(); print(f"  consensus table not read: {e}")
try:
    ranked = pd.read_csv(RANKED_PATH)
    ranked = ranked.sort_values('fit_likelihood', ascending=False).reset_index(drop=True)
    vrank = {g: i + 1 for i, g in enumerate(ranked['gene'])}
except Exception as e:
    vrank = {}; print(f"  velocity ranking not read: {e}")
try:
    lt = pd.read_csv(LT_PATH).set_index('gene')['spearman_rho']
except Exception as e:
    lt = pd.Series(dtype=float); print(f"  latent-time table not read: {e}")
print(f"  {'gene':9s} {'vel_rank':>8s} {'n_var/7':>8s} {'rho_LT':>8s}")
for g in tab.loc[tab['plotted'], 'gene']:
    vr = vrank.get(g, '-')
    nv = int(cons.loc[g, 'n_variants']) if (len(cons) and g in cons.index) else '-'
    rho = f"{lt[g]:+.3f}" if g in lt.index else '-'
    print(f"  {g:9s} {str(vr):>8s} {str(nv):>8s} {rho:>8s}")
print(f"  (velocity genes total: {len(vrank)}; consensus drivers = n_var >= 6)")

# ============================================================ FIGURE
hdr("RENDER")
CM = 1 / 2.54
fig = plt.figure(figsize=(17.5 * CM, 17.0 * CM))
plot_groups = {}
for pl, ptitle, groups in PANELS:
    keep = []
    for gname, gl in groups:
        gl2 = [g for g in gl if max(pct[g].values()) >= MIN_PCT]
        if gl2:
            keep.append((gname, gl2))
    plot_groups[pl] = (ptitle, keep)
nA = sum(len(gl) for _, gl in plot_groups['A'][1])
nB = sum(len(gl) for _, gl in plot_groups['B'][1])

gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.62,
                      left=0.11, right=0.90, top=0.90, bottom=0.10)
gs_top = gs[0].subgridspec(1, 2, width_ratios=[max(nA, 3), max(nB, 3)], wspace=0.12)
cells = {'A': gs_top[0, 0], 'B': gs_top[0, 1], 'C': gs[1]}

all_means = [mean[g][s] for _, (_, kp) in plot_groups.items() for _, gl in kp for g in gl for s in SAMPLES]
norm_m = Normalize(0, np.percentile(all_means, 95))
cmap_m = plt.cm.Reds
norm_fc = Normalize(-1, 1)
cmap_fc = plt.cm.RdBu_r


def size(p):
    return 4 + 70 * (p / 100.0)


def stars(q):
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''


def draw_panel(cell, letter, title, groups, show_ylabels):
    genes = [g for _, gl in groups for g in gl]
    n = len(genes)
    sg = cell.subgridspec(2, 1, height_ratios=[3, 4], hspace=0.08)
    axd = fig.add_subplot(sg[0]); axh = fig.add_subplot(sg[1], sharex=axd)
    for yi, s in enumerate(SAMPLES):
        for xi, g in enumerate(genes):
            axd.scatter(xi, yi, s=size(pct[g][s]), c=[cmap_m(norm_m(mean[g][s]))],
                        edgecolors='#555555', linewidths=0.3, zorder=3)
    axd.set_ylim(len(SAMPLES) - 0.5, -0.5)
    axd.set_yticks(range(len(SAMPLES)))
    axd.set_yticklabels([DISP[s] for s in SAMPLES] if show_ylabels else [''] * 3)
    if show_ylabels:
        for tl, s in zip(axd.get_yticklabels(), SAMPLES):
            tl.set_color(SAMPLE_COLORS[s]); tl.set_fontweight('bold')
    axd.tick_params(axis='x', labelbottom=False, length=0)
    axd.tick_params(axis='y', length=0)
    for sp in axd.spines.values():
        sp.set_color('#AAAAAA'); sp.set_linewidth(0.4)
    # group brackets above
    x0 = 0
    for gname, gl in groups:
        x1 = x0 + len(gl) - 1
        axd.plot([x0 - 0.35, x1 + 0.35], [-0.85, -0.85], color='#444444', lw=0.6, clip_on=False)
        axd.text((x0 + x1) / 2, -1.0, textwrap.fill(gname, max(12, 7 * len(gl))),
                 ha='center', va='bottom', fontsize=5.3, fontweight='bold', clip_on=False,
                 linespacing=0.9)
        x0 = x1 + 1
    # heatmap
    M = np.full((len(CONTRASTS), n), np.nan)
    for ci, (lab, *_) in enumerate(CONTRASTS):
        for xi, g in enumerate(genes):
            M[ci, xi] = tab.loc[tab.gene == g, f'log2FC | {lab}'].values[0]
    rgba = np.ones((len(CONTRASTS), n, 4)); rgba[..., :3] = 0.93   # not passing = light grey
    for ci, (lab, *_) in enumerate(CONTRASTS):
        for xi, g in enumerate(genes):
            if tab.loc[tab.gene == g, f'pass | {lab}'].values[0]:
                rgba[ci, xi] = cmap_fc(norm_fc(M[ci, xi]))
    axh.imshow(rgba, aspect='auto', interpolation='nearest',
               extent=(-0.5, n - 0.5, len(CONTRASTS) - 0.5, -0.5))
    for ci, (lab, *_) in enumerate(CONTRASTS):
        for xi, g in enumerate(genes):
            row = tab.loc[tab.gene == g].iloc[0]
            lfc = row[f'log2FC | {lab}']; q = row[f'FDR | {lab}']; ok = row[f'pass | {lab}']
            if np.isnan(lfc):
                continue
            txt = f"{lfc:+.2f}" + (f"\n{stars(q)}" if ok else "")
            col_t = ('white' if abs(lfc) > 0.65 else 'black') if ok else '#8C8C8C'
            axh.text(xi, ci, txt, ha='center', va='center', fontsize=4.6,
                     color=col_t, linespacing=0.8)
    axh.set_yticks(range(len(CONTRASTS)))
    axh.set_yticklabels([lab for lab, *_ in CONTRASTS] if show_ylabels else [''] * 4, fontsize=5.6)
    axh.set_xticks(range(n))
    axh.set_xticklabels(genes, rotation=45, ha='right', fontstyle='italic', fontsize=6)
    axh.tick_params(length=0)
    for xs in np.arange(-0.5, n, 1):
        axh.axvline(xs, color='white', lw=0.5)
    for ys in np.arange(-0.5, len(CONTRASTS), 1):
        axh.axhline(ys, color='white', lw=0.5)
    for sp in axh.spines.values():
        sp.set_visible(False)
    axd.set_xlim(-0.6, n - 0.4)
    axd.set_title(title, loc='left', fontsize=7.5, fontweight='bold', pad=22)
    axd.text(-0.02, 1.0, letter, fontsize=11, fontweight='bold', va='bottom', ha='right',
             transform=axd.transAxes + matplotlib.transforms.ScaledTranslation(-0.30, 0.30, fig.dpi_scale_trans))
    return axd, axh


axA = draw_panel(cells['A'], 'A', plot_groups['A'][0], plot_groups['A'][1], True)
axB = draw_panel(cells['B'], 'B', plot_groups['B'][0], plot_groups['B'][1], False)
axC = draw_panel(cells['C'], 'C', plot_groups['C'][0], plot_groups['C'][1], True)

# legends (right margin)
lx = 0.915
cax1 = fig.add_axes([lx, 0.70, 0.012, 0.13])
cb1 = fig.colorbar(ScalarMappable(norm_m, cmap_m), cax=cax1)
cb1.set_label('Mean expr.\n(log-norm)', fontsize=5.6); cb1.ax.tick_params(labelsize=5)
cax2 = fig.add_axes([lx, 0.18, 0.012, 0.16])
cb2 = fig.colorbar(ScalarMappable(norm_fc, cmap_fc), cax=cax2, extend='both')
cb2.set_label('log2FC', fontsize=5.6); cb2.ax.tick_params(labelsize=5)
axs = fig.add_axes([lx - 0.005, 0.42, 0.08, 0.18]); axs.axis('off')
axs.text(0.0, 1.0, '% cells', fontsize=5.6, va='top')
for i, p in enumerate([10, 40, 80]):
    axs.scatter(0.15, 0.72 - i * 0.24, s=size(p), c='#CCCCCC', edgecolors='#555555', linewidths=0.3)
    axs.text(0.45, 0.72 - i * 0.24, f'{p}%', fontsize=5.2, va='center')
axs.set_xlim(0, 1); axs.set_ylim(0, 1)

foot = (f"Wilcoxon rank-sum per cell, BH-FDR over {adr.n_vars:,} genes; cluster 8 excluded. "
        f"Value = log2FC; stars = FDR (* <0.05, ** <0.01, *** <0.001) shown only when "
        f"|log2FC| ≥ {LFC_T}; grey cells = not passing (value shown for reference). ")
if dropped:
    foot += f"Not shown (<{MIN_PCT:.0f}% cells in every subline): {', '.join(dropped)}."
fig.text(0.11, 0.015, textwrap.fill(foot, 175), fontsize=5.2, color='#444444', va='bottom')

for ext, kw in [('png', {'dpi': 200}), ('tiff', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}}),
                ('pdf', {})]:
    os.makedirs(f"{OUT}/{ext}", exist_ok=True)
    p = f"{OUT}/{ext}/Figure_9_DGE_v2.{ext if ext != 'png' else 'png'}"
    if ext == 'png':
        p = f"{OUT}/png/Figure_9_DGE_v2_preview.png"
    fig.savefig(p, bbox_inches='tight', pad_inches=0.05, **kw)
    print(f"saved {p}")
plt.close()
print("\nDONE.")
