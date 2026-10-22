"""
Figure 1 (script numbering) = manuscript Figure 3 — Composite Master v2
=======================================================================
v2 (20.10.2026), per team corrections:
  - old panel A (schematic) removed: duplicates wet-lab Fig 1A
  - old panel F (functional compass) removed: DGE genes go to the new DGE figure
  - new sample names + colors: NP231 #0000FF, HP231 #FF8000, LP231 #00C000
  - cluster palette: dominant cluster of each line = line color, rare clusters highlighted
  - heatmap cluster numbers on top

Panels:
  A — UMAP colored by sample (NP231/HP231/LP231)
  B — UMAP colored by Leiden cluster (12 clusters)
  C — Marker gene heatmap (top 5 per cluster x 12 clusters)
  D — Stacked bar of cluster composition per sample

Data keeps sample_id = C231/N231/N436; display names come from DISP.
Set SYNTH=1 to render with synthetic data (layout check only).

Run:
    conda run --no-capture-output -n b2bio python Figure_1_master_v2.py
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SYNTH = os.environ.get("SYNTH") == "1"

plt.rcParams.update({
    'font.family':     'DejaVu Sans' if SYNTH else 'Arial',
    'font.size':       7,
    'axes.linewidth':  0.5,
    'axes.titlesize':  8,
    'axes.labelsize':  7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
    'svg.fonttype':    'none',
})

# ============================================================
# PALETTES + DISPLAY NAMES
# ============================================================
SAMPLES = ['C231', 'N231', 'N436']
DISP = {'C231': 'NP231', 'N231': 'HP231', 'N436': 'LP231'}
SAMPLE_PALETTE = {'C231': '#0000FF', 'N231': '#FF8000', 'N436': '#00C000'}

CLUSTER_PALETTE = {
    '2': '#0000FF', '3': '#5B8DEF', '4': '#9DBBFF',      # NP231 family (2 dominant)
    '1': '#FF8000', '5': '#FFC266',                      # HP231 family (1 dominant)
    '0': '#00C000', '6': '#005A00',                      # LP231 family (0 dominant, 6 = G2/M)
    '7': '#8C564B', '9': '#D4A6C8',                      # mixed
    '10': '#E6007E', '11': '#E41A1C',                    # rare, highlighted
    '8': '#BDBDBD',                                      # QC
}
CLUSTER_ORDER = [str(i) for i in range(12)]

# ============================================================
# DATA
# ============================================================
if not SYNTH:
    import scanpy as sc
    CR_PATH = DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad"
    S1_PATH = DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad"
    S1_FATE_PATH = DATA_ROOT + "/velocity/03_outputs/fate0_analysis/stage1_with_cellrank_fates.h5ad"
    MARKERS_CSV = DATA_ROOT + "/scanpy_out_MDA231/tables_stage3_RAW/markers_allclusters_top100_RAW.csv"

    print("Loading data...")
    ad_cr = sc.read_h5ad(CR_PATH)
    ad_s1 = sc.read_h5ad(S1_PATH)
    ad_fate = sc.read_h5ad(S1_FATE_PATH)
    markers_df = pd.read_csv(MARKERS_CSV)
    markers_df['cluster'] = markers_df['cluster'].astype(str)

    def normalize_cr(bc):
        sample = bc.split('_')[0]
        barcode = bc.split(':')[1].rstrip('x')
        return f"{barcode}-{sample}"

    def normalize_s1(bc):
        parts = bc.split('-')
        return f"{parts[0]}-{parts[-1]}"

    cr_keys = pd.Index([normalize_cr(bc) for bc in ad_cr.obs_names])
    s1_keys = pd.Index([normalize_s1(bc) for bc in ad_fate.obs_names])
    s1_idx_map = pd.Series(index=s1_keys, data=range(len(s1_keys)))
    mapped = np.array([s1_idx_map[k] for k in cr_keys])

    samples_cr = ad_fate.obs['sample_id'].values[mapped].astype(str)
    leiden_cr = pd.Series(ad_fate.obs['leiden'].values[mapped]).astype(str).values
    umap_cr = ad_cr.obsm['X_umap']
    print(f"  Cells: {len(samples_cr)}")

    # Heatmap: top 5 markers per cluster, mean expr, z-score per gene
    top = (markers_df.sort_values(['cluster', 'scores'], ascending=[True, False])
                     .groupby('cluster').head(5))
    heatmap_genes = []
    for c in CLUSTER_ORDER:
        for g in top[top['cluster'] == c]['names'].tolist():
            if g not in heatmap_genes:
                heatmap_genes.append(g)
    if ad_s1.raw is not None:
        var_names, X = ad_s1.raw.var_names, ad_s1.raw.X
    else:
        var_names, X = ad_s1.var_names, ad_s1.X
    leiden_s1 = ad_s1.obs['leiden'].astype(str).values
    expr = pd.DataFrame(0.0, index=heatmap_genes, columns=CLUSTER_ORDER)
    for c in CLUSTER_ORDER:
        m = leiden_s1 == c
        if m.sum() == 0:
            continue
        for g in heatmap_genes:
            if g not in var_names:
                continue
            v = X[m, var_names.get_loc(g)]
            v = v.toarray().ravel() if hasattr(v, 'toarray') else np.asarray(v).ravel()
            expr.loc[g, c] = float(v.mean())
    zs = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1) + 1e-9, axis=0).clip(-2, 2)

    # Stacked bar
    ct = pd.crosstab(ad_fate.obs['sample_id'].astype(str).values,
                     ad_fate.obs['leiden'].astype(str).values).reindex(SAMPLES)[CLUSTER_ORDER]
else:
    rng = np.random.default_rng(0)
    centers = {'C231': (-2, 2), 'N231': (2.5, 1), 'N436': (0, -3)}
    ns = {'C231': 6547, 'N231': 5073, 'N436': 6333}
    fam = {'C231': (['2', '3', '4', '7', '8', '9'], [.42, .23, .2, .05, .05, .05]),
           'N231': (['1', '5', '7', '8', '9', '10', '11'], [.66, .19, .07, .03, .03, .015, .005]),
           'N436': (['0', '6', '7', '8'], [.8, .13, .04, .03])}
    samples_cr, leiden_cr, U = [], [], []
    for s in SAMPLES:
        samples_cr += [s] * ns[s]
        leiden_cr += list(rng.choice(fam[s][0], ns[s], p=fam[s][1]))
        U.append(rng.normal(centers[s], 1.0, size=(ns[s], 2)))
    samples_cr = np.array(samples_cr); leiden_cr = np.array(leiden_cr); umap_cr = np.vstack(U)
    heatmap_genes = [f"GENE{i}" for i in range(57)]
    zs = pd.DataFrame(rng.normal(0, 0.8, (57, 12)).clip(-2, 2), index=heatmap_genes, columns=CLUSTER_ORDER)
    ct = pd.crosstab(samples_cr, leiden_cr).reindex(SAMPLES).reindex(columns=CLUSTER_ORDER, fill_value=0)

proportions = ct.div(ct.sum(axis=1), axis=0)
sample_n = ct.sum(axis=1).to_dict()

# ============================================================
# FIGURE LAYOUT
#   left column : A (UMAP sample) over B (UMAP cluster)
#   right column: C (heatmap) over D (stacked bar)
# ============================================================
CM = 1 / 2.54
fig = plt.figure(figsize=(17.5 * CM, 19.0 * CM))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.12,
              left=0.06, right=0.97, top=0.95, bottom=0.06)
gsL = gs[0, 0].subgridspec(2, 2, width_ratios=[1.0, 0.36], hspace=0.22, wspace=0.0)  # col 1 = legend space
gsR = gs[0, 1].subgridspec(2, 2, height_ratios=[2.6, 1.0], width_ratios=[0.30, 1.0], hspace=0.22, wspace=0.0)  # col 0 = label space


def style_umap(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=7, labelpad=2)
    ax.set_ylabel('UMAP 2', fontsize=7, labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888888')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, fontsize=8, fontweight='bold', loc='left', pad=3)


def panel_label(ax, letter, x=-0.12, y=1.02):
    ax.text(x, y, letter, fontsize=11, fontweight='bold', va='bottom', ha='left',
            transform=ax.transAxes)


# ---- A: UMAP by sample
axA = fig.add_subplot(gsL[0, 0])
for s in SAMPLES:
    m = samples_cr == s
    axA.scatter(umap_cr[m, 0], umap_cr[m, 1], c=SAMPLE_PALETTE[s],
                s=0.6, alpha=0.65, linewidths=0, rasterized=True)
style_umap(axA, 'Sample')
axA.legend(handles=[plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=SAMPLE_PALETTE[s],
                               markersize=4.5, markeredgewidth=0,
                               label=f"{DISP[s]}\n(n={(samples_cr == s).sum():,})")
                    for s in SAMPLES],
           loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
           handletextpad=0.3, labelspacing=0.9, borderaxespad=0, fontsize=6.5)
panel_label(axA, 'A')

# ---- B: UMAP by cluster
axB = fig.add_subplot(gsL[1, 0])
for c in CLUSTER_ORDER:
    m = leiden_cr == c
    if m.sum() == 0:
        continue
    axB.scatter(umap_cr[m, 0], umap_cr[m, 1], c=CLUSTER_PALETTE[c],
                s=2.0 if c in ['10', '11'] else 0.6,
                alpha=0.4 if c == '8' else 0.7, linewidths=0, rasterized=True)
style_umap(axB, 'Leiden cluster')
axB.legend(handles=[plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=CLUSTER_PALETTE[c],
                               markersize=4, markeredgewidth=0,
                               label=f"{c} (QC)" if c == '8' else c)
                    for c in CLUSTER_ORDER],
           loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
           handletextpad=0.2, labelspacing=0.3, borderaxespad=0, fontsize=6.5, ncol=1)
panel_label(axB, 'B')

# ---- C: marker heatmap, cluster numbers on top
gsC = gsR[0, 1].subgridspec(2, 2, height_ratios=[0.7, 30], width_ratios=[20, 1],
                            hspace=0.03, wspace=0.06)
axCa = fig.add_subplot(gsC[0, 0]); axC = fig.add_subplot(gsC[1, 0]); axCb = fig.add_subplot(gsC[1, 1])
for i, c in enumerate(CLUSTER_ORDER):
    axCa.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=CLUSTER_PALETTE[c], edgecolor='white', linewidth=0.4))
axCa.set_xlim(0, 12); axCa.set_ylim(0, 1); axCa.set_yticks([])
axCa.set_xticks([i + 0.5 for i in range(12)]); axCa.set_xticklabels(CLUSTER_ORDER, fontsize=6)
axCa.xaxis.tick_top(); axCa.tick_params(axis='x', length=0, pad=1.5)
axCa.xaxis.set_label_position('top'); axCa.set_xlabel('Leiden cluster', fontsize=7, labelpad=3)
for sp in axCa.spines.values():
    sp.set_visible(False)
im = axC.imshow(zs.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
axC.set_xticks([])
axC.set_yticks(range(len(heatmap_genes))); axC.set_yticklabels(heatmap_genes, fontsize=4.5, style='italic')
axC.tick_params(axis='both', length=0, pad=1)
for sp in axC.spines.values():
    sp.set_linewidth(0.4); sp.set_color('#888888')
cb = plt.colorbar(im, cax=axCb)
cb.set_label('z-score', fontsize=6, labelpad=1); cb.ax.tick_params(labelsize=5.5, length=1.5)
cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor('#888888')
panel_label(axCa, 'C', x=-0.30, y=1.6)

# ---- D: stacked bar (colors as in B)
axD = fig.add_subplot(gsR[1, 1])
x = np.arange(len(SAMPLES)); bottom = np.zeros(len(SAMPLES))
for c in CLUSTER_ORDER:
    v = proportions[c].values
    axD.barh(x, v, 0.65, left=bottom, color=CLUSTER_PALETTE[c], edgecolor='white', linewidth=0.3)
    bottom += v
axD.set_yticks(x); axD.set_yticklabels([f"{DISP[s]}\n(n={sample_n[s]:,})" for s in SAMPLES], fontsize=6.5)
axD.invert_yaxis()
axD.set_xlim(0, 1); axD.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axD.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=6)
axD.set_xlabel('Proportion of cells (colors as in B)', fontsize=7, labelpad=2)
for sp in ['top', 'right']:
    axD.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    axD.spines[sp].set_linewidth(0.4); axD.spines[sp].set_color('#888888')
axD.tick_params(axis='both', length=2, width=0.4, color='#888888')
axD.set_title('Cluster composition', fontsize=8, fontweight='bold', loc='left', pad=3)
panel_label(axD, 'D', x=-0.30)

# ============================================================
# SAVE
# ============================================================
OUT = DATA_ROOT + '/figures' if not SYNTH else '.'
for sub in ['pdf', 'tiff', 'png']:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
paths = {
    'pdf':  (os.path.join(OUT, 'pdf', 'Figure_1_v2.pdf'), {}),
    'tiff': (os.path.join(OUT, 'tiff', 'Figure_1_v2.tiff'), {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}}),
    'png':  (os.path.join(OUT, 'png', 'Figure_1_v2_preview.png'), {'dpi': 200}),
}
for ext, (p, kw) in paths.items():
    if SYNTH and ext != 'png':
        continue
    fig.savefig(p, bbox_inches='tight', pad_inches=0.1, **kw)
    print(f"Saved {ext.upper()}: {p}  ({os.path.getsize(p) / 1e6:.2f} MB)")
plt.close()
print("\nDONE.")
