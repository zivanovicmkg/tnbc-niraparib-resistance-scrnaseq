import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
# scRNAseq_MDA231_stage1_QC_UMAP_WITHRAW.py

import scanpy as sc
import os

# =====================================================
# INPUT FILES – exact paths on your machine
# =====================================================
SAMPLES = {
    "C231": DATA_ROOT + "/cellranger/C231/filtered_feature_bc_matrix.h5",
    "N231": DATA_ROOT + "/cellranger/N231/filtered_feature_bc_matrix.h5",
    "N436": DATA_ROOT + "/cellranger/N436/filtered_feature_bc_matrix.h5",
}

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================
OUT_DIR = DATA_ROOT + "/scanpy_out_MDA231"
FIG_DIR = os.path.join(OUT_DIR, "figures")
OBJ_DIR = os.path.join(OUT_DIR, "objects")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OBJ_DIR, exist_ok=True)

# =====================================================
# Scanpy settings
# =====================================================
sc.settings.verbosity = 3
sc.settings.figdir = FIG_DIR
sc.settings.set_figure_params(dpi=300, fontsize=10)

# =====================================================
# Load data
# =====================================================
adatas = []

for sample_id, h5_path in SAMPLES.items():
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing file: {h5_path}")

    print(f"\nLoading {sample_id} from:\n  {h5_path}")
    ad = sc.read_10x_h5(h5_path)
    ad.var_names_make_unique()
    ad.obs["sample_id"] = sample_id
    adatas.append(ad)

# Merge into single AnnData
adata = adatas[0].concatenate(
    adatas[1:],
    batch_key="sample_id",
    batch_categories=list(SAMPLES.keys())
)
adata.var_names_make_unique()

# =====================================================
# Metadata (from experimental design)
# =====================================================
condition_map = {
    "C231": "control",
    "N231": "resistant_low",
    "N436": "resistant_high",
}
adata.obs["condition"] = adata.obs["sample_id"].map(condition_map)
adata.obs["cell_line"] = "MDA-MB-231"

# =====================================================
# QC metrics
# =====================================================
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# QC violin plot
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    groupby="sample_id",
    jitter=0.4,
    multi_panel=True,
    save="_QC_violin.png"
)

# =====================================================
# Normalization + log1p
# =====================================================
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# =====================================================
# CRITICAL: store FULL gene matrix here
# After this, we can subset HVGs for PCA/UMAP,
# but we still keep all genes accessible via adata.raw
# =====================================================
adata.raw = adata

# =====================================================
# HVGs + PCA + UMAP (HVG only)
# =====================================================
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# scale() densifies – acceptable here (HVG only)
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)

# =====================================================
# Leiden clustering
# =====================================================
sc.tl.leiden(adata, resolution=0.5)

# =====================================================
# UMAP figures (overview)
# =====================================================
sc.pl.umap(
    adata,
    color="leiden",
    title="UMAP – clusters (Leiden)",
    save="_UMAP_clusters.png"
)
sc.pl.umap(
    adata,
    color="condition",
    title="UMAP – condition",
    save="_UMAP_condition.png"
)
sc.pl.umap(
    adata,
    color="sample_id",
    title="UMAP – samples",
    save="_UMAP_samples.png"
)

# =====================================================
# UMAP per sample (separate panels, robust)
# =====================================================
for sid in ["C231", "N231", "N436"]:
    ad_sub = adata[adata.obs["sample_id"] == sid].copy()
    sc.pl.umap(
        ad_sub,
        color="leiden",
        title=f"UMAP – {sid}",
        save=f"_UMAP_{sid}.png"
    )

# =====================================================
# Save object for Stage 2 (DEGs) + Stage4 scoring via raw
# =====================================================
out_h5ad = os.path.join(OBJ_DIR, "MDA231_stage1_adata_WITHRAW.h5ad")
adata.write(out_h5ad)

print("\nSTAGE 1 (WITH RAW) COMPLETED SUCCESSFULLY")
print("Figures saved to:", FIG_DIR)
print("Object saved to :", out_h5ad)
print("adata shape (HVG):", adata.shape)
print("adata.raw shape (ALL genes):", adata.raw.X.shape if adata.raw is not None else None)
