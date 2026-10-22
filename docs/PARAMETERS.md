# Analysis parameters and implementation details

Details that are needed to reproduce the analysis but are not part of the manuscript text.
Scientific parameters are also given in the Supplementary Methods.

## Preprocessing and clustering (Scanpy 1.11.5)

- Input: Cell Ranger v7.0.0 `filtered_feature_bc_matrix.h5` for C231, N231, N436; concatenated into one AnnData object, sample label in `obs['sample']`.
- No additional cell filters; QC metrics `n_genes_by_counts`, `total_counts`, `pct_counts_mt` computed per cell.
- `sc.pp.normalize_total(target_sum=1e4)`, `sc.pp.log1p`; full log-normalised matrix (36,601 genes) stored in `adata.raw`.
- `sc.pp.highly_variable_genes(flavor='seurat', n_top_genes=2000)`, `sc.pp.scale(max_value=10)`.
- `sc.tl.pca` (default), `sc.pp.neighbors(n_neighbors=10, n_pcs=30)`, `sc.tl.umap` (default), `sc.tl.leiden(resolution=0.5)`.
- No batch correction.
- Cluster 8 (n = 418; median `pct_counts_mt` 93.4%) flagged as low quality and excluded from quantitative panels.
- Markers (Figure 3C): `sc.tl.rank_genes_groups(method='wilcoxon', corr_method='benjamini-hochberg', use_raw=True)` on the full log-normalised transcriptome; top 100 genes per cluster exported (`markers_allclusters_top100_RAW.csv`). Script: `analysis/01_preprocessing/stage3_cluster_markers_RAW.py`.

## RNA velocity and CellRank

- velocyto.py 0.17.17 on Cell Ranger BAM files → one `.loom` per sample; merged with the Scanpy object on barcodes.
- scVelo 0.3.3 (main run, values recorded in the deposited object or confirmed by reconstruction):
  `scv.pp.filter_and_normalize(min_shared_counts=20, n_top_genes=2000)`;
  `scv.pp.moments(n_pcs=30, n_neighbors=30)`;
  `scv.tl.velocity(mode='stochastic')`, which defines the candidate genes;
  `scv.tl.recover_dynamics` on those genes (92 fitted);
  `scv.tl.velocity(mode='dynamical')` (66 velocity genes, fit_likelihood > 0.1);
  `scv.tl.velocity_graph`; `scv.tl.latent_time` (default root/end-point detection); `scv.tl.velocity_confidence`.
- Velocity length = Euclidean norm of each cell's row in the `velocity` layer.
- CellRank 2.0.7: `VelocityKernel` → `GPCCA`; 5 macrostates; terminal states M0 (cluster-8 cohort) and M3 (LP231-enriched); absorption probabilities stored as `obs['fate_0']`, `obs['fate_3']`.
- The UMAP used in all figures is `obsm['X_umap']` of `merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad`.

## Parameter sweep

Separate from the main run. Each variant re-runs `filter_and_normalize(min_shared_counts=20)`, `moments`, `recover_dynamics` (without the stochastic pre-step), `velocity(mode='dynamical')`, `velocity_graph`, `latent_time`.


| Variant | n_neighbors | n_pcs | n_top_genes |
|---|---|---|---|
| ref | 10 | 30 | 2000 |
| v1 | 15 | 30 | 2000 |
| v2 | 30 | 30 | 2000 |
| v3 | 10 | 20 | 2000 |
| v4 | 10 | 50 | 2000 |
| v5 | 10 | 30 | 1500 |
| v6 | 10 | 30 | 3000 |

Consensus drivers: genes in the top-likelihood lists of ≥ 6/7 variants (n = 17); ≥ 7/7 gives 4 genes, ≥ 5/7 gives 27.

## Enrichment

- ORA: `gseapy.enrichr`, libraries MSigDB_Hallmark_2020, GO_Biological_Process_2023, Reactome_2022, KEGG_2021_Human.
- GSEA: `gseapy.prerank`, ranking = Spearman ρ (expression vs latent time), permutations = 1000, min_size = 15, max_size = 500, seed = 42.

## Targeted differential expression

- `sc.tl.rank_genes_groups(method='wilcoxon', tie_correct=True)` on `.raw`, genes detected in ≥ 10 cells (23,086), cluster 8 excluded.
- Contrasts: N231 vs C231, N436 vs C231, (N231 + N436) vs C231, N436 vs N231.
- Thresholds: FDR < 0.05 and |log2FC| ≥ 0.5; genes detected in < 10% of cells in every subline not shown.

## Statistics in figures

- Two-sided Wilcoxon rank-sum (`scipy.stats.ranksums`); Cliff's δ as effect size.
- p-values below 1×10⁻³⁰ displayed as `p < 1×10⁻³⁰`.

## Validation dataset (GSE297522)

- MDA-MB-231 vehicle samples only: parental vs niraparib-, olaparib-, talazoparib-resistant (n = 3 each).
- Duplicated gene symbols summed; genes with ≥ 10 counts in ≥ 3 samples (16,520).
- PyDESeq2 0.5.4, Wald test, design `~condition`, reference = parental; mean CPM < 1 in every group = not detected.
