# Niraparib resistance in MDA-MB-231: single-cell transcriptomics and RNA velocity

Analysis code for the manuscript

> **Functional and single-cell transcriptomic characterization of niraparib resistance following long-term PARP inhibition in triple-negative breast cancer**
> (authors, journal and DOI to be added on publication)

The study profiles three MDA-MB-231 sublines by 10x Genomics scRNA-seq (17,953 cells):
parental cells and two niraparib-resistant sublines that share a common selection history
and differ in their long-term maintenance concentration. The code covers preprocessing and
clustering, dynamical RNA velocity (scVelo), CellRank fate analysis, a seven-configuration
parameter sweep defining the consensus driver genes, latent-time correlation and enrichment,
NMF decomposition, the cluster-6 cell-cycle analysis, targeted differential expression of
drug-transporter, EMT and DNA-damage-response genes, and validation in an independent bulk
RNA-seq dataset (GSE297522).

## Sample names

The data objects use the original sample labels. Figures and the manuscript use the final names.

| In data / code | In manuscript | Description | Colour |
|---|---|---|---|
| `C231` | **NP231** | parental, niraparib-naïve | `#0000FF` |
| `N231` | **HP231** | resistant, maintained at the higher niraparib concentration | `#FF8000` |
| `N436` | **LP231** | resistant, maintained at the lower niraparib concentration | `#00C000` |

All figure scripts map the labels through a `DISP` dictionary.

## Repository layout

```
analysis/
  01_preprocessing/            QC, normalisation, HVG, PCA, Leiden clustering, cluster markers
  02_velocity_cellrank/        scVelo dynamical model, CellRank GPCCA, velocity QC, negative controls
  03_parameter_sweep/          seven velocity configurations and the consensus driver set
  04_latent_time_enrichment/   gene–latent-time correlation, ORA (Enrichr) and prerank GSEA
  05_nmf/                      NMF decomposition of the velocity signal
  06_cluster6/                 cell-cycle regression and sub-state analysis of cluster 6 (LP231)
  07_validation_GSE297522/     PyDESeq2 analysis of the independent bulk RNA-seq dataset
figures/                       one script per manuscript figure (see docs/FIGURE_MAP.md)
docs/                          figure map, analysis parameters
env/                           conda environment files
```

## Data

| Data | Location |
|---|---|
| Raw FASTQ, Cell Ranger filtered matrices | GEO, accession to be added |
| Processed AnnData objects and result tables | Zenodo, DOI to be added |
| GSE297522 count table | [GEO GSE297522](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE297522) |

Large files (`.h5ad`, `.loom`, `.mtx`, count tables) are not stored in this repository.

## Running the code

1. Create the environments (see `env/README.md`):

   ```bash
   conda env create -f env/b2bio.yml     # scanpy, scVelo, CellRank, gseapy, plotting
   conda env create -f env/deseq.yml     # PyDESeq2 (validation dataset only)
   ```

2. Point the scripts to the data root. All paths are built from one variable:

   ```bash
   export DATA_ROOT=/path/to/project_root
   ```

   The expected structure under `DATA_ROOT` mirrors the Zenodo archive
   (`scanpy_out_MDA231/`, `velocity/03_outputs/`, `validation_datasets/`, `figures/`).

3. Run the analysis steps in numerical order, then the figure scripts:

   ```bash
   conda activate b2bio
   python figures/Figure_1_master_v2.py      # manuscript Figure 3
   ...
   conda run -n deseq python analysis/07_validation_GSE297522/validation_GSE297522_deseq2.py
   ```

   Figure scripts write PNG/PDF and 600-dpi TIFF files to `$DATA_ROOT/figures/`.

## Reference objects and known gaps

- **Main velocity run.** The interactive session that produced the main velocity/CellRank object
  (`merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad`) was not saved as a script. This object
  is deposited on Zenodo and is the reference input for all figures.
  `analysis/02_velocity_cellrank/reconstruct_velocity_main.py` re-runs scVelo with the parameters
  recorded in the object and compares the result with it. The reconstruction recovers exactly the
  same 92 fitted genes and 66 velocity genes (stochastic velocity followed by the dynamical model on
  the stochastic velocity genes), with per-cell Spearman ρ = 0.97 for latent time and 0.92 for
  velocity length. It is not bit-identical, so figures should be regenerated from the deposited object.
- **CellRank.** VelocityKernel → GPCCA, 5 macrostates, terminal states M0 and M3 (parameters in
  `docs/PARAMETERS.md`); the fitted estimator output is stored in the deposited object.
- **NMF.** The script that computed the NMF decomposition is not yet included; its loadings are
  deposited on Zenodo.

## Statistical note

Each subline was profiled as a single culture. Cells are therefore not independent biological
replicates and per-cell p-values are inflated. The scripts report effect sizes (log2 fold-change,
Cliff's δ) alongside p-values, and targeted differential expression uses Benjamini–Hochberg FDR
together with a fold-change threshold (|log2FC| ≥ 0.5) as the primary criterion.

## Software

Python 3, Scanpy 1.11.5, scVelo 0.3.3, velocyto.py 0.17.17, CellRank 2.0.7, gseapy 1.2.1,
scikit-learn (NMF), SciPy, PyDESeq2 0.5.4. Exact versions are pinned in `env/`.
Analysis parameters are listed in `docs/PARAMETERS.md`.

The main velocity object differs from the reference variant of the parameter sweep: the main run
used moments on 30 neighbours (66 velocity genes), whereas the sweep reference uses 10 neighbours
(44 velocity genes). The sweep is a separate robustness analysis of the driver genes.

## Citation

Until the article is published, please cite this repository (see `CITATION.cff`). The article reference will be added on publication.

## License

MIT, see `LICENSE`.
