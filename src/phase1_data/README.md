# Phase 1: Real-World Data Acquisition & Preprocessing

This module programmatically acquires and preprocesses transcriptomic datasets from specialized biological archives.

## Data Sources

1.  **TCGA-BRCA**: Acquired from the Genomic Data Commons (GDC) API. Contains 150+ real-world tumor samples from breast cancer patients.
2.  **GTEx Normal Breast Baseline**: Fetched from the official GTEx Portal GCS bucket using MAMMARY TISSUE as a healthy baseline.
3.  **Hugo Symbol Alignment**: All Ensembl IDs are mapped to Hugo Symbols to ensure accurate Protein-Protein Interaction (PPI) network construction.

## Preprocessing Pipeline

*   **Wait-list Filtering**: Removes genes with extremely low expression or zero variance across samples.
*   **Variance Funnel**: Subsets the 55,000+ available genes to the top 2,000 highest biological variance genes, plus key cancer markers, to optimize graph complexity.
*   **Normalization**: Log1p transformation followed by sample-wise Z-score scaling.
*   **Dataset Partitioning**: GTEx samples are split 80/20 for training and validation purposes.

## Deliverables

- `data/tcga_brca.csv`: Processed cancer expression matrix.
- `data/gtex_train.csv` / `data/gtex_val.csv`: Healthy expression matrices.
- `data/pre_ppi_gene_list.json`: Aligned identifier list.
