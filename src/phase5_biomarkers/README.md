# Phase 5: Biomarker Validation

This module validates the mathematical candidates identified in Phase 4 by cross-referencing them against established oncology ground truths.

## Clinical Benchmarking

*   **COSMIC & DisGeNET Libraries**: Cross-references our top 50 candidates against 30+ canonical breast cancer genes (e.g., `ATM`, `AKT1`, `CDH1`, `KMT2C`).
*   **Precision@N**: Measures the accuracy of the top 50 identified genes.
*   **Recall@N**: Measures how many of the true primary drivers we successfully recovered from raw, uncurated RNA-Seq data.

## Validation Logic

*   **Discovery Metrics**: Success is defined by high Precision/Recall scores, proving the Deep Graph Infomax (DGI) network successfully isolated the "cancer-driving" topological features.
*   **Topological Significance**: High-ranking biomarkers recovered in this phase are exported directly for frontend dashboard visualization as the primary "targets."

## Deliverables

- `data/biomarker_rankings.csv`: Final ranked list of validated biomarkers.
- `logs/validation_metrics.json`: Final accuracy report for the pipeline.
