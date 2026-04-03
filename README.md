# OncoNexus: Cancer Biomarker Detection via Graph Neural Networks

OncoNexus is an advanced bioinformatics pipeline designed to identify perturbed gene biomarkers in breast cancer through Graph Neural Networks (GNNs). The system employs a self-supervised learning architecture, specifically Deep Graph Infomax (DGI) combined with Graph Attention Networks (GATv2), to model high-dimensional transcriptomic data within the context of biological protein-protein interaction networks.

## Core Architecture

The pipeline is engineered to overcome traditional hardware limitations (e.g., 4GB VRAM GPU constraints) while maintaining biological integrity. It distinguishes between healthy (GTEx) and cancerous (TCGA-BRCA) tissue samples to isolate topological shifts in regulatory pathways.

### Key Technical Features

*   **Self-Supervised Learning**: Utilizes Deep Graph Infomax (DGI) to learn structural node representations without the need for labeled edge reconstruction.
*   **Graph Attention Mechanisms**: Employs GATv2 layers to dynamically weigh the importance of protein-protein interactions based on learned biological relevance.
*   **Hardware Optimization**: Optimized for local GPU execution (NVIDIA RTX 2050) using micro-batching and low-complexity discriminators.
*   **Supervised Discovery**: Integrates an XGBoost/Random Forest classification head on top of the latent GNN embeddings to rank biomarkers based on their contribution to disease state classification.
*   **Medical Data Integration**: Programmatic acquisition of real-world data from the Genomic Data Commons (GDC) and GTEx Portal APIs.

## Project Structure

*   `data/`: Contains all intermediate and final biological datasets (CSV, JSON, PT).
*   `models/`: Stores trained GNN weights and classification models.
*   `logs/`: Training convergence and validation metrics.
*   `src/`:
    *   `phase1_data/`: Data acquisition, normalization, and variance funneling.
    *   `phase2_network/`: PPI graph construction via STRING API.
    *   `phase3_train/`: Self-supervised GNN training (DGI + GATv2).
    *   `phase4_perturbation/`: Supervised biomarker extraction and topological importance analysis.
    *   `phase5_biomarkers/`: Validation against COSMIC and DisGeNET ground truths.

## Installation

Ensure a Python environment (3.10+) with CUDA-enabled PyTorch is configured.

```bash
# Core Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric pandas scikit-learn networkx requests xgboost
```

## Usage

The entire pipeline can be orchestrated through the central CLI:

```bash
# Execute the full pipeline
python run_pipeline.py --phase all

# Execute specific phases
python run_pipeline.py --phase 3
```

