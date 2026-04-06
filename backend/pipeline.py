import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, DeepGraphInfomax
import pandas as pd
import numpy as np
import json
import os

# 1. Architecture Definitions (Must match exactly with Phase 3)
class GATEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, out_channels=128):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=False)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))

class OncoGNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, out_channels=128):
        super(OncoGNN, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels)
        self.dgi = DeepGraphInfomax(
            hidden_channels=out_channels,
            encoder=self.encoder,
            summary=summary,
            corruption=corruption
        )

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

def run_inference(df, model, genes, edge_index, device='cpu'):
    """
    Optimized inference function.
    genes and edge_index are pre-computed and passed from the app cache.
    """
    # 1. Preprocess input DataFrame (Align genes)
    if 'gene' in df.columns and 'expression' in df.columns:
        input_data = df.set_index('gene')['expression'].reindex(genes).fillna(0).values
    else:
        # If columns are genes
        input_data = df.reindex(columns=genes).fillna(0).mean().values

    x = torch.tensor(input_data, dtype=torch.float).to(device)
    if x.dim() == 1:
        x = x.unsqueeze(1)

    # 2. GNN Inference - Graph-level Topological Mapping
    model.eval()
    with torch.no_grad():
        # Get latent embeddings [num_genes, 128]
        z = model(x, edge_index)
        
    # 3. Extract results
    z_mag = torch.norm(z, dim=1).cpu().numpy()
    # Safe normalization
    z_min, z_max = z_mag.min(), z_mag.max()
    z_mag_norm = (z_mag - z_min) / (z_max - z_min + 1e-9)
    
    # Map back to genes
    scores_dict = [{"gene": g, "score": float(s)} for g, s in zip(genes, z_mag_norm)]
    
    # Calculate overall risk
    high_perturbations = [s for s in z_mag_norm if s > 0.6]
    risk_score = float(np.mean(high_perturbations)) if high_perturbations else float(np.mean(z_mag_norm))

    # Identify top biomarkers for the response
    top_biomarkers = sorted(scores_dict, key=lambda x: x['score'], reverse=True)[:20]

    return {
        "risk_score": risk_score,
        "perturbation_scores": scores_dict,
        "top_biomarkers": top_biomarkers,
        "status": "success"
    }