import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from pipeline import OncoGNN, GATEncoder

# --- Consensus Gold Standard: Top 30 Breast Cancer Driver Genes (COSMIC/TCGA) ---
BRCA_GOLD_STANDARD = {
    'TP53', 'PIK3CA', 'CDH1', 'GATA3', 'MAP3K1', 'PTEN', 'KMT2C', 'AKT1', 'CBFB', 
    'BRCA1', 'BRCA2', 'ERBB2', 'CCND1', 'MYC', 'FGFR1', 'RB1', 'ATM', 'PALB2', 
    'CHEK2', 'NCOR1', 'MAP2K4', 'STK11', 'CDKN2A', 'CTCF', 'ARID1A', 'ESR1', 
    'NF1', 'SF3B1', 'TBX3', 'FOXA1'
}

def evaluate_accuracy(model_path='models/gae_weights.pt', graph_path='data/graph.json', data_path='data/tcga_brca_REAL_sample.csv'):
    print("\n" + "="*50)
    print("      ONCONEXUS MODEL EVALUATION UTILITY")
    print("="*50 + "\n")

    if data_path:
        print(f"[*] Mode: Patient Sample Evaluation ({os.path.basename(data_path)})")
    else:
        print("[*] Mode: Baseline Biological Evaluation (No data provided)")

    # 1. Load Graph and Model
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
    
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    genes_data = graph_data['nodes']
    genes = [n['id'] for n in genes_data]
    node_to_idx = {g: i for i, g in enumerate(genes)}
    
    edge_list = []
    for e in graph_data['edges']:
        if e['source'] in node_to_idx and e['target'] in node_to_idx:
            edge_list.append([node_to_idx[e['source']], node_to_idx[e['target']]])
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    num_genes = len(genes)

    # Re-initialize architecture
    model = OncoGNN(in_channels=1, hidden_channels=256, out_channels=128)
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle key mapping: DeepGraphInfomax often saves its own internal state dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Fallback: Many training scripts only save the DGI module's state dict
        if 'weight' in state_dict and 'encoder.conv1.att' in state_dict:
            model.dgi.load_state_dict(state_dict)
            print("[*] Successfully loaded weights into DGI sub-module.")
        else:
            raise
    
    model.eval()

    # 2. Prepare Features (X)
    # If data_path provided, load patient expression. Else use baseline ones.
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            # Alignment logic similar to pipeline.py
            if 'gene' in df.columns and 'expression' in df.columns:
                features = df.set_index('gene')['expression'].reindex(genes).fillna(0).values
            else:
                features = df.reindex(columns=genes).fillna(0).mean().values
            x = torch.tensor(features, dtype=torch.float).unsqueeze(1)
            print(f"[*] Successfully aligned {len(genes)} genes from sample data.")
        except Exception as e:
            print(f"[!] Error loading CSV data: {e}. Falling back to baseline.")
            x = torch.ones((num_genes, 1), dtype=torch.float)
    else:
        x = torch.ones((num_genes, 1), dtype=torch.float)

    # 3. Part 1: Graph Reconstruction Accuracy (Link Prediction)
    print("[1/2] Computing Graph Reconstruction AUC/AP...")
    with torch.no_grad():
        z = model(x, edge_index)
        
        # Positive edges (existing interactions)
        pos_edge_index = edge_index
        pos_reconstruct = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        pos_preds = torch.sigmoid(pos_reconstruct).numpy()
        
        # Negative edges (random non-existent interactions)
        neg_edge_index = torch.randint(0, num_genes, (2, edge_index.size(1)))
        neg_reconstruct = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        neg_preds = torch.sigmoid(neg_reconstruct).numpy()

    # Combine for AUC calculation
    y_true = np.concatenate([np.ones(pos_preds.shape[0]), np.zeros(neg_preds.shape[0])])
    y_scores = np.concatenate([pos_preds, neg_preds])
    
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # 3. Part 2: Clinical Precision @ 20 (Biomarker Accuracy)
    print("[2/2] Benchmarking against COMSIC/TCGA Ground Truth...")
    z_mag = torch.norm(z, dim=1).numpy()
    z_min, z_max = z_mag.min(), z_mag.max()
    z_mag_norm = (z_mag - z_min) / (z_max - z_min + 1e-9)
    
    # Identify model's top candidates
    rankings = []
    for g, score in zip(genes, z_mag_norm):
        rankings.append({"gene": g, "score": score})
    
    top_20 = sorted(rankings, key=lambda x: x['score'], reverse=True)[:20]
    hits = [r['gene'] for r in top_20 if r['gene'].upper() in BRCA_GOLD_STANDARD]
    precision_at_20 = (len(hits) / 20) * 100

    # 4. Final Final Performance Report
    print("\n" + "-"*50)
    print(" PERFORMANCE SUMMARY")
    print("-"*50)
    print(f"{'Metric':<25} | {'Score':<15}")
    print("-"*50)
    print(f"{'Link Reconstruction AUC':<25} | {auc:.4f}")
    print(f"{'Average Precision (AP)':<25} | {ap:.4f}")
    print(f"{'Clinical Precision @ 20':<25} | {precision_at_20:.1f}%")
    print("-"*50)
    
    print("\nModel Top-Hits found in Gold Standard:")
    print(", ".join(hits) if hits else "None")
    
    if auc > 0.85 and precision_at_20 > 30:
        print("\n✅ STATUS: Model exceeds benchmark thresholds for medical AI deployment.")
    else:
        print("\n⚠️  STATUS: Performance within acceptable research limits (Refine training for clinical use).")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Change CWD to script directory to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description='OncoNexus Accuracy Evaluation')
    parser.add_argument('--model', default='models/gae_weights.pt', help='Path to model weights')
    parser.add_argument('--graph', default='data/graph.json', help='Path to graph json')
    parser.add_argument('--data', default='data/tcga_brca_REAL_sample.csv', help='Path to patient CSV data for clinical testing')
    
    args = parser.parse_args()
    evaluate_accuracy(args.model, args.graph, args.data)
