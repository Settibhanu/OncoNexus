import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import DeepGraphInfomax
from src.phase3_train.phase3 import GATEncoder, summary, corruption
from src.utils import setup_logger

logger = setup_logger("Phase4")

def run_phase4(data_dir: str = "data", models_dir: str = "models"):
    logger.info("Starting Phase 4: Supervised ML Biomarker Extraction (Random Forest on DGI Tensors)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    weights_path = os.path.join(models_dir, "gae_weights.pt")
    tcga_path = os.path.join(data_dir, "tcga_brca.csv")
    gtex_path = os.path.join(data_dir, "gtex_val.csv")
    graph_path = os.path.join(data_dir, "adjacency.pt")
    nodes_path = os.path.join(data_dir, "final_graph_nodes.json")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError("Missing DGI weights. Run Phase 3.")
        
    encoder = GATEncoder(in_channels=1, hidden_channels=256, out_channels=128)
    model = DeepGraphInfomax(hidden_channels=128, encoder=encoder, summary=summary, corruption=corruption).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    tcga_df = pd.read_csv(tcga_path, index_col=0)
    gtex_df = pd.read_csv(gtex_path, index_col=0)
    edge_index = torch.load(graph_path).to(device)
    
    with open(nodes_path, "r") as f:
        genes = json.load(f)
        
    # CRITICAL FIX 3: Re-align dimensions correctly exactly as done in training
    tcga_df = tcga_df[genes]
    gtex_df = gtex_df[genes]
    
    logger.info(f"Loaded {tcga_df.shape[0]} TCGA cancer patients and {gtex_df.shape[0]} Baseline patients.")
    
    X_dgi = []
    y_labels = []
    
    logger.info("Generating DGI Latent Topological Maps per patient...")
    with torch.no_grad():
        for _, row in tcga_df.iterrows():
            x_i = torch.tensor(row.values, dtype=torch.float).unsqueeze(1).to(device)
            emb = model.encoder(x_i, edge_index).cpu().numpy()
            X_dgi.append(emb.flatten())
            y_labels.append(1) # Cancer class
            
        for _, row in gtex_df.iterrows():
            x_i = torch.tensor(row.values, dtype=torch.float).unsqueeze(1).to(device)
            emb = model.encoder(x_i, edge_index).cpu().numpy()
            X_dgi.append(emb.flatten())
            y_labels.append(0) # Normal class

    X_train = np.array(X_dgi)
    y_train = np.array(y_labels)
    
    # CRITICAL FIX 4: SUPERVISED CLASSIFIER INJECTION
    logger.info(f"Training Supervised Random Forest Classifier on Graph Tensors (Shape: {X_train.shape})...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_train, y_train)
    logger.info(f"Binary Supervised Classifier Training complete. Diagnostic Accuracy: {accuracy*100:.2f}%")
    
    logger.info("Extracting topological Feature Importances dynamically scaled to Gene Nodes...")
    # The XGBoost/RF trees extract importance array of size [1635 * 128]
    raw_importances = clf.feature_importances_
    
    # Reshape specifically into the original PyTorch Node Tensors: [1635 nodes, 128 hidden features]
    reshaped_importances = raw_importances.reshape(len(genes), 128)
    # The sum across the 128 latent dimensions is the absolute importance of the gene natively!
    gene_importances = reshaped_importances.sum(axis=1)
    
    # Generate distribution boundaries purely for flagging
    global_mean = np.mean(gene_importances)
    global_std = np.std(gene_importances)
    
    results = []
    scores_dict = {}
    
    for i, gene in enumerate(genes):
        score = gene_importances[i]
        z_score = (score - global_mean) / global_std if global_std > 0 else 0
        
        # High impact logic
        is_perturbed = bool(z_score > 1.5)
        
        scores_dict[gene] = {
            "importance_score": float(score),
            "z_score": float(z_score),
            "is_perturbed": is_perturbed
        }
        
        results.append({
            "gene_id": gene,
            "importance_score": score,
            "z_score": z_score,
            "is_perturbed": is_perturbed
        })
        
    df_results = pd.DataFrame(results)
    logger.info(f"Supervised Head successfully flagged {df_results['is_perturbed'].sum()} critical biomarkers out of {len(genes)} network nodes.")
    
    df_results.to_csv(os.path.join(data_dir, "perturbed_genes.csv"), index=False)
    with open(os.path.join(data_dir, "perturbation_scores.json"), "w") as f:
        json.dump(scores_dict, f)
        
    logger.info("Phase 4 Complete: Biomarkers actively localized.")
