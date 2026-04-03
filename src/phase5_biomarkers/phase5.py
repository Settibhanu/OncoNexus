import os
import json
import pandas as pd
from src.utils import setup_logger

logger = setup_logger("Phase5")

# Extended COSMIC / DisGeNET validated breast cancer targets
KNOWN_BIOMARKERS = {
    'BRCA1', 'BRCA2', 'TP53', 'PIK3CA', 'PTEN', 'AKT1', 'ERBB2', 'ESR1', 
    'GATA3', 'CDH1', 'MAP3K1', 'NCOR1', 'PALB2', 'ATM', 'CHEK2', 'MYC', 
    'CCND1', 'RB1', 'PIK3R1', 'ARID1A', 'KMT2C', 'CBFB', 'RUNX1', 'MUC16',
    'FOXA1', 'PGR', 'FGFR1'
}

def run_phase5(data_dir: str = "data"):
    logger.info("Starting Phase 5: Biomarker Validation")
    
    scores_path = os.path.join(data_dir, "perturbation_scores.json")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} is missing. Run Phase 4.")
        
    with open(scores_path, "r") as f:
        scores_dict = json.load(f)
        
    # Sort genes descending exactly by explicit XGBoost/RF Supervised Classifications
    sorted_genes = sorted(scores_dict.keys(), key=lambda g: scores_dict[g]["importance_score"], reverse=True)
    
    # Calculate Recovery Metrics
    top_n = 50
    top_genes = sorted_genes[:top_n]
    
    logger.info(f"Top {top_n} predicted biomarkers generated.")
    
    recovered = [g for g in top_genes if g in KNOWN_BIOMARKERS]
    
    precision_at_n = len(recovered) / top_n
    recall_at_n = len(recovered) / len(KNOWN_BIOMARKERS)
    
    logger.info("Evaluation metrics successfully computed:")
    logger.info(f"- Precision@{top_n}: {precision_at_n:.3f}")
    logger.info(f"- Recall@{top_n}:    {recall_at_n:.3f}")
    
    if recovered:
        logger.info(f"- Strongly Validated Targets successfully identified: {', '.join(recovered)}")
    else:
        logger.warning(f"- ZERO known targets localized. Please check Dataset limits or DGI Convergence.")
        
    out_df = pd.DataFrame([
        {"Rank": i+1, "Gene": g, "Importance Score": scores_dict[g]["importance_score"], "Valid Target": (g in KNOWN_BIOMARKERS)}
        for i, g in enumerate(sorted_genes)
    ])
    
    out_df.to_csv(os.path.join(data_dir, "biomarker_rankings.csv"), index=False)
    logger.info("Phase 5 complete! V2 Pipeline execution finished successfully.")
