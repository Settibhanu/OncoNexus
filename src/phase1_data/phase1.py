import os
import json
import logging
import requests
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import setup_logger

logger = setup_logger("Phase1")

KNOWN_BRCA_GENES = {
    'BRCA1', 'BRCA2', 'TP53', 'PIK3CA', 'ESR1', 'ERBB2', 'MYC', 
    'PTEN', 'CCND1', 'CDH1', 'RB1', 'PALB2', 'ATM', 'CHEK2', 'AKT1', 'MAP3K1', 'PIK3R1'
}

def download_tcga_brca_api(download_dir):
    logger.info("Querying GDC API for True TCGA-BRCA RNA-Seq Count Data...")
    
    files_endpt = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content":[
            {"op": "in", "content":{"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content":{"field": "data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content":{"field": "data_type", "value": ["Gene Expression Quantification"]}}
        ]
    }
    
    # Pulled size up to 150 for massive real biological variance
    params = {"filters": json.dumps(filters), "fields": "file_id,file_name", "format": "JSON", "size": "150"}
    
    response = requests.get(files_endpt, params=params)
    response.raise_for_status()
    file_hits = response.json()["data"]["hits"]
    file_ids = [f["file_id"] for f in file_hits]
    
    logger.info(f"Downloading {len(file_ids)} true TCGA-BRCA sequence files...")
    data_endpt = "https://api.gdc.cancer.gov/data"
    headers = {"Content-Type": "application/json"}
    
    dl_response = requests.post(data_endpt, data=json.dumps({"ids": file_ids}), headers=headers)
    dl_response.raise_for_status()
    
    tar_path = os.path.join(download_dir, "tcga_brca.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(dl_response.content)
        
    logger.info("Extracting TCGA real sample expressions...")
    expressions = []
    
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".tsv"):
                f = tar.extractfile(member)
                df = pd.read_csv(f, sep="\t", skiprows=1)
                df = df.dropna(subset=['gene_name']).drop_duplicates(subset=['gene_name']).set_index('gene_name')
                expressions.append(df.iloc[:, 2]) # Unstranded tpm/counts
                    
    tcga_matrix = pd.concat(expressions, axis=1).T
    tcga_matrix.index = [f"TCGA_sample_{i}" for i in range(tcga_matrix.shape[0])]
    
    return tcga_matrix.fillna(0)

def download_gtex_api(download_dir):
    logger.info("Executing precise Pandas slice on GTEx 1.5GB Dataframe...")
    
    # 1. Fetch metadata mapping to find only Breast columns securely
    meta_url = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    logger.info("Fetching Sample Metadata...")
    meta_df = pd.read_csv(meta_url, sep='\t')
    
    # Find SAMPIDs corresponding to real breast tissue
    breast_samples = meta_df[meta_df['SMTSD'] == 'Breast - Mammary Tissue']['SAMPID'].tolist()
    
    if len(breast_samples) > 200:
        breast_samples = breast_samples[:200]
        
    logger.info(f"Isolated {len(breast_samples)} legitimate GTEx Breast patients. Slicing matrix natively...")
    
    # 2. Extract strictly required columns from the 1.5GB GCT to avoid local System OOM crashes
    tpm_url = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    
    # pandas skips loading columns not in `usecols`, natively saving RAM
    # Since GCT format lists available columns, we must ensure they match exactly or simply read safely.
    # We will read only the first two cols for names, and map manually to ensure robust network safety
    header = pd.read_csv(tpm_url, sep='\t', skiprows=2, nrows=0).columns.tolist()
    valid_breast_samples = [s for s in breast_samples if s in header]
    
    target_cols = ['Description'] + valid_breast_samples
    
    df_gtex = pd.read_csv(tpm_url, sep='\t', skiprows=2, usecols=target_cols)
    df_gtex = df_gtex.drop_duplicates(subset=['Description']).set_index('Description')
    df_gtex.index.name = 'gene_name'
    
    gtex_matrix = df_gtex.T
    
    return gtex_matrix.fillna(0)

def normalize_expression(df: pd.DataFrame) -> pd.DataFrame:
    df_log = np.log1p(df)
    df_norm = df_log.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x, axis=1)
    return df_norm.fillna(0)

def run_phase1(data_dir: str = "data"):
    logger.info("Starting Phase 1: Real-World Patient Data Acquisition")
    
    tcga_df = download_tcga_brca_api(data_dir)
    gtex_df = download_gtex_api(data_dir)

    logger.info("Aligning gene IDs...")
    common_genes = list(set(gtex_df.columns).intersection(set(tcga_df.columns)))
    
    logger.info("Filtering to Top 2000 highest biological variance genes (Variance Funnel)...")
    tcga_aligned = tcga_df[common_genes]
    
    variances = tcga_aligned.var(axis=0)
    top_variant_genes = variances.nlargest(1950).index.tolist()
    target_genes = [g for g in KNOWN_BRCA_GENES if g in common_genes]
    
    final_genes = list(set(top_variant_genes + target_genes))
    
    gtex_aligned = gtex_df[final_genes]
    tcga_aligned = tcga_df[final_genes]

    logger.info("Applying log1p + z-score normalization...")
    gtex_norm = normalize_expression(gtex_aligned)
    tcga_norm = normalize_expression(tcga_aligned)

    logger.info("Splitting GTEx dataset into Train (80%) and Validation (20%)...")
    gtex_train, gtex_val = train_test_split(gtex_norm, test_size=0.2, random_state=42)

    logger.info("Exporting preliminary matrices to disk...")
    # NOTE: These matrices have 1959+ genes. They must be subset later in Phase 3
    # based on the 1635 nodes actually approved by STRING PPI!
    gtex_train.to_csv(os.path.join(data_dir, "gtex_train.csv"))
    gtex_val.to_csv(os.path.join(data_dir, "gtex_val.csv"))
    tcga_norm.to_csv(os.path.join(data_dir, "tcga_brca.csv"))
    
    with open(os.path.join(data_dir, "pre_ppi_gene_list.json"), "w") as f:
        json.dump(final_genes, f)
        
    logger.info("Phase 1 complete!")
