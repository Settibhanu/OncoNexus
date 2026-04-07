from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import os
import torch
import numpy as np
from pipeline import OncoGNN, run_inference

app = Flask(__name__, static_folder='../frontend')
CORS(app)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, 'data')
MODEL_PATH = os.path.join(BASE, 'models', 'gae_weights.pt')

# ── Global Cache ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
genes_cache = []
edge_index_cache = None
graph_cache = {}
biomarker_cache = []
perturbation_cache = {}

def load_app_data():
    global model, genes_cache, edge_index_cache, graph_cache, biomarker_cache, perturbation_cache
    
    # 1. Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = OncoGNN(in_channels=1, hidden_channels=256, out_channels=128)
            state_dict = torch.load(MODEL_PATH, map_location=device)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                # Fallback: Many training scripts only save the DGI module's state dict
                if 'weight' in state_dict and 'encoder.conv1.att' in state_dict:
                    model.dgi.load_state_dict(state_dict)
                    print("[*] Loaded weights via dgi sub-module.")
                else:
                    raise
            model.to(device)
            model.eval()
            print(f"[*] OncoGNN model loaded.")
        except Exception as e:
            print(f"[!] Error loading model: {e}")

    # 2. Cache Graph & Pre-compute Tensors
    graph_path = os.path.join(DATA, 'graph.json')
    if os.path.exists(graph_path):
        with open(graph_path, 'r') as f:
            graph_cache = json.load(f)
            genes_cache = [n['id'] for n in graph_cache['nodes']]
            
            # Pre-compute edge_index
            gene_to_idx = {gene: i for i, gene in enumerate(genes_cache)}
            sources = [gene_to_idx[e['source']] for e in graph_cache['edges'] if e['source'] in gene_to_idx and e['target'] in gene_to_idx]
            targets = [gene_to_idx[e['target']] for e in graph_cache['edges'] if e['source'] in gene_to_idx and e['target'] in gene_to_idx]
            edge_index_cache = torch.tensor([sources, targets], dtype=torch.long).to(device)
            print(f"[*] Graph and Tensors cached (Nodes: {len(genes_cache)}, Edges: {len(sources)})")

    # 3. Cache Perturbation Scores
    perturb_path = os.path.join(DATA, 'perturbation_scores.json')
    if os.path.exists(perturb_path):
        with open(perturb_path, 'r') as f:
            perturbation_cache = json.load(f)
            print(f"[*] Perturbation scores cached.")

    # 4. Cache Biomarkers (Top 100)
    biomarker_path = os.path.join(DATA, 'biomarker_rankings.csv')
    if os.path.exists(biomarker_path):
        df = pd.read_csv(biomarker_path).head(100)
        # Vectorized mapping for performance
        biomarker_cache = df.rename(columns={
            'Rank': 'rank',
            'Gene': 'gene',
            'Importance Score': 'score',
            'Valid Target': 'known'
        }).to_dict(orient='records')
        for item in biomarker_cache:
            item['pathway'] = 'Biological Process'
            item['score'] = round(float(item['score']), 4)
        print(f"[*] Biomarkers cached.")

load_app_data()

# ── Static frontend pages ────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    return send_from_directory('../frontend', filename)

# ── API: health check ────────────────────────────────────────────────────────
@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'data_ready': bool(genes_cache)
    })

@app.route('/api/graph')
def graph():
    if not graph_cache:
        return jsonify({'error': 'Graph data not found'}), 404
        
    # Calculate max importance for normalization across all nodes
    max_imp = 0.0001 # Avoid div by zero
    for gene_id in perturbation_cache:
        max_imp = max(max_imp, perturbation_cache[gene_id].get('importance_score', 0))
        
    nodes = []
    for n in graph_cache.get('nodes', []):
        node_data = n.copy()
        gene_id = n['id']
        if gene_id in perturbation_cache:
            p = perturbation_cache[gene_id]
            raw_imp = p.get('importance_score', 0)
            node_data['importance'] = raw_imp
            node_data['z_score'] = p.get('z_score', 0)
            # Normalize score to 0-1 range for frontend filtering (30% = 0.3)
            node_data['score'] = raw_imp / max_imp
        else:
            node_data['score'] = 0
            
        nodes.append(node_data)
        
    return jsonify({
        'nodes': nodes,
        'edges': graph_cache.get('edges', [])
    })

# ── API: perturbation scores ─────────────────────────────────────────────────
@app.route('/api/perturbation')
def perturbation():
    if not perturbation_cache:
        return jsonify({'error': 'Perturbation data not found'}), 404
        
    genes = list(perturbation_cache.keys())
    importance = [d['importance_score'] for d in perturbation_cache.values()]
    
    # Healthy baseline stays mocked for UX visualization
    healthy = [round(np.random.normal(0.05, 0.02), 3) for _ in genes]
    cancer = importance

    return jsonify({
        'genes': genes,
        'healthy': healthy,
        'cancer': cancer
    })

# ── API: biomarkers ──────────────────────────────────────────────────────────
@app.route('/api/biomarkers')
def biomarkers():
    if not biomarker_cache:
        return jsonify({'error': 'Biomarker data not found'}), 404
    return jsonify(biomarker_cache)

# ── API: predict (live analysis) ─────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'expression_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['expression_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        
        if model is None or edge_index_cache is None:
            return jsonify({'error': 'System not fully initialized'}), 500
        
        # Determine gene count (excluding logic column if any)
        genes_in_file = [c for c in df.columns if c != 'sample_id']
        
        # Run optimized inference
        results = run_inference(df, model, genes_cache, edge_index_cache, device=device)
        
        # Add metadata for the new UI
        results['sample_count'] = len(df)
        results['gene_count'] = len(genes_in_file)
        
        return jsonify(results)

    except Exception as e:
        print(f"[!] Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Hugging Face Spaces and other platforms use the PORT environment variable
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
