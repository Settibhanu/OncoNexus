# OncoTrace — Integration Guide

## Project structure
```
oncontrace/
├── backend/
│   ├── app.py                ← Flask API (edit to connect your model)
│   ├── pipeline.py           ← YOUR inference logic goes here
│   ├── requirements.txt
│   ├── data/
│   │   ├── graph_nodes.json  ← from your backend
│   │   ├── graph_edges.json  ← from your backend
│   │   ├── perturbation_scores.json ← from your backend
│   │   └── biomarkers_ranked.csv   ← from your backend
│   └── model/
│       └── model.pt          ← your trained GNN weights
└── frontend/
    ├── index.html            ← upload + home page
    ├── network.html          ← cytoscape gene network
    ├── perturbation.html     ← plotly heatmap
    └── biomarkers.html       ← ranked biomarker table
```

---

## Step 1 — Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

---

## Step 2 — Drop in your backend data files

Copy these files from your completed backend into `backend/data/`:

### graph_nodes.json
```json
[
  {"id": "BRCA1", "label": "BRCA1", "score": 0.91, "known": true},
  {"id": "TP53",  "label": "TP53",  "score": 0.87, "known": true}
]
```
- `id` + `label`: gene name
- `score`: perturbation score (0.0 – 1.0) from your GNN
- `known`: true if gene is a known breast cancer gene

### graph_edges.json
```json
[
  {"source": "BRCA1", "target": "TP53", "weight": 0.9}
]
```
- `source` / `target`: gene IDs matching nodes
- `weight`: interaction confidence from STRING (0.0 – 1.0)

### perturbation_scores.json
```json
{
  "genes":   ["BRCA1", "TP53", "ERBB2"],
  "healthy": [0.05,    0.08,   0.06],
  "cancer":  [0.91,    0.87,   0.76]
}
```
- Parallel arrays: genes[i] has healthy[i] and cancer[i] scores

### biomarkers_ranked.csv
```csv
rank,gene,score,known,pathway
1,BRCA1,0.912,True,DNA repair
2,TP53,0.874,True,Apoptosis
3,ERBB2,0.761,True,Cell proliferation
```

---

## Step 3 — Connect your model (pipeline.py)

Edit `backend/app.py` around line 60:

```python
# REPLACE this block in the /api/predict route:
from pipeline import run_inference
result = run_inference(df, model)
return jsonify(result)
```

Your `pipeline.py` should look like:
```python
import torch
import numpy as np

def run_inference(df, model):
    """
    df: pandas DataFrame of gene expression values (genes as columns)
    model: your loaded PyTorch GNN model
    Returns: dict with risk_score, perturbation_scores, top_biomarkers
    """
    # 1. Preprocess: align gene IDs, normalize
    # 2. Build graph from df
    # 3. model.eval(); with torch.no_grad(): out = model(graph)
    # 4. Compute deviation scores vs healthy baseline
    # 5. Return structured result:
    genes = df.columns.tolist()
    scores = [...]  # your model output per gene

    return {
        "risk_score": float(np.mean([s for s in scores if s > 0.5])),
        "perturbation_scores": [{"gene": g, "score": s} for g, s in zip(genes, scores)],
        "top_biomarkers": sorted(
            [{"gene": g, "score": s} for g, s in zip(genes, scores)],
            key=lambda x: -x["score"]
        )[:10],
        "sample_count": len(df)
    }
```

Also load your model at startup in app.py:
```python
from pipeline import OncoGNN   # your model class
model = OncoGNN(in_channels=..., hidden=128, out=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
```

---

## Step 4 — Run the app
```bash
cd backend
python app.py
```
Open: http://localhost:5000

The frontend pages are served automatically by Flask.
All pages work with MOCK DATA if your JSON files aren't present yet —
so you can test the UI immediately.

---

## Step 5 — Test each page

| Page | URL | What to check |
|------|-----|---------------|
| Home + upload | / | Upload a .csv, see risk score appear |
| Network | /network.html | Nodes load, colors match scores, click works |
| Perturbation | /perturbation.html | Heatmap loads, toggle healthy/cancer works |
| Biomarkers | /biomarkers.html | Table loads, search/filter/export works |

---

## API endpoints reference

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | /api/status | model loaded status |
| GET | /api/graph | nodes + edges JSON |
| GET | /api/perturbation | healthy + cancer scores |
| GET | /api/biomarkers | ranked biomarker list |
| POST | /api/predict | risk score + results from uploaded file |

---

## Demo day checklist
- [ ] All 4 JSON/CSV data files in backend/data/
- [ ] model.pt in backend/model/
- [ ] pipeline.py connected in app.py
- [ ] python app.py runs without errors
- [ ] Upload a test .csv on index.html — risk score appears
- [ ] Network viewer loads real nodes
- [ ] Heatmap shows real healthy vs cancer difference
- [ ] Biomarker table shows your ranked genes
- [ ] Export CSV works
- [ ] Test on the demo laptop specifically — not just your dev machine
