# GeoBind: Geometric Shape Profiling and Binding Prediction for Drug-Receptor Molecules

> A production-ready computational pipeline for predicting protein-ligand binding affinity using geometric shape profiling and machine learning.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Project Architecture](#project-architecture)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Setup & Installation](#setup--installation)
7. [Running the Project](#running-the-project)
8. [Running Tests](#running-tests)
9. [Phase Reference](#phase-reference)
   - [Phase 1 – Data Ingestion & Chemoinformatics](#phase-1--data-ingestion--chemoinformatics)
   - [Phase 2 – Physics & Geometry Engine](#phase-2--physics--geometry-engine)
   - [Phase 3 – Machine Learning Pipeline](#phase-3--machine-learning-pipeline)
   - [Phase 4 – REST API Service](#phase-4--rest-api-service)
   - [Phase 5 – Web Frontend](#phase-5--web-frontend)
10. [API Reference](#api-reference)
11. [Working with Real Molecular Data](#working-with-real-molecular-data)
12. [License](#license)

---

## Introduction

**GeoBind** is an end-to-end pipeline that predicts whether a small-molecule ligand will bind to a target protein receptor. It combines classical computational chemistry (3-D shape analysis, electrostatics, hydrogen-bond detection, van der Waals potentials) with modern machine learning (XGBoost classifier) and wraps the result in a REST API and interactive Streamlit web interface.

The pipeline covers the full workflow:

1. Parsing raw PDB and SDF/MOL2 structure files
2. Computing an 8-feature *Complementarity Vector* that quantifies shape and physics compatibility
3. Training and evaluating a binary XGBoost classifier
4. Serving predictions through a FastAPI REST endpoint
5. Providing an interactive browser-based UI for exploration and batch predictions

---

## Objectives

- **Predict binding**: Given a receptor PDB file and a ligand SDF file, output a binary binding prediction (binder / non-binder) and a binding probability score.
- **Interpretable features**: Expose the 8-dimensional complementarity vector so researchers can understand *why* a molecule is predicted to bind.
- **Scalable infrastructure**: Support both single-pair and batch API requests with a clean REST interface.
- **Interactive exploration**: Provide a point-and-click Streamlit UI for non-programmers and rapid prototyping.
- **Reproducibility**: Ship all dependencies, model weights, and test fixtures so results can be reproduced on any clean system.

---

## Project Architecture

```
Receptor PDB + Ligand SDF
        │
        ▼
┌──────────────────────┐
│  Phase 1             │  Parse & sanitise molecular structures
│  Data Ingestion      │  (BioPython + RDKit)
└────────┬─────────────┘
         │  AtomicCoordinates objects
         ▼
┌──────────────────────┐
│  Phase 2             │  Compute 8-feature Complementarity Vector
│  Physics & Geometry  │  (distance, angle, electrostatics, H-bonds,
└────────┬─────────────┘   hydrophobics, vdW, shape match, pocket fit)
         │  Feature vector [8 values]
         ▼
┌──────────────────────┐
│  Phase 3             │  Normalise → engineer features → XGBoost
│  ML Pipeline         │  Binary classifier (binder / non-binder)
└────────┬─────────────┘
         │  Trained model + scaler (joblib)
         ▼
┌──────────────────────┐     ┌────────────────────────┐
│  Phase 4             │────▶│  Phase 5               │
│  FastAPI REST API    │     │  Streamlit Web UI       │
└──────────────────────┘     └────────────────────────┘
```

---

## Project Structure

```
GeoBind/
├── src/
│   ├── core/
│   │   ├── phase1_data_ingestion.py     # Data ingestion & chemoinformatics
│   │   └── phase2_physics_geometry.py   # Physics & geometry engine
│   ├── ml/
│   │   ├── phase3_ml_pipeline.py        # ML training pipeline
│   │   ├── phase3_ml_pipeline_optimized.py
│   │   ├── phase3_dataset_builder.py    # Dataset construction utilities
│   │   └── combine_all_training_data.py
│   ├── api/
│   │   ├── phase4_api_service.py        # FastAPI REST service
│   │   └── phase4_client.py             # Python API client
│   └── ui/
│       └── phase5_web_frontend.py       # Streamlit web interface
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase3_dataset_builder.py
│   ├── test_phase4_api.py
│   ├── test_model_loading.py
│   └── test_prediction.py
├── models/                              # Trained model artefacts
├── run_api.py                           # API server entry point
├── run_ui.py                            # Web UI entry point
├── quick_test.py                        # Quick end-to-end smoke test
├── requirements.txt                     # Pinned Python dependencies
├── pyproject.toml                       # Build & tool configuration
└── README.md
```

---

## Prerequisites

| Requirement | Minimum version | Notes |
|-------------|-----------------|-------|
| Python | 3.10 | 3.11 / 3.12 also supported |
| pip | 22+ | Bundled with Python ≥ 3.10 |
| Git | any recent | To clone the repository |

No GPU is required. All calculations run on CPU.

---

## Setup & Installation

Follow these steps on a **clean system** (Linux, macOS, or Windows).

### 1. Clone the repository

```bash
git clone https://github.com/ANYTNG-ANYTM/GeoBind.git
cd GeoBind
```

### 2. Create and activate a virtual environment

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including:

| Package | Purpose |
|---------|---------|
| `numpy` / `scipy` | Numerical computing |
| `pandas` | Data manipulation |
| `Biopython` | PDB parsing |
| `rdkit` | Chemoinformatics & ligand processing |
| `scikit-learn` | Feature normalisation |
| `xgboost` | Binary classification model |
| `fastapi` / `uvicorn` | REST API server |
| `streamlit` / `plotly` | Web frontend |
| `joblib` | Model persistence |

### 4. Verify the installation

```bash
python -c "import numpy, scipy, Bio, rdkit, sklearn, xgboost, fastapi, streamlit; print('All dependencies OK')"
```

---

## Running the Project

### Start the REST API

```bash
python run_api.py
```

The API starts on `http://127.0.0.1:8000` by default.

Options:
```bash
python run_api.py --host 0.0.0.0 --port 9000   # custom host / port
python run_api.py --reload                       # enable auto-reload (development)
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

### Start the Web UI

The web UI requires the API to be running first.

```bash
python run_ui.py
```

Opens the Streamlit interface at `http://localhost:8501`.

Options:
```bash
python run_ui.py --port 9001   # custom port
```

---

### Quick end-to-end smoke test

With the API running in a separate terminal:

```bash
python quick_test.py
```

---

## Running Tests

All tests live in the `tests/` directory and use pytest.

**Run the full test suite:**
```bash
pytest tests/ -v
```

**Run a specific phase:**
```bash
pytest tests/test_phase1.py -v
pytest tests/test_phase2.py -v
pytest tests/test_phase3.py -v
pytest tests/test_phase4_api.py -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Phase Reference

### Phase 1 – Data Ingestion & Chemoinformatics

**Module:** `src/core/phase1_data_ingestion.py`

Handles all molecular file I/O and produces clean `AtomicCoordinates` objects for downstream calculations.

**Key classes:**

| Class | Responsibility |
|-------|---------------|
| `AtomicCoordinates` | Container for 3-D coordinates, atom names, elements, residues |
| `ReceptorParser` | Parse PDB files, sanitise (remove water, isolate chain) |
| `LigandParser` | Parse SDF / MOL2 files via RDKit |
| `DataIngestionPipeline` | Unified interface that orchestrates parsing |

**Basic usage:**

```python
from src.core.phase1_data_ingestion import DataIngestionPipeline

pipeline = DataIngestionPipeline()
receptor_meta = pipeline.load_receptor('protein.pdb', chain_id='A')
ligand_meta   = pipeline.load_ligand('ligand.sdf')

print(f"Receptor atoms : {len(pipeline.receptor)}")
print(f"Ligand MW      : {ligand_meta['molecular_weight']:.2f} g/mol")
```

**Notes:**
- All coordinates are in Ångströms (Å) stored as `float32` NumPy arrays.
- Water molecules (`HOH`, `WAT`) are removed automatically.
- If 3-D coordinates are absent from a ligand file, RDKit generates them via ETKDG.

---

### Phase 2 – Physics & Geometry Engine

**Module:** `src/core/phase2_physics_geometry.py`

Computes the **8-feature Complementarity Vector** that encodes the physical compatibility of a receptor-ligand pair:

$$V = [\text{dist\\_score},\ \text{angle\\_score},\ \text{electrostatic\\_energy},\ \text{hbond\\_count},\ \text{hydrophobic\\_score},\ \text{vdW\\_score},\ \text{shape\\_match},\ \text{pocket\\_fit}]$$

**Feature definitions:**

| Feature | Calculation | Range | Favourable direction |
|---------|-------------|-------|----------------------|
| `dist_score` | Inverse centre-of-mass distance | 0–1 | Higher |
| `angle_score` | Principal moment of inertia alignment | 0–1 | Higher |
| `electrostatic_energy` | Coulombic interactions (ε = 4.0) | ℝ kcal/mol | Lower (negative) |
| `hbond_count` | Distance ≤ 3.5 Å + angle ≥ 120° | Integer | Higher |
| `hydrophobic_score` | Lipophilic contact surface area | 0–1 | Higher |
| `vdW_score` | Lennard-Jones 12-6 potential | ℝ kcal/mol | Lower (negative) |
| `shape_match` | Volume overlap (Jaccard index) | 0–1 | Higher |
| `pocket_fit` | Optimal volume ratio to receptor | 0–1 | Higher |

**Key physics parameters:**
- Van der Waals radii (Bondi set): C = 1.70 Å, N = 1.55 Å, O = 1.52 Å, S = 1.80 Å
- Lennard-Jones σ and ε from CHARMM force field (geometric / arithmetic mixing)
- Dielectric constant: 4.0 (protein interior)
- Coulombic constant: 332.06 kcal·Å / (mol·e²)
- H-bond criterion: donor-acceptor ≤ 3.5 Å and D-H···A ≥ 120°

**Usage:**

```python
from src.core.phase1_data_ingestion import DataIngestionPipeline
from src.core.phase2_physics_geometry import ComplementarityVectorGenerator

pipeline = DataIngestionPipeline()
pipeline.load_receptor('protein.pdb')
pipeline.load_ligand('ligand.sdf')

generator = ComplementarityVectorGenerator()
vector, metadata = generator.calculate_complementarity_vector(
    pipeline.receptor, pipeline.ligand
)

print(f"Distance score : {vector[0]:.4f}")
print(f"VDW energy     : {vector[5]:.4f} kcal/mol")
print(f"H-bonds        : {int(vector[3])}")
print(generator.describe_vector(vector))
```

---

### Phase 3 – Machine Learning Pipeline

**Module:** `src/ml/phase3_ml_pipeline.py`

Trains and evaluates a binary XGBoost classifier on pre-computed complementarity vectors.

**Workflow:**
1. Load CSV dataset (8 raw features + `label` column)
2. Z-score normalisation via `StandardScaler`
3. Engineer 14 derived features (ratios, interactions, cosine similarity)
4. 80 / 20 train-validation split (stratified)
5. Fit `XGBClassifier`
6. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
7. Persist model and scaler with `joblib`

**Run training:**
```bash
python -m src.ml.phase3_ml_pipeline --data path/to/dataset.csv --output models/
```

---

### Phase 4 – REST API Service

**Module:** `src/api/phase4_api_service.py`  
**Entry point:** `run_api.py`

A FastAPI application that loads the trained model and exposes HTTP endpoints for single and batch predictions.

**Start the server:**
```bash
python run_api.py
```

---

### Phase 5 – Web Frontend

**Module:** `src/ui/phase5_web_frontend.py`  
**Entry point:** `run_ui.py`

A Streamlit application that communicates with the Phase 4 API and provides:
- File upload for PDB and SDF files
- Single-pair prediction with probability gauge
- Batch prediction with downloadable results table
- Feature importance and complementarity vector visualisation

**Start the UI (requires API to be running):**
```bash
python run_ui.py
```

---

## API Reference

Base URL: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

### `POST /predict`

Predict binding for a single receptor-ligand pair.

**Request body:**
```json
{
  "receptor_pdb_path": "path/to/protein.pdb",
  "ligand_sdf_path":   "path/to/ligand.sdf",
  "receptor_chain":    "A",
  "threshold":         0.5
}
```

**Response:**
```json
{
  "pair_id":                   "protein_ligand",
  "binding_probability":       0.87,
  "binding_prediction":        1,
  "confidence":                0.74,
  "complementarity_vector":    [0.82, 0.71, -3.4, 2, 0.65, -1.2, 0.78, 0.91],
  "engineered_features":       [...],
  "predicted_at":              "2026-04-17T08:00:00"
}
```

`binding_prediction`: `1` = binder, `0` = non-binder.

### `POST /predict/batch`

Submit multiple pairs in a single request. Accepts a list of receptor/ligand path pairs and returns a list of `PredictionResponse` objects.

### `GET /health`

Returns `{"status": "ok"}` when the service is running and the model is loaded.

---

## Working with Real Molecular Data

### Download from RCSB PDB

```bash
# Download a protein-ligand complex (e.g. 2D3D – Acetylcholinesterase)
curl -o 2D3D.pdb https://files.rcsb.org/download/2D3D.pdb
```

Or from Python:
```python
from urllib import request

request.urlretrieve("https://files.rcsb.org/download/2D3D.pdb", "2D3D.pdb")
```

Browse complexes at <https://www.rcsb.org>.

### Create synthetic test data

The test helpers can generate minimal PDB / SDF files for development:

```python
from tests.test_phase1 import create_minimal_test_pdb, create_minimal_test_ligand

create_minimal_test_pdb("test_protein.pdb")   # 3-residue alpha helix
create_minimal_test_ligand("test_ligand.sdf") # Benzene
```

---

## Team Members:
- Yash Pratap Singh (https://github.com/ANYTNG-ANYTM)
- Devansh Singh (https://github.com/Devansh-04)
- Sahaj Kumar (https://github.com/k-sahaj)
- Harshvardhan Patil (https://github.com/Torpid-Quark) 

---

## License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

*GeoBind v1.0 · Computational Biology + Drug Design*
