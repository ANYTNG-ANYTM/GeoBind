# GeoBind: Geometric Shape Profiling and Binding Prediction for Drug-Receptor Molecules

A computational pipeline for predicting protein-ligand binding affinity using geometric shape profiling and machine learning.

## Project Structure

```
GeoBind/
├── phase1_data_ingestion.py    # Data ingestion & chemoinformatics (Phase 1)
├── phase2_physics_geometry.py   # Physics & geometry engine (Phase 2)
├── test_phase1.py              # Test suite for Phase 1
├── test_phase2.py              # Test suite for Phase 2
├── requirements.txt             # Python dependencies
└── README.md
```

## Phase 1: Data Ingestion & Chemoinformatics Setup

### Overview
Phase 1 handles:
- **PDB Parsing**: Extract atomic coordinates from receptor protein structures
- **Ligand Parsing**: Parse ligand structures from SDF/MOL2 files
- **Sanitization**: Remove water molecules, isolate target chains
- **Data Extraction**: 3D coordinates, atom types, residue information

---

## Phase 2: Physics & Geometry Engine

### Overview
Phase 2 computes the **8-feature Complementarity Vector** that characterizes protein-ligand binding:

$$V = [dist\_score, angle\_score, electrostatic\_energy, hbond\_count, hydrophobic\_score, vdW\_score, shape\_match, pocket\_fit]$$

**Feature Definitions:**

| Feature | Calculation | Range | Favorable |
|---------|------------|-------|-----------|
| **dist_score** | Inverse of center-of-mass distance | 0-1 | Higher |
| **angle_score** | Principal moment of inertia alignment | 0-1 | Higher |
| **electrostatic_energy** | Coulombic interactions (ε=4.0) | ℝ kcal/mol | Lower (negative) |
| **hbond_count** | Distance ≤3.5 Å + angle ≥120° | Integer | Higher |
| **hydrophobic_score** | Lipophilic contact surface area | 0-1 | Higher |
| **vdW_score** | Lennard-Jones 12-6 potential | ℝ kcal/mol | Lower (negative) |
| **shape_match** | Volume overlap (Jaccard index) | 0-1 | Higher |
| **pocket_fit** | Optimal volume ratio to receptor | 0-1 | Higher |

### Key Components (phase2_physics_geometry.py)

**Scoring Classes:**
- **`DistanceAngleScorer`**: Distance and alignment calculations
- **`ElectrostaticCalculator`**: Coulombic interaction energy
- **`HydrogenBondCounter`**: H-bond detection with geometry constraints
- **`HydrophobicScorer`**: Lipophilic contact calculations
- **`VDWCalculator`**: Lennard-Jones potential energy
- **`ShapeComplementarityCalculator`**: Volume overlap and pocket fitting
- **`ComplementarityVectorGenerator`**: Unified vector generation interface

### Usage

```python
from phase1_data_ingestion import DataIngestionPipeline
from phase2_physics_geometry import ComplementarityVectorGenerator

# Load structures
pipeline = DataIngestionPipeline()
pipeline.load_receptor('protein.pdb')
pipeline.load_ligand('ligand.sdf')

# Generate Complementarity Vector
generator = ComplementarityVectorGenerator()
vector, metadata = generator.calculate_complementarity_vector(
    pipeline.receptor,
    pipeline.ligand
)

# Access features
print(f"Distance score: {vector[0]:.4f}")
print(f"VDW energy: {vector[5]:.4f} kcal/mol")
print(f"H-bonds: {int(vector[3])}")

# Human-readable output
print(generator.describe_vector(vector))
```

### Physics Parameters

**Van der Waals Radii** (Bondi set):
- C: 1.70 Å, N: 1.55 Å, O: 1.52 Å, S: 1.80 Å

**Lennard-Jones Parameters** (CHARMM FF):
- σ (optimal distance) and ε (well depth) per atom type
- Combined using geometric mean (σ) and arithmetic mean (ε)

**Electrostatic**:
- Dielectric constant: 4.0 (protein environment)
- Coulombic constant: 332.06 kcal·Å/(mol·e²)

**Hydrogen Bonding**:
- Max donor-acceptor distance: 3.5 Å
- Min D-H...A angle: 120°

### Testing Phase 2

Run the comprehensive test suite:

```bash
python test_phase2.py
```

**Tests:**
1. Distance & Angle Scores
2. Electrostatic Energy
3. Hydrogen Bond Detection
4. Hydrophobic Scoring
5. Van der Waals Scoring
6. Shape Matching & Pocket Fitting
7. Full Vector Generation
8. Vector Value Validation

All tests verify numerical correctness, value ranges, and physical reasonableness.

---

## Phase 1: Data Ingestion & Chemoinformatics Setup

### Overview

#### `phase1_data_ingestion.py`

**Classes:**
- **`AtomicCoordinates`**: Container for atomic coordinates and metadata
  - `atom_names`: List of atom identifiers
  - `coordinates`: NumPy array (N×3) of x, y, z positions
  - `residues`: Residue information
  - `elements`: Element symbols

- **`ReceptorParser`**: Parse PDB files
  - `parse_pdb()`: Extract receptor coordinates from PDB file
  - `sanitize_receptor()`: Clean receptor data

- **`LigandParser`**: Parse ligand files (SDF/MOL2)
  - `parse_sdf()`: Extract ligand from SDF format
  - `parse_mol2()`: Extract ligand from MOL2 format

- **`DataIngestionPipeline`**: Unified interface
  - `load_receptor()`: Load and parse PDB
  - `load_ligand()`: Load and parse ligand
  - `get_summary()`: Get structure summaries

### Installation

1. **Create a Python virtual environment** (recommended):
```bash
cd c:\Yash\BT_305\Project\GeoBind
python -m venv venv
venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This installs:
- `numpy`: Numerical computing
- `scipy`: Scientific computing
- `Biopython`: PDB parsing and biological sequence analysis
- `rdkit`: Chemoinformatics toolkit for ligand processing
- `scikit-learn`: Machine learning preprocessing (needed for future phases)
- `xgboost`: ML model (needed for Phase 3)
- `fastapi` & `uvicorn`: Web API (needed for Phase 4)

### Usage

#### Basic Usage:

```python
from phase1_data_ingestion import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline()

# Load receptor (PDB file)
receptor_meta = pipeline.load_receptor('path/to/protein.pdb', chain_id='A')

# Load ligand (SDF or MOL2)
ligand_meta = pipeline.load_ligand('path/to/ligand.sdf')

# Get coordinates
receptor_coords = pipeline.receptor  # AtomicCoordinates object
ligand_coords = pipeline.ligand

# Access data
print(f"Receptor atoms: {len(receptor_coords)}")
print(f"Ligand atoms: {len(ligand_coords)}")
print(f"Receptor center: {receptor_meta['center_of_mass']}")
print(f"Ligand MW: {ligand_meta['molecular_weight']} g/mol")
```

#### Manual Parsing:

```python
from phase1_data_ingestion import ReceptorParser, LigandParser

# Parse PDB
receptor_coords, metadata = ReceptorParser.parse_pdb('protein.pdb')

# Parse SDF
ligand_coords, metadata = LigandParser.parse_sdf('ligand.sdf')

# Access coordinates (NumPy arrays)
xyz = receptor_coords.coordinates  # Shape: (N, 3)
```

### Testing Phase 1

Run the test suite:

```bash
python test_phase1.py
```

**What the tests verify:**
1. **Test 1 - Receptor Parsing**: PDB file loading and coordinate extraction
2. **Test 2 - Ligand Parsing**: SDF file loading with RDKit
3. **Test 3 - Full Pipeline**: Integrated receptor + ligand loading
4. **Test 4 - Coordinate Operations**: Numerical stability and basic calculations

The test script automatically creates minimal test structures:
- `test_protein.pdb`: 3-residue alpha helix
- `test_ligand.sdf`: Benzene molecule

**Expected Output:**
```
============================================================
PHASE 1: DATA INGESTION TEST SUITE
============================================================

============================================================
TEST 1: Receptor (PDB) Parsing
============================================================

✓ Successfully parsed PDB file
  - Atoms extracted: 15
  - Coordinate shape: (15, 3)
  ...

✓ ALL TESTS PASSED!
```

### Working with Real PDB/Ligand Files

#### Option 1: Download from RCSB PDB (recommended)

1. Go to https://www.rcsb.org
2. Search for a protein-ligand complex (e.g., "2D3D" - Acetylcholinesterase with bound inhibitor)
3. Download:
   - PDB Structure: `2D3D.pdb` (receptor)
   - Ligand SDF: `2D3D_ligand.sdf` (ligand)

Example:
```bash
# Download PDB
curl -o protein.pdb https://files.rcsb.org/download/2D3D.pdb

# Download Ligand (requires parsing the RCSB API for specific format)
```

#### Option 2: Use Python to fetch and save:

```python
from urllib import request
from phase1_data_ingestion import DataIngestionPipeline

# Download PDB
pdb_url = "https://files.rcsb.org/download/2D3D.pdb"
request.urlretrieve(pdb_url, "2D3D.pdb")

# Now use the pipeline
pipeline = DataIngestionPipeline()
pipeline.load_receptor("2D3D.pdb")
```

#### Option 3: Create synthetic test data:

The test script includes utilities to create minimal structures for development:
```python
from test_phase1 import create_minimal_test_pdb, create_minimal_test_ligand

create_minimal_test_pdb("my_protein.pdb")
create_minimal_test_ligand("my_ligand.sdf")
```

### Key Features

✓ **Robust PDB Parsing**: Uses BioPython with permissive error handling  
✓ **Water Removal**: Automatically filters out water molecules (HOH, WAT)  
✓ **Chain Isolation**: Specify target chain or defaults to first chain  
✓ **RDKit Integration**: 3D coordinate generation if missing  
✓ **Metadata Extraction**: Bounding boxes, centers of mass, molecular properties  
✓ **Type Safety**: NumPy arrays with explicit float32 precision  
✓ **Comprehensive Logging**: Track parsing steps with detailed log messages  
✓ **Error Handling**: Clear exceptions for invalid files or corrupt data  

### Important Notes

- **Coordinate System**: All coordinates are in Ångströms (Å)
- **Precision**: Float32 arrays for GPU compatibility in later phases
- **Chain Selection**: For multi-chain proteins, explicitly specify `chain_id` parameter
- **Ligand Format**: SDF preferred; MOL2 also supported
- **3D Coordinates**: Required for all calculations; auto-generation if missing

### Next Phase

**Phase 2: Physics & Geometry Engine** will use these parsed structures to calculate the 8-feature Complementarity Vector:
$$V = [dist\_score, angle\_score, electrostatic\_energy, hbond\_count, hydrophobic\_score, vdW\_score, shape\_match, pocket\_fit]$$

---

**GeoBind v1.0** | Computational Biology + Drug Design
