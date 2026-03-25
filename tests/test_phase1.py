"""
Test Script for Phase 1: Data Ingestion & Chemoinformatics Setup

This script tests the data ingestion pipeline with sample PDB and ligand files.

Instructions for obtaining test files:
    1. Download a PDB file from RCSB PDB (https://www.rcsb.org)
       Example: 2D3D (a small protein-ligand complex)
       URL: https://files.rcsb.org/download/2D3D.pdb
    
    2. Download the ligand SDF from RCSB:
       URL: https://files.rcsb.org/download/2D3D_ligand.sdf (or use RDKit to generate)
    
    Or use the included test functions to create minimal test structures.
"""

import os
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.phase1_data_ingestion import (
    DataIngestionPipeline,
    ReceptorParser,
    LigandParser,
    AtomicCoordinates
)


def create_minimal_test_pdb(output_path: str):
    """
    Create a minimal PDB file for testing (small alpha-helix).
    
    Args:
        output_path: Path to save the PDB file
    """
    pdb_content = """HEADER    TEST STRUCTURE
TITLE     MINIMAL TEST PROTEIN
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   2.300   0.650  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000  -0.800  -1.200  1.00  0.00           C
ATOM      6  N   GLY A   2       3.100   1.600  -0.700  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.750   2.900  -0.750  1.00  0.00           C
ATOM      8  C   GLY A   2       5.250   2.800  -0.500  1.00  0.00           C
ATOM      9  O   GLY A   2       5.900   1.800  -0.850  1.00  0.00           O
ATOM     10  N   SER A   3       5.750   3.800   0.200  1.00  0.00           N
ATOM     11  CA  SER A   3       7.200   3.800   0.400  1.00  0.00           C
ATOM     12  C   SER A   3       7.700   5.200   0.700  1.00  0.00           C
ATOM     13  O   SER A   3       7.100   6.100   0.100  1.00  0.00           O
ATOM     14  CB  SER A   3       7.850   3.200   1.600  1.00  0.00           C
ATOM     15  OG  SER A   3       7.400   1.900   1.900  1.00  0.00           O
END
"""
    with open(output_path, 'w') as f:
        f.write(pdb_content)
    print(f"✓ Created minimal test PDB: {output_path}")


def create_minimal_test_ligand(output_path: str):
    """
    Create a minimal ligand (benzene) for testing.
    
    Args:
        output_path: Path to save the SDF file
    """
    mol = Chem.MolFromSmiles('c1ccccc1')  # Benzene
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()
    print(f"✓ Created minimal test ligand (benzene): {output_path}")


def test_receptor_parsing():
    """Test PDB file parsing."""
    print("\n" + "="*60)
    print("TEST 1: Receptor (PDB) Parsing")
    print("="*60)
    
    test_pdb = Path(__file__).parent / "test_protein.pdb"
    
    if not test_pdb.exists():
        create_minimal_test_pdb(str(test_pdb))
    
    try:
        receptor_coords, metadata = ReceptorParser.parse_pdb(str(test_pdb))
        
        print(f"\n✓ Successfully parsed PDB file")
        print(f"  - Atoms extracted: {len(receptor_coords)}")
        print(f"  - Coordinate shape: {receptor_coords.coordinates.shape}")
        print(f"  - Center of mass: {metadata['center_of_mass']}")
        print(f"  - Bounding box min: {metadata['bounding_box_min']}")
        print(f"  - Bounding box max: {metadata['bounding_box_max']}")
        
        # Print first 3 atoms
        print(f"\n  First 3 atoms:")
        for i in range(min(3, len(receptor_coords))):
            print(f"    {i+1}. {receptor_coords.atom_names[i]:5s} "
                  f"({receptor_coords.residues[i]:10s}) "
                  f"Coords: {receptor_coords.coordinates[i]}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error parsing PDB: {e}")
        return False


def test_ligand_parsing():
    """Test ligand (SDF) file parsing."""
    print("\n" + "="*60)
    print("TEST 2: Ligand (SDF) Parsing")
    print("="*60)
    
    test_sdf = Path(__file__).parent / "test_ligand.sdf"
    
    if not test_sdf.exists():
        create_minimal_test_ligand(str(test_sdf))
    
    try:
        ligand_coords, metadata = LigandParser.parse_sdf(str(test_sdf))
        
        print(f"\n✓ Successfully parsed SDF file")
        print(f"  - Atoms extracted: {len(ligand_coords)}")
        print(f"  - Molecular formula: {metadata['molecular_formula']}")
        print(f"  - Molecular weight: {metadata['molecular_weight']:.2f} g/mol")
        print(f"  - Coordinate shape: {ligand_coords.coordinates.shape}")
        print(f"  - Center of mass: {metadata['center_of_mass']}")
        
        # Print all atoms for small molecules
        print(f"\n  Atoms:")
        for i in range(len(ligand_coords)):
            print(f"    {i+1}. {ligand_coords.atom_names[i]:5s} "
                  f"({ligand_coords.elements[i]:2s}) "
                  f"Coords: {ligand_coords.coordinates[i]}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error parsing SDF: {e}")
        return False


def test_pipeline():
    """Test the full data ingestion pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Full Pipeline (Receptor + Ligand)")
    print("="*60)
    
    test_pdb = Path(__file__).parent / "test_protein.pdb"
    test_sdf = Path(__file__).parent / "test_ligand.sdf"
    
    # Create test files if they don't exist
    if not test_pdb.exists():
        create_minimal_test_pdb(str(test_pdb))
    if not test_sdf.exists():
        create_minimal_test_ligand(str(test_sdf))
    
    try:
        pipeline = DataIngestionPipeline()
        
        # Load receptor
        receptor_meta = pipeline.load_receptor(str(test_pdb))
        print(f"\n✓ Loaded receptor")
        print(f"  - File: {receptor_meta['pdb_file']}")
        print(f"  - Chain: {receptor_meta['chain_id']}")
        print(f"  - Atoms: {receptor_meta['n_atoms']}")
        
        # Load ligand
        ligand_meta = pipeline.load_ligand(str(test_sdf))
        print(f"\n✓ Loaded ligand")
        print(f"  - File: {ligand_meta['sdf_file']}")
        print(f"  - Formula: {ligand_meta['molecular_formula']}")
        print(f"  - Atoms: {ligand_meta['n_atoms']}")
        
        # Get summary
        summary = pipeline.get_summary()
        print(f"\n✓ Pipeline Summary:")
        print(f"  - Receptor atoms: {summary['receptor']['n_atoms']}")
        print(f"  - Ligand atoms: {summary['ligand']['n_atoms']}")
        print(f"  - Total atoms: {summary['receptor']['n_atoms'] + summary['ligand']['n_atoms']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_operations():
    """Test coordinate operations and calculations."""
    print("\n" + "="*60)
    print("TEST 4: Coordinate Operations")
    print("="*60)
    
    test_sdf = Path(__file__).parent / "test_ligand.sdf"
    
    if not test_sdf.exists():
        create_minimal_test_ligand(str(test_sdf))
    
    try:
        ligand_coords, _ = LigandParser.parse_sdf(str(test_sdf))
        
        # Calculate basic statistics
        mean_coord = np.mean(ligand_coords.coordinates, axis=0)
        std_coord = np.std(ligand_coords.coordinates, axis=0)
        distances_from_origin = np.linalg.norm(ligand_coords.coordinates, axis=1)
        
        print(f"\n✓ Coordinate statistics:")
        print(f"  - Mean coordinates: {mean_coord}")
        print(f"  - Std coordinates: {std_coord}")
        print(f"  - Min distance from origin: {np.min(distances_from_origin):.3f} Å")
        print(f"  - Max distance from origin: {np.max(distances_from_origin):.3f} Å")
        print(f"  - Mean distance from origin: {np.mean(distances_from_origin):.3f} Å")
        
        # Calculate pairwise distances (first 5 atoms)
        from scipy.spatial.distance import pdist
        n_atoms = min(5, len(ligand_coords))
        pairwise_distances = pdist(ligand_coords.coordinates[:n_atoms], metric='euclidean')
        print(f"\n  - Pairwise distances (first {n_atoms} atoms): {pairwise_distances[:5]}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error in coordinate operations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE 1: DATA INGESTION TEST SUITE")
    print("="*60)
    
    results = {
        'Receptor Parsing': test_receptor_parsing(),
        'Ligand Parsing': test_ligand_parsing(),
        'Full Pipeline': test_pipeline(),
        'Coordinate Operations': test_coordinate_operations(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nPhase 1 is ready for the next phases.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the errors above.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
