"""
Test Script for Phase 2: Physics & Geometry Engine

Tests all 8-feature Complementarity Vector calculations.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.phase1_data_ingestion import DataIngestionPipeline
from src.core.phase2_physics_geometry import ComplementarityVectorGenerator
from test_phase1 import create_minimal_test_pdb, create_minimal_test_ligand


def test_distance_and_angle_scores():
    """Test distance and angle score calculations."""
    print("\n" + "="*60)
    print("TEST 1: Distance and Angle Scores")
    print("="*60)
    
    try:
        # Create test files
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        if not test_pdb.exists():
            create_minimal_test_pdb(str(test_pdb))
        if not test_sdf.exists():
            create_minimal_test_ligand(str(test_sdf))
        
        # Load structures
        pipeline = DataIngestionPipeline()
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        # Calculate scores
        from phase2_physics_geometry import DistanceAngleScorer
        scorer = DistanceAngleScorer()
        
        dist_score = scorer.calculate_dist_score(
            pipeline.receptor.coordinates,
            pipeline.ligand.coordinates
        )
        angle_score = scorer.calculate_angle_score(
            pipeline.receptor.coordinates,
            pipeline.ligand.coordinates
        )
        
        print(f"\n✓ Distance and Angle Scores calculated")
        print(f"  - Distance Score: {dist_score:.4f}")
        print(f"  - Angle Score: {angle_score:.4f}")
        print(f"  - Both scores in [0, 1]: {0 <= dist_score <= 1 and 0 <= angle_score <= 1}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_electrostatic_energy():
    """Test electrostatic energy calculation."""
    print("\n" + "="*60)
    print("TEST 2: Electrostatic Energy Calculation")
    print("="*60)
    
    try:
        # Load test structures
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        # Calculate electrostatic energy
        from phase2_physics_geometry import ElectrostaticCalculator
        calc = ElectrostaticCalculator()
        
        energy = calc.calculate_electrostatic_energy(
            pipeline.receptor.coordinates,
            pipeline.receptor.elements,
            pipeline.ligand.coordinates,
            pipeline.ligand.elements
        )
        
        print(f"\n✓ Electrostatic Energy calculated")
        print(f"  - Energy: {energy:.4f} kcal/mol")
        print(f"  - Reasonable range: {-1000 < energy < 1000}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hydrogen_bonds():
    """Test hydrogen bond detection."""
    print("\n" + "="*60)
    print("TEST 3: Hydrogen Bond Detection")
    print("="*60)
    
    try:
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        from phase2_physics_geometry import HydrogenBondCounter
        hbond = HydrogenBondCounter()
        
        count = hbond.find_hbonds(
            pipeline.receptor.coordinates,
            pipeline.receptor.elements,
            pipeline.ligand.coordinates,
            pipeline.ligand.elements
        )
        
        print(f"\n✓ Hydrogen Bonds detected")
        print(f"  - H-bond count: {count}")
        print(f"  - Count is non-negative integer: {isinstance(count, int) and count >= 0}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hydrophobic_score():
    """Test hydrophobic scoring."""
    print("\n" + "="*60)
    print("TEST 4: Hydrophobic Score Calculation")
    print("="*60)
    
    try:
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        from phase2_physics_geometry import HydrophobicScorer
        scorer = HydrophobicScorer()
        
        score = scorer.calculate_hydrophobic_score(
            pipeline.receptor.coordinates,
            pipeline.receptor.elements,
            pipeline.ligand.coordinates,
            pipeline.ligand.elements
        )
        
        print(f"\n✓ Hydrophobic Score calculated")
        print(f"  - Score: {score:.4f}")
        print(f"  - In valid range [0, 1]: {0 <= score <= 1}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vdw_score():
    """Test Van der Waals scoring."""
    print("\n" + "="*60)
    print("TEST 5: Van der Waals Score Calculation")
    print("="*60)
    
    try:
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        from phase2_physics_geometry import VDWCalculator
        calc = VDWCalculator()
        
        score = calc.calculate_vdw_score(
            pipeline.receptor.coordinates,
            pipeline.receptor.elements,
            pipeline.ligand.coordinates,
            pipeline.ligand.elements
        )
        
        print(f"\n✓ VDW Score calculated")
        print(f"  - Energy: {score:.4f} kcal/mol")
        print(f"  - In valid range [-100, 100]: {-100 <= score <= 100}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shape_and_pocket():
    """Test shape matching and pocket fitting."""
    print("\n" + "="*60)
    print("TEST 6: Shape Matching and Pocket Fitting")
    print("="*60)
    
    try:
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        from phase2_physics_geometry import ShapeComplementarityCalculator
        calc = ShapeComplementarityCalculator()
        
        shape = calc.calculate_shape_match(
            pipeline.receptor.coordinates,
            pipeline.ligand.coordinates
        )
        pocket = calc.calculate_pocket_fit(
            pipeline.receptor.coordinates,
            pipeline.ligand.coordinates
        )
        
        print(f"\n✓ Shape and Pocket Scores calculated")
        print(f"  - Shape Match Score: {shape:.4f}")
        print(f"  - Pocket Fit Score: {pocket:.4f}")
        print(f"  - Both in [0, 1]: {0 <= shape <= 1 and 0 <= pocket <= 1}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_complementarity_vector():
    """Test complete Complementarity Vector generation."""
    print("\n" + "="*60)
    print("TEST 7: Full Complementarity Vector Generation")
    print("="*60)
    
    try:
        # Load test data
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        # Generate vector
        generator = ComplementarityVectorGenerator()
        vector, metadata = generator.calculate_complementarity_vector(
            pipeline.receptor,
            pipeline.ligand
        )
        
        print(f"\n✓ Complementarity Vector generated")
        print(f"  - Vector shape: {vector.shape}")
        print(f"  - Vector dtype: {vector.dtype}")
        print(f"  - Has 8 features: {len(vector) == 8}")
        
        print(f"\n✓ Vector Components:")
        print(f"  [0] Distance Score:       {vector[0]:10.4f}")
        print(f"  [1] Angle Score:          {vector[1]:10.4f}")
        print(f"  [2] Electrostatic Energy: {vector[2]:10.4f} kcal/mol")
        print(f"  [3] H-Bond Count:         {vector[3]:10.1f}")
        print(f"  [4] Hydrophobic Score:    {vector[4]:10.4f}")
        print(f"  [5] VDW Score:            {vector[5]:10.4f} kcal/mol")
        print(f"  [6] Shape Match:          {vector[6]:10.4f}")
        print(f"  [7] Pocket Fit:           {vector[7]:10.4f}")
        
        print(f"\n✓ Metadata:")
        print(f"  - Receptor atoms: {metadata['n_receptor_atoms']}")
        print(f"  - Ligand atoms: {metadata['n_ligand_atoms']}")
        print(f"  - Features: {', '.join(metadata['features'])}")
        
        # Print human-readable description
        print(f"\n✓ Vector Description:")
        print(generator.describe_vector(vector))
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_reasonability():
    """Test that vector values are in reasonable ranges."""
    print("\n" + "="*60)
    print("TEST 8: Vector Value Validation")
    print("="*60)
    
    try:
        # Load test data
        pipeline = DataIngestionPipeline()
        test_pdb = Path(__file__).parent / "test_protein.pdb"
        test_sdf = Path(__file__).parent / "test_ligand.sdf"
        
        pipeline.load_receptor(str(test_pdb))
        pipeline.load_ligand(str(test_sdf))
        
        # Generate vector
        generator = ComplementarityVectorGenerator()
        vector, _ = generator.calculate_complementarity_vector(
            pipeline.receptor,
            pipeline.ligand
        )
        
        # Validate ranges
        validations = [
            ("dist_score [0,1]", 0 <= vector[0] <= 1),
            ("angle_score [0,1]", 0 <= vector[1] <= 1),
            ("electrostatic_energy reasonable", -1000 < vector[2] < 1000),
            ("hbond_count non-negative", vector[3] >= 0),
            ("hydrophobic_score [0,1]", 0 <= vector[4] <= 1),
            ("vdW_score reasonable", -100 <= vector[5] <= 100),  # Clamped to [-100, 100]
            ("shape_match [0,1]", 0 <= vector[6] <= 1),
            ("pocket_fit [0,1]", 0 <= vector[7] <= 1),
        ]
        
        all_valid = True
        for check_name, is_valid in validations:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {check_name}: {is_valid}")
            all_valid = all_valid and is_valid
        
        return all_valid
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE 2: PHYSICS & GEOMETRY ENGINE TEST SUITE")
    print("="*60)
    
    results = {
        'Distance & Angle Scores': test_distance_and_angle_scores(),
        'Electrostatic Energy': test_electrostatic_energy(),
        'Hydrogen Bonds': test_hydrogen_bonds(),
        'Hydrophobic Score': test_hydrophobic_score(),
        'VDW Score': test_vdw_score(),
        'Shape & Pocket': test_shape_and_pocket(),
        'Full Vector Generation': test_full_complementarity_vector(),
        'Vector Validation': test_vector_reasonability(),
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
        print("\nPhase 2 is ready for the next phases.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the errors above.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
