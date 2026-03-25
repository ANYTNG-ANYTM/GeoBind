"""
Tests for phase3_dataset_builder.py.

This validates:
1. Positive/negative manifest generation workflow.
2. Negative pair shuffling.
3. Feature vectorization from manifest into training CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from src.ml.phase3_dataset_builder import (
    TrainingDatasetBuilder,
    combine_positive_and_negative_manifests,
    create_manifest_template,
    generate_negative_pairs_by_shuffling,
)
from src.ml.phase3_ml_pipeline import FEATURE_COLUMNS
from test_phase1 import create_minimal_test_pdb


def _write_ligand(smiles: str, out_sdf: Path, seed: int) -> None:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(str(out_sdf))
    writer.write(mol)
    writer.close()


def test_dataset_builder_workflow(base_dir: Path) -> bool:
    print("\n" + "=" * 60)
    print("TEST: Phase 3 Dataset Builder Workflow")
    print("=" * 60)

    try:
        fixtures = base_dir / "fixtures_phase3"
        fixtures.mkdir(exist_ok=True)

        # Receptors (2 PDB files)
        receptor_a = fixtures / "receptor_A.pdb"
        receptor_b = fixtures / "receptor_B.pdb"
        create_minimal_test_pdb(str(receptor_a))
        create_minimal_test_pdb(str(receptor_b))

        # Ligands (3 SDF files)
        lig_a = fixtures / "ligand_A.sdf"
        lig_b = fixtures / "ligand_B.sdf"
        lig_c = fixtures / "ligand_C.sdf"
        _write_ligand("c1ccccc1", lig_a, seed=11)  # benzene
        _write_ligand("CCO", lig_b, seed=13)       # ethanol
        _write_ligand("CCN", lig_c, seed=17)       # ethylamine

        # Positive manifest
        pos_manifest = fixtures / "positive_manifest.csv"
        pos_df = pd.DataFrame(
            [
                {"pair_id": "POS_1", "receptor_file": str(receptor_a), "ligand_file": str(lig_a), "label": 1, "chain_id": "A"},
                {"pair_id": "POS_2", "receptor_file": str(receptor_b), "ligand_file": str(lig_b), "label": 1, "chain_id": "A"},
                {"pair_id": "POS_3", "receptor_file": str(receptor_a), "ligand_file": str(lig_c), "label": 1, "chain_id": "A"},
            ]
        )
        pos_df.to_csv(pos_manifest, index=False)

        # Generate negatives
        neg_manifest = fixtures / "negative_manifest.csv"
        generate_negative_pairs_by_shuffling(
            positive_manifest_csv=pos_manifest,
            output_negative_manifest_csv=neg_manifest,
            negatives_per_positive=1,
            seed=42,
        )

        neg_df = pd.read_csv(neg_manifest)
        print(f"Generated negatives: {len(neg_df)}")
        if len(neg_df) == 0:
            return False

        # Combine manifests
        combined_manifest = fixtures / "combined_manifest.csv"
        combine_positive_and_negative_manifests(
            positive_manifest_csv=pos_manifest,
            negative_manifest_csv=neg_manifest,
            output_manifest_csv=combined_manifest,
        )
        combined_df = pd.read_csv(combined_manifest)
        print(
            "Combined manifest rows="
            f"{len(combined_df)} positives={(combined_df['label'] == 1).sum()} negatives={(combined_df['label'] == 0).sum()}"
        )

        # Vectorize combined manifest
        output_vectors = fixtures / "training_vectors.csv"
        failed_log = fixtures / "training_vectors_failed.csv"
        builder = TrainingDatasetBuilder()
        summary = builder.vectorize_manifest(
            manifest_csv=combined_manifest,
            output_csv=output_vectors,
            failed_log_csv=failed_log,
        )

        print(
            "Vectorization summary: "
            f"total={summary.total_rows} success={summary.successful_rows} failed={summary.failed_rows}"
        )

        vectors_df = pd.read_csv(output_vectors)
        missing_cols = [c for c in FEATURE_COLUMNS + ["label", "pair_id"] if c not in vectors_df.columns]

        print(f"Output vectors rows: {len(vectors_df)}")
        print(f"Missing required columns: {missing_cols}")

        return (
            summary.successful_rows > 0
            and summary.failed_rows == 0
            and len(vectors_df) == len(combined_df)
            and len(missing_cols) == 0
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return False


def main() -> int:
    base_dir = Path(__file__).parent
    ok = test_dataset_builder_workflow(base_dir)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
