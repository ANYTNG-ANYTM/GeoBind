"""
Test Script for Phase 3: Normalization & ML Pipeline.

This suite verifies:
1. Z-score normalization behavior.
2. Cosine similarity formula implementation.
3. End-to-end XGBoost training from CSV.
4. Artifact generation (.pkl model + feature importance plot).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import logging
import sys
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.ml.phase3_ml_pipeline import (
    FEATURE_COLUMNS,
    FeaturePreprocessor,
    TrainingArtifacts,
    train_xgboost_from_csv,
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def generate_synthetic_dataset(output_csv: Path, n_samples: int = 400, seed: int = 42) -> Path:
    """Generate synthetic complementarity vectors for binders (1) and decoys (0)."""
    rng = np.random.default_rng(seed)

    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # Positive class: generally better complementarity
    pos = pd.DataFrame(
        {
            "dist_score": rng.normal(0.72, 0.10, n_pos).clip(0, 1),
            "angle_score": rng.normal(0.70, 0.12, n_pos).clip(0, 1),
            "electrostatic_energy": rng.normal(-18.0, 8.0, n_pos),
            "hbond_count": rng.poisson(3.2, n_pos),
            "hydrophobic_score": rng.normal(0.68, 0.12, n_pos).clip(0, 1),
            "vdW_score": rng.normal(-14.0, 6.0, n_pos),
            "shape_match": rng.normal(0.64, 0.10, n_pos).clip(0, 1),
            "pocket_fit": rng.normal(0.70, 0.10, n_pos).clip(0, 1),
            "label": 1,
        }
    )

    # Negative class: poorer complementarity
    neg = pd.DataFrame(
        {
            "dist_score": rng.normal(0.28, 0.12, n_neg).clip(0, 1),
            "angle_score": rng.normal(0.35, 0.14, n_neg).clip(0, 1),
            "electrostatic_energy": rng.normal(8.0, 10.0, n_neg),
            "hbond_count": rng.poisson(0.8, n_neg),
            "hydrophobic_score": rng.normal(0.24, 0.13, n_neg).clip(0, 1),
            "vdW_score": rng.normal(6.0, 10.0, n_neg),
            "shape_match": rng.normal(0.22, 0.12, n_neg).clip(0, 1),
            "pocket_fit": rng.normal(0.30, 0.13, n_neg).clip(0, 1),
            "label": 0,
        }
    )

    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def test_normalization() -> bool:
    _print_header("TEST 1: Z-Score Normalization")
    try:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(120, 8)).astype(np.float32)
        pre = FeaturePreprocessor()
        Xn = pre.fit_transform(X)

        means = Xn.mean(axis=0)
        stds = Xn.std(axis=0)

        mean_ok = np.allclose(means, 0.0, atol=1e-6)
        std_ok = np.allclose(stds, 1.0, atol=1e-6)

        print(f"Mean near 0: {mean_ok}")
        print(f"Std near 1: {std_ok}")
        return bool(mean_ok and std_ok)
    except Exception as exc:
        print(f"Error: {exc}")
        return False


def test_cosine_similarity() -> bool:
    _print_header("TEST 2: Cosine Similarity")
    try:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        v4 = np.array([-1.0, 0.0, 0.0])

        s12 = FeaturePreprocessor.cosine_similarity(v1, v2)
        s13 = FeaturePreprocessor.cosine_similarity(v1, v3)
        s14 = FeaturePreprocessor.cosine_similarity(v1, v4)

        print(f"cos(v1, v2) = {s12:.4f} (expected ~1)")
        print(f"cos(v1, v3) = {s13:.4f} (expected ~0)")
        print(f"cos(v1, v4) = {s14:.4f} (expected ~-1)")

        return (
            abs(s12 - 1.0) < 1e-9
            and abs(s13 - 0.0) < 1e-9
            and abs(s14 + 1.0) < 1e-9
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return False


def test_xgboost_training(base_dir: Path) -> bool:
    _print_header("TEST 3: XGBoost Training from CSV")
    try:
        data_csv = base_dir / "data" / "phase3_synthetic_vectors.csv"
        model_path = base_dir / "models" / "geobind_xgb_test.pkl"
        plot_path = base_dir / "models" / "feature_importance_test.png"

        generate_synthetic_dataset(data_csv, n_samples=400, seed=42)

        results = train_xgboost_from_csv(
            csv_path=data_csv,
            output_model_path=model_path,
            output_plot_path=plot_path,
            label_column="label",
            feature_columns=FEATURE_COLUMNS,
            random_state=42,
        )

        print("Metrics:")
        for key, value in results["metrics"].items():
            print(f"  {key:10s}: {value:.4f}")

        model_exists = model_path.exists()
        plot_exists = plot_path.exists()

        print(f"Model saved: {model_exists} -> {model_path}")
        print(f"Plot saved:  {plot_exists} -> {plot_path}")

        # Load artifact and validate type
        artifacts = joblib.load(model_path)
        is_artifacts = isinstance(artifacts, TrainingArtifacts)
        print(f"Artifact type valid: {is_artifacts}")

        # Basic quality gate for synthetic separable dataset
        auc_ok = results["metrics"]["roc_auc"] >= 0.85
        f1_ok = results["metrics"]["f1"] >= 0.80
        print(f"AUC >= 0.85: {auc_ok}")
        print(f"F1  >= 0.80: {f1_ok}")

        return bool(model_exists and plot_exists and is_artifacts and auc_ok and f1_ok)
    except Exception as exc:
        print(f"Error: {exc}")
        return False


def main() -> int:
    base_dir = Path(__file__).parent

    test_results = {
        "Normalization": test_normalization(),
        "Cosine Similarity": test_cosine_similarity(),
        "XGBoost Training": test_xgboost_training(base_dir),
    }

    _print_header("PHASE 3 TEST SUMMARY")
    for name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

    all_passed = all(test_results.values())
    print("\nALL TESTS PASSED" if all_passed else "\nSOME TESTS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
