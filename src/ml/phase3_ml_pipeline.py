"""
Phase 3: Normalization & ML Pipeline for GeoBind.

This module implements:
1. Z-score normalization of complementarity vectors.
2. Cosine similarity between normalized vectors.
3. XGBoost binary classifier training from CSV data.
4. Train/validation split (80/20), model persistence, and feature importance plot.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FEATURE_COLUMNS: List[str] = [
    "dist_score",
    "angle_score",
    "electrostatic_energy",
    "hbond_count",
    "hydrophobic_score",
    "vdW_score",
    "shape_match",
    "pocket_fit",
]


@dataclass
class TrainingArtifacts:
    """Container for persisted Phase 3 artifacts."""

    model: XGBClassifier
    scaler: StandardScaler
    feature_columns: List[str]
    metrics: Dict[str, float]


class FeaturePreprocessor:
    """Preprocessing utilities for complementarity vectors."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and return Z-score normalized feature matrix."""
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using an already-fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted. Call fit_transform first.")
        X = np.asarray(X, dtype=np.float32)
        return self.scaler.transform(X).astype(np.float32)

    @staticmethod
    def cosine_similarity(vector_1: Sequence[float], vector_2: Sequence[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        S = (V1 dot V2) / (||V1|| * ||V2||)
        """
        v1 = np.asarray(vector_1, dtype=np.float64)
        v2 = np.asarray(vector_2, dtype=np.float64)

        if v1.shape != v2.shape:
            raise ValueError(f"Vector shapes must match. Got {v1.shape} vs {v2.shape}")

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            raise ValueError("Cosine similarity is undefined for zero-magnitude vectors.")

        similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        return similarity


class XGBoostBindingPredictorTrainer:
    """Training pipeline for binding prediction model."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.preprocessor = FeaturePreprocessor()
        self.model: Optional[XGBClassifier] = None

    def load_dataset(
        self,
        csv_path: str | Path,
        label_column: str = "label",
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Load dataset and validate schema."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if label_column not in df.columns:
            raise ValueError(f"Missing required label column: {label_column}")

        selected_features = feature_columns or FEATURE_COLUMNS
        missing_features = [col for col in selected_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing feature columns: {missing_features}")

        X = df[selected_features].copy()
        y = df[label_column].astype(int)

        if y.nunique() < 2:
            raise ValueError("Training requires both positive and negative samples.")

        logger.info("Loaded dataset with %d rows and %d features", len(df), len(selected_features))
        logger.info("Class distribution: %s", y.value_counts().to_dict())
        return X, y, selected_features

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, object]:
        """Train XGBoost model with an 80/20 split by default."""
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        stratify_labels = y if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        X_train_scaled = self.preprocessor.fit_transform(X_train.values)
        X_test_scaled = self.preprocessor.transform(X_test.values)

        # Keep feature names after scaling so feature importance is interpretable.
        X_train_scaled_df = pd.DataFrame(
            X_train_scaled,
            columns=list(X.columns),
            index=X_train.index,
        )
        X_test_scaled_df = pd.DataFrame(
            X_test_scaled,
            columns=list(X.columns),
            index=X_test.index,
        )

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=4,
        )

        self.model.fit(X_train_scaled_df, y_train)

        y_pred = self.model.predict(X_test_scaled_df)
        y_prob = self.model.predict_proba(X_test_scaled_df)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        logger.info("Training complete. Metrics: %s", metrics)

        return {
            "metrics": metrics,
            "X_train_shape": X_train_scaled_df.shape,
            "X_test_shape": X_test_scaled_df.shape,
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def save_artifacts(
        self,
        output_model_path: str | Path,
        feature_columns: List[str],
        metrics: Dict[str, float],
    ) -> Path:
        """Persist model + scaler as a single .pkl file."""
        if self.model is None:
            raise ValueError("No trained model found. Train the model before saving.")

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        artifacts = TrainingArtifacts(
            model=self.model,
            scaler=self.preprocessor.scaler,
            feature_columns=feature_columns,
            metrics=metrics,
        )
        joblib.dump(artifacts, output_model_path)
        logger.info("Saved model artifacts to %s", output_model_path)
        return output_model_path

    def plot_feature_importance(
        self,
        output_plot_path: str | Path,
        max_num_features: int = 8,
    ) -> Path:
        """Generate and save feature importance plot."""
        if self.model is None:
            raise ValueError("No trained model found. Train model before plotting.")

        output_plot_path = Path(output_plot_path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            plt.figure(figsize=(10, 6))
            plot_importance(
                self.model,
                max_num_features=max_num_features,
                importance_type="gain",
                title="GeoBind XGBoost Feature Importance",
                xlabel="Gain",
            )
            plt.tight_layout()
            plt.savefig(output_plot_path, dpi=300)
            plt.close()
        except ValueError:
            # Fallback for degenerate models where Booster.get_score() is empty.
            feature_names = getattr(self.model, "feature_names_in_", None)
            importances = getattr(self.model, "feature_importances_", None)
            if importances is None:
                raise

            importances = np.asarray(importances, dtype=np.float64)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(len(importances))]

            order = np.argsort(importances)[::-1]
            order = order[:max_num_features]

            plt.figure(figsize=(10, 6))
            plt.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
            plt.title("GeoBind XGBoost Feature Importance (Fallback)")
            plt.xlabel("Feature importance")
            plt.tight_layout()
            plt.savefig(output_plot_path, dpi=300)
            plt.close()

        logger.info("Saved feature importance plot to %s", output_plot_path)
        return output_plot_path


def train_xgboost_from_csv(
    csv_path: str | Path,
    output_model_path: str | Path = "models/geobind_xgb.pkl",
    output_plot_path: str | Path = "models/feature_importance.png",
    label_column: str = "label",
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Convenience function to run the full Phase 3 training pipeline.

    Returns a dictionary with metrics and saved artifact paths.
    """
    trainer = XGBoostBindingPredictorTrainer(random_state=random_state)
    X, y, selected_features = trainer.load_dataset(
        csv_path=csv_path,
        label_column=label_column,
        feature_columns=feature_columns,
    )

    train_results = trainer.train(X, y, test_size=0.2)
    saved_model = trainer.save_artifacts(
        output_model_path=output_model_path,
        feature_columns=selected_features,
        metrics=train_results["metrics"],
    )
    saved_plot = trainer.plot_feature_importance(output_plot_path=output_plot_path)

    return {
        "model_path": str(saved_model),
        "feature_importance_plot": str(saved_plot),
        "metrics": train_results["metrics"],
        "classification_report": train_results["classification_report"],
        "confusion_matrix": train_results["confusion_matrix"],
        "X_train_shape": train_results["X_train_shape"],
        "X_test_shape": train_results["X_test_shape"],
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoBind Phase 3 XGBoost trainer")
    parser.add_argument("--vectors", required=True, help="Input vectors CSV path")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for model and feature-importance artifacts",
    )
    parser.add_argument("--label-column", default="label", help="Label column name in vectors CSV")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser


def main() -> int:
    args = _build_cli_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "geobind_xgb.pkl"
    plot_path = out_dir / "feature_importance.png"

    results = train_xgboost_from_csv(
        csv_path=args.vectors,
        output_model_path=model_path,
        output_plot_path=plot_path,
        label_column=args.label_column,
        random_state=args.random_state,
    )

    logger.info("Training complete. Model: %s", results["model_path"])
    logger.info("Feature importance plot: %s", results["feature_importance_plot"])
    logger.info("Metrics: %s", results["metrics"])
    logger.info("Train shape: %s, Test shape: %s", results["X_train_shape"], results["X_test_shape"])
    logger.info("Confusion matrix:\n%s", results["confusion_matrix"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
