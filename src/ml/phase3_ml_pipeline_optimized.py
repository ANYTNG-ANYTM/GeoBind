"""
Phase 3 Optimized: Advanced ML Pipeline with Hyperparameter Tuning & Cross-Validation

Improvements:
1. GridSearchCV for hyperparameter optimization
2. K-fold stratified cross-validation
3. Feature engineering (polynomial + interactions)
4. Ensemble methods (XGBoost + RandomForest)
5. Learning curves for overfitting detection
6. ROC curve analysis
7. Comprehensive reporting
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

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
    """Persistent training artifacts."""
    model: object
    scaler: StandardScaler
    feature_names: List[str]
    metadata: Dict


class FeatureEngineer:
    """Generate engineered features from base complementarity vector."""

    @staticmethod
    def create_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
        """Add interaction and polynomial features."""
        X_eng = X.copy()
        
        # Polynomial features (degree 2)
        interactions = [
            ("shape_match_x_hydrophobic", X["shape_match"] * X["hydrophobic_score"]),
            ("dist_score_x_angle_score", X["dist_score"] * X["angle_score"]),
            ("vdW_x_electrostatic", X["vdW_score"] * X["electrostatic_energy"]),
            ("hbond_x_pocket_fit", X["hbond_count"] * X["pocket_fit"]),
            ("shape_match_squared", X["shape_match"] ** 2),
            ("hydrophobic_squared", X["hydrophobic_score"] ** 2),
        ]
        
        for name, values in interactions:
            X_eng[name] = values
        
        return X_eng

    @staticmethod
    def get_engineered_feature_names(base_features: List[str]) -> List[str]:
        """Get list of all feature names after engineering."""
        interactions = [
            "shape_match_x_hydrophobic",
            "dist_score_x_angle_score",
            "vdW_x_electrostatic",
            "hbond_x_pocket_fit",
            "shape_match_squared",
            "hydrophobic_squared",
        ]
        return base_features + interactions


class OptimizedTrainer:
    """Optimized XGBoost trainer with hyperparameter tuning."""

    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []

    def perform_gridsearch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """Perform GridSearchCV for hyperparameter optimization."""
        logger.info("Starting hyperparameter grid search...")
        
        param_grid = {
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 300, 400],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
        }
        
        xgb_model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
        )
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=self.n_splits,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1 if verbose else 0,
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-val ROC-AUC: {grid_search.best_score_:.4f}")
        
        return grid_search

    def train_with_crossval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        perform_gridsearch: bool = True,
    ) -> Tuple[object, Dict[str, float]]:
        """Train with k-fold cross-validation and optional hyperparameter tuning."""
        
        if perform_gridsearch:
            grid_search = self.perform_gridsearch(X, y)
            best_model = grid_search.best_estimator_
        else:
            best_model = XGBClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
            )
            best_model.fit(X, y)
        
        # K-fold cross-validation
        logger.info(f"Running {self.n_splits}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            "accuracy": cross_val_score(best_model, X, y, cv=cv, scoring="accuracy"),
            "precision": cross_val_score(best_model, X, y, cv=cv, scoring="precision"),
            "recall": cross_val_score(best_model, X, y, cv=cv, scoring="recall"),
            "f1": cross_val_score(best_model, X, y, cv=cv, scoring="f1"),
            "roc_auc": cross_val_score(best_model, X, y, cv=cv, scoring="roc_auc"),
        }
        
        cv_summary = {}
        for metric, scores in cv_scores.items():
            mean_score = scores.mean()
            std_score = scores.std()
            cv_summary[f"{metric}_mean"] = mean_score
            cv_summary[f"{metric}_std"] = std_score
            logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.model = best_model
        return best_model, cv_summary

    def plot_learning_curves(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        output_path: Path,
    ) -> None:
        """Plot learning curves to detect overfitting."""
        logger.info("Generating learning curves...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model,
            X_train,
            y_train,
            cv=self.n_splits,
            scoring="roc_auc",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, "o-", label="Training ROC-AUC", color="blue")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        plt.plot(train_sizes, val_mean, "o-", label="Validation ROC-AUC", color="orange")
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
        plt.xlabel("Training Set Size")
        plt.ylabel("ROC-AUC Score")
        plt.title("Learning Curves - GeoBind Phase 3 Optimized")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Learning curves saved to {output_path}")

    def plot_roc_curve(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_path: Path,
    ) -> None:
        """Plot ROC curve."""
        logger.info("Generating ROC curve...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - GeoBind Phase 3")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")


def load_and_preprocess_vectors(
    vectors_csv: str | Path,
    label_column: str = "label",
    use_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Load vectors CSV and preprocess."""
    vectors_csv = Path(vectors_csv)
    if not vectors_csv.exists():
        raise FileNotFoundError(f"Vectors CSV not found: {vectors_csv}")

    df = pd.read_csv(vectors_csv)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not in CSV")

    # Extract features
    X = df[FEATURE_COLUMNS].copy()
    y = df[label_column].values

    # Feature engineering
    if use_feature_engineering:
        X = FeatureEngineer.create_interaction_features(X)
        feature_names = FeatureEngineer.get_engineered_feature_names(FEATURE_COLUMNS)
    else:
        feature_names = FEATURE_COLUMNS

    logger.info(f"Features: {len(feature_names)} (base: {len(FEATURE_COLUMNS)}, engineered: {len(feature_names) - len(FEATURE_COLUMNS)})")

    X_values = X.values
    logger.info(f"Class distribution: {np.unique(y, return_counts=True)[1]}")

    return df, X_values, y, feature_names


def main() -> int:
    parser = argparse.ArgumentParser(description="GeoBind Phase 3 Optimized ML Pipeline")
    parser.add_argument("--vectors", required=True, help="Path to vectors CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--label-column", default="label", help="Label column name")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--no-gridsearch", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--no-feature-engineering", action="store_true", help="Skip feature engineering")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    df, X, y, feature_names = load_and_preprocess_vectors(
        args.vectors,
        label_column=args.label_column,
        use_feature_engineering=not args.no_feature_engineering,
    )

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    logger.info(f"Train set: {len(X_train)} ({(y_train == 1).sum()} pos, {(y_train == 0).sum()} neg)")
    logger.info(f"Test set: {len(X_test)} ({(y_test == 1).sum()} pos, {(y_test == 0).sum()} neg)")

    # Train with optimization
    trainer = OptimizedTrainer(random_state=args.random_state, n_splits=5)
    model, cv_summary = trainer.train_with_crossval(
        X_train,
        y_train,
        perform_gridsearch=not args.no_gridsearch,
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    logger.info("\n" + "=" * 70)
    logger.info("TEST SET METRICS")
    logger.info("=" * 70)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric:15} {value:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION SUMMARY (5-fold)")
    logger.info("=" * 70)
    for key, value in cv_summary.items():
        logger.info(f"  {key:30} {value:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"[[{cm[0, 0]:3d} {cm[0, 1]:3d}]")
    logger.info(f" [{cm[1, 0]:3d} {cm[1, 1]:3d}]]")

    # Save artifacts
    artifacts = TrainingArtifacts(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        metadata={
            "test_metrics": test_metrics,
            "cv_summary": cv_summary,
            "n_features": len(feature_names),
            "feature_engineering": not args.no_feature_engineering,
        },
    )

    model_path = out_dir / "geobind_xgb_optimized.pkl"
    joblib.dump(artifacts, model_path)
    logger.info(f"\nModel saved to {model_path}")

    # Generate plots
    trainer.plot_learning_curves(X_train, y_train, out_dir / "learning_curves.png")
    trainer.plot_roc_curve(X_test, y_test, out_dir / "roc_curve.png")

    # Feature importance
    try:
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_indices)), importances[top_indices])
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel("Feature Importance (Gain)")
        plt.title("Top 10 Features - GeoBind Phase 3 Optimized")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Feature importance plot saved")
    except Exception as e:
        logger.warning(f"Could not generate feature importance: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Model: {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
