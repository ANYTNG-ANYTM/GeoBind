"""
Phase 3 Dataset Builder for GeoBind.

Purpose:
1. Build training-ready vector datasets from receptor-ligand manifests.
2. Generate negative (non-binder) receptor-ligand pairs by decoy shuffling.
3. Export CSVs directly compatible with phase3_ml_pipeline.py.

Expected manifest columns:
- pair_id: unique identifier for the receptor-ligand pair
- receptor_file: path to receptor PDB file
- ligand_file: path to ligand SDF/MOL2 file
- label: 1 for binder, 0 for non-binder/decoy
Optional:
- chain_id: protein chain to isolate (default: first chain)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.phase1_data_ingestion import AtomicCoordinates, DataIngestionPipeline
from ..core.phase2_physics_geometry import ComplementarityVectorGenerator
from .phase3_ml_pipeline import FEATURE_COLUMNS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


REQUIRED_MANIFEST_COLUMNS = ["pair_id", "receptor_file", "ligand_file", "label"]


@dataclass
class VectorizationSummary:
    """Summary of vectorization results."""

    total_rows: int
    successful_rows: int
    failed_rows: int
    output_csv: str
    failed_log_csv: Optional[str]


class ManifestValidator:
    """Validation helper for receptor-ligand manifests."""

    @staticmethod
    def validate_manifest(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError("Manifest label column must contain only 0/1 values.")

        if df["pair_id"].duplicated().any():
            duplicates = df[df["pair_id"].duplicated()]["pair_id"].tolist()[:5]
            raise ValueError(f"Manifest pair_id must be unique. Example duplicates: {duplicates}")


class TrainingDatasetBuilder:
    """
    Build feature-vector datasets from receptor-ligand manifests.

    Output columns are exactly FEATURE_COLUMNS plus:
    - label
    - pair_id
    - receptor_file
    - ligand_file
    - chain_id
    """

    def __init__(self) -> None:
        self.vector_generator = ComplementarityVectorGenerator()
        self._receptor_cache: Dict[Tuple[str, Optional[str]], AtomicCoordinates] = {}

    def _resolve_path(self, path_value: str, root_dir: Optional[Path]) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        if root_dir is not None:
            return (root_dir / path).resolve()
        return path.resolve()

    def _load_receptor_cached(self, receptor_path: Path, chain_id: Optional[str]) -> AtomicCoordinates:
        key = (str(receptor_path), chain_id)
        if key in self._receptor_cache:
            return self._receptor_cache[key]

        pipeline = DataIngestionPipeline()
        pipeline.load_receptor(str(receptor_path), chain_id=chain_id)
        self._receptor_cache[key] = pipeline.receptor
        return pipeline.receptor

    def vectorize_manifest(
        self,
        manifest_csv: str | Path,
        output_csv: str | Path,
        receptor_root_dir: Optional[str | Path] = None,
        ligand_root_dir: Optional[str | Path] = None,
        failed_log_csv: Optional[str | Path] = None,
    ) -> VectorizationSummary:
        """
        Vectorize all receptor-ligand rows in a manifest.

        Args:
            manifest_csv: Path to input manifest CSV.
            output_csv: Path to output vectors CSV.
            receptor_root_dir: Base directory for relative receptor_file paths.
            ligand_root_dir: Base directory for relative ligand_file paths.
            failed_log_csv: Optional path to write failed rows + error details.
        """
        manifest_csv = Path(manifest_csv)
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        receptor_root = Path(receptor_root_dir).resolve() if receptor_root_dir else None
        ligand_root = Path(ligand_root_dir).resolve() if ligand_root_dir else None

        df = pd.read_csv(manifest_csv)
        ManifestValidator.validate_manifest(df)

        successful_rows: List[Dict[str, object]] = []
        failed_rows: List[Dict[str, object]] = []

        total = len(df)
        logger.info("Vectorizing manifest rows: %d", total)

        for idx, row in df.iterrows():
            pair_id = str(row["pair_id"])
            chain_id = str(row["chain_id"]).strip() if "chain_id" in df.columns and pd.notna(row.get("chain_id")) else None

            try:
                receptor_path = self._resolve_path(str(row["receptor_file"]), receptor_root)
                ligand_path = self._resolve_path(str(row["ligand_file"]), ligand_root)

                receptor = self._load_receptor_cached(receptor_path, chain_id)

                pipeline = DataIngestionPipeline()
                pipeline.ligand = None
                pipeline.load_ligand(str(ligand_path))
                ligand = pipeline.ligand

                vector, _ = self.vector_generator.calculate_complementarity_vector(receptor, ligand)

                feature_data = {name: float(value) for name, value in zip(FEATURE_COLUMNS, vector)}
                feature_data.update(
                    {
                        "label": int(row["label"]),
                        "pair_id": pair_id,
                        "receptor_file": str(receptor_path),
                        "ligand_file": str(ligand_path),
                        "chain_id": chain_id or "",
                    }
                )
                successful_rows.append(feature_data)

            except Exception as exc:
                failed_rows.append(
                    {
                        "row_index": int(idx),
                        "pair_id": pair_id,
                        "receptor_file": row.get("receptor_file", ""),
                        "ligand_file": row.get("ligand_file", ""),
                        "label": row.get("label", ""),
                        "error": str(exc),
                    }
                )

        result_df = pd.DataFrame(successful_rows)
        if not result_df.empty:
            ordered_cols = FEATURE_COLUMNS + ["label", "pair_id", "receptor_file", "ligand_file", "chain_id"]
            result_df = result_df[ordered_cols]
        result_df.to_csv(output_csv, index=False)

        failed_path = None
        if failed_rows:
            failed_path_obj = Path(failed_log_csv) if failed_log_csv else output_csv.with_name(f"{output_csv.stem}_failed.csv")
            failed_path_obj.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failed_rows).to_csv(failed_path_obj, index=False)
            failed_path = str(failed_path_obj)

        summary = VectorizationSummary(
            total_rows=total,
            successful_rows=len(successful_rows),
            failed_rows=len(failed_rows),
            output_csv=str(output_csv),
            failed_log_csv=failed_path,
        )
        logger.info(
            "Vectorization done. Total=%d Success=%d Failed=%d",
            summary.total_rows,
            summary.successful_rows,
            summary.failed_rows,
        )
        return summary


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoBind Phase 3 dataset vector builder")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV path")
    parser.add_argument("--out-csv", required=True, help="Output vectors CSV path")
    parser.add_argument(
        "--receptor-root-dir",
        default=None,
        help="Optional base directory for relative receptor_file paths",
    )
    parser.add_argument(
        "--ligand-root-dir",
        default=None,
        help="Optional base directory for relative ligand_file paths",
    )
    parser.add_argument(
        "--failed-log-csv",
        default=None,
        help="Optional CSV path to write failed rows",
    )
    return parser


def main() -> int:
    args = _build_cli_parser().parse_args()
    builder = TrainingDatasetBuilder()
    summary = builder.vectorize_manifest(
        manifest_csv=args.manifest,
        output_csv=args.out_csv,
        receptor_root_dir=args.receptor_root_dir,
        ligand_root_dir=args.ligand_root_dir,
        failed_log_csv=args.failed_log_csv,
    )
    logger.info(
        "Vectorization summary: total=%d success=%d failed=%d output=%s",
        summary.total_rows,
        summary.successful_rows,
        summary.failed_rows,
        summary.output_csv,
    )
    if summary.failed_log_csv:
        logger.info("Failed rows written to: %s", summary.failed_log_csv)
    return 0


def generate_negative_pairs_by_shuffling(
    positive_manifest_csv: str | Path,
    output_negative_manifest_csv: str | Path,
    negatives_per_positive: int = 1,
    seed: int = 42,
) -> Path:
    """
    Generate negative receptor-ligand pairs by mismatching ligands to receptors.

    Strategy:
    - Start from positive pairs.
    - For each receptor, sample ligands from different positive pairs.
    - Ensure generated (receptor_file, ligand_file) is not an existing positive pair.

    Output CSV uses same schema as manifest with label fixed to 0.
    """
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be >= 1")

    pos_path = Path(positive_manifest_csv)
    if not pos_path.exists():
        raise FileNotFoundError(f"Positive manifest not found: {pos_path}")

    pos_df = pd.read_csv(pos_path)
    ManifestValidator.validate_manifest(pos_df)

    pos_df = pos_df[pos_df["label"] == 1].copy()
    if len(pos_df) < 2:
        raise ValueError("Need at least 2 positive pairs to generate shuffled negatives.")

    rng = np.random.default_rng(seed)
    existing_pos = set(zip(pos_df["receptor_file"], pos_df["ligand_file"]))

    ligands = pos_df[["pair_id", "ligand_file"]].to_dict("records")

    neg_rows: List[Dict[str, object]] = []

    for _, row in pos_df.iterrows():
        receptor_file = row["receptor_file"]
        chain_id = row["chain_id"] if "chain_id" in pos_df.columns else ""

        created = 0
        attempts = 0
        max_attempts = max(50, negatives_per_positive * 20)

        while created < negatives_per_positive and attempts < max_attempts:
            attempts += 1
            candidate = ligands[rng.integers(0, len(ligands))]
            ligand_file = candidate["ligand_file"]

            if (receptor_file, ligand_file) in existing_pos:
                continue

            neg_rows.append(
                {
                    "pair_id": f"NEG_{row['pair_id']}_{created + 1}",
                    "receptor_file": receptor_file,
                    "ligand_file": ligand_file,
                    "label": 0,
                    "chain_id": chain_id,
                }
            )
            created += 1

        if created < negatives_per_positive:
            logger.warning(
                "Could only create %d/%d negatives for receptor pair_id=%s",
                created,
                negatives_per_positive,
                row["pair_id"],
            )

    out_path = Path(output_negative_manifest_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(neg_rows).to_csv(out_path, index=False)

    logger.info("Generated %d negative rows at %s", len(neg_rows), out_path)
    return out_path


def combine_positive_and_negative_manifests(
    positive_manifest_csv: str | Path,
    negative_manifest_csv: str | Path,
    output_manifest_csv: str | Path,
) -> Path:
    """Combine positive and negative manifests into one training manifest."""
    pos_df = pd.read_csv(positive_manifest_csv)
    neg_df = pd.read_csv(negative_manifest_csv)

    ManifestValidator.validate_manifest(pos_df)
    ManifestValidator.validate_manifest(neg_df)

    combined = pd.concat([pos_df, neg_df], ignore_index=True)

    if combined["pair_id"].duplicated().any():
        # Make pair_id unique deterministically when combining.
        counts: Dict[str, int] = {}
        new_ids: List[str] = []
        for pid in combined["pair_id"].astype(str):
            counts[pid] = counts.get(pid, 0) + 1
            suffix = counts[pid]
            new_ids.append(pid if suffix == 1 else f"{pid}_{suffix}")
        combined["pair_id"] = new_ids

    output_manifest_csv = Path(output_manifest_csv)
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_manifest_csv, index=False)

    logger.info(
        "Combined manifest saved to %s (rows=%d, positives=%d, negatives=%d)",
        output_manifest_csv,
        len(combined),
        int((combined["label"] == 1).sum()),
        int((combined["label"] == 0).sum()),
    )
    return output_manifest_csv


def create_manifest_template(output_csv: str | Path) -> Path:
    """Create an empty manifest template CSV with required columns."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        columns=["pair_id", "receptor_file", "ligand_file", "label", "chain_id"]
    )
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    raise SystemExit(main())
