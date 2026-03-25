#!/usr/bin/env python3
"""
Master Data Combiner for Phase 3:

Combines training data from:
1. master_data/final_training_manifest_*.csv (PDB+ChEMBL pairs with affinity)
2. data/real_training_*/manifests/train_manifest_sampled.csv (FGFR family data)

Output: Large balanced training dataset for Phase 3 optimization
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_filter_manifest(csv_path: str) -> pd.DataFrame:
    """Load manifest and ensure required columns exist."""
    df = pd.read_csv(csv_path)
    
    # Ensure required columns
    required = ["pair_id", "receptor_file", "ligand_file", "label"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    
    # Ensure label is 0 or 1
    if not df["label"].isin([0, 1]).all():
        return pd.DataFrame()
    
    return df[required + (["chain_id"] if "chain_id" in df.columns else [])].copy()


def combine_all_datasets() -> pd.DataFrame:
    """Combine all training datasets available in workspace."""
    logger.info("="*70)
    logger.info("MASTER DATA COMBINATION FOR PHASE 3")
    logger.info("="*70 + "\n")
    
    all_dataframes = []
    source_summaries = []
    
    # Source 1: master_data manifests
    logger.info("📁 SOURCE 1: master_data/final_training_manifest_*.csv")
    logger.info("-" * 70)
    
    master_data_dir = Path("master_data")
    master_manifests = list(master_data_dir.glob("final_training_manifest_*.csv"))
    
    for manifest_file in sorted(master_manifests):
        # Skip the combined FGFR one we already have
        if "combined_fgfr" in manifest_file.name:
            continue
            
        df = load_and_filter_manifest(manifest_file)
        if len(df) == 0:
            continue
        
        pos_count = (df["label"] == 1).sum()
        neg_count = (df["label"] == 0).sum()
        
        logger.info(f"  {manifest_file.name:45} | {len(df):5} pairs | pos: {pos_count:4} neg: {neg_count:4}")
        
        # Add source identifier
        df["source"] = manifest_file.stem
        all_dataframes.append(df)
        source_summaries.append({
            "source": manifest_file.stem,
            "rows": len(df),
            "pos": pos_count,
            "neg": neg_count,
        })
    
    # Source 2: FGFR family data
    logger.info("\n📁 SOURCE 2: data/real_training_*/manifests/train_manifest_sampled.csv")
    logger.info("-" * 70)
    
    data_dir = Path("data")
    for family_dir in sorted(data_dir.glob("real_training_*")):
        manifests_dir = family_dir / "manifests"
        if not manifests_dir.exists():
            continue
        
        family_name = family_dir.name.replace("real_training_", "")
        manifest_file = manifests_dir / "train_manifest_sampled.csv"
        
        if not manifest_file.exists():
            continue
        
        df = load_and_filter_manifest(manifest_file)
        if len(df) == 0:
            continue
        
        pos_count = (df["label"] == 1).sum()
        neg_count = (df["label"] == 0).sum()
        
        logger.info(f"  {family_name:45} | {len(df):5} pairs | pos: {pos_count:4} neg: {neg_count:4}")
        
        # Add source identifier
        df["source"] = f"fgfr_{family_name}"
        all_dataframes.append(df)
        source_summaries.append({
            "source": f"fgfr_{family_name}",
            "rows": len(df),
            "pos": pos_count,
            "neg": neg_count,
        })
    
    if not all_dataframes:
        logger.error("No training data found!")
        return pd.DataFrame()
    
    # Combine all
    combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Fix paths to use forward slashes
    combined["receptor_file"] = combined["receptor_file"].str.replace("\\", "/")
    combined["ligand_file"] = combined["ligand_file"].str.replace("\\", "/")
    
    # Remove duplicates by (receptor_file, ligand_file, label) tuple
    logger.info(f"\nTotal rows before deduplication: {len(combined)}")
    combined_dedup = combined.drop_duplicates(
        subset=["receptor_file", "ligand_file", "label"],
        keep="first"
    )
    logger.info(f"Total rows after deduplication: {len(combined_dedup)}")
    
    # Fix pair_id to be unique
    combined_dedup["pair_id"] = [f"PAIR_{i:06d}" for i in range(len(combined_dedup))]
    
    # Output summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY BY SOURCE")
    logger.info("="*70)
    
    totals = {"rows": 0, "pos": 0, "neg": 0}
    for summary in source_summaries:
        logger.info(f"  {summary['source']:45} | {summary['rows']:5} pairs | pos: {summary['pos']:4} neg: {summary['neg']:4}")
        totals["rows"] += summary["rows"]
        totals["pos"] += summary["pos"]
        totals["neg"] += summary["neg"]
    
    logger.info("="*70)
    logger.info(f"COMBINED DATASET STATISTICS")
    logger.info("="*70)
    
    pos_final = (combined_dedup["label"] == 1).sum()
    neg_final = (combined_dedup["label"] == 0).sum()
    
    logger.info(f"Total rows (with dedup):  {len(combined_dedup)}")
    logger.info(f"Positive (binders):       {pos_final} ({100*pos_final/len(combined_dedup):.1f}%)")
    logger.info(f"Negative (non-binders):   {neg_final} ({100*neg_final/len(combined_dedup):.1f}%)")
    logger.info(f"Balance ratio:            {pos_final/neg_final:.3f}")
    logger.info(f"Unique sources:           {len(set(combined_dedup['source']))}")
    logger.info("="*70 + "\n")
    
    return combined_dedup


def main():
    """Combine all datasets and save."""
    combined = combine_all_datasets()
    
    if len(combined) == 0:
        logger.error("Failed to combine datasets!")
        return 1
    
    # Save combined manifest
    output_path = Path("master_data/final_training_manifest_all_sources.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select columns for output
    output_cols = ["pair_id", "receptor_file", "ligand_file", "label"]
    if "chain_id" in combined.columns:
        output_cols.append("chain_id")
    output_cols.append("source")
    
    combined[output_cols].to_csv(output_path, index=False)
    logger.info(f"✅ Combined manifest saved to: {output_path}")
    logger.info(f"   Total pairs: {len(combined)}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
