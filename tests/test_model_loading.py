#!/usr/bin/env python3
"""Test if model can be loaded with TrainingArtifacts class available."""

from phase3_ml_pipeline_optimized import TrainingArtifacts
import joblib

print("Testing model loading...")
try:
    artifacts = joblib.load("../models/master_optimized/geobind_xgb_optimized.pkl")
    print('✅ Model loaded successfully!')
    print(f'   Type: {type(artifacts).__name__}')
    print(f'   Has model: {hasattr(artifacts, "model")}')
    print(f'   Has scaler: {hasattr(artifacts, "scaler")}')
    print(f'   Has feature_names: {hasattr(artifacts, "feature_names")}')
    print(f'   Has metadata: {hasattr(artifacts, "metadata")}')
    
    if hasattr(artifacts, 'model'):
        print(f'\n   Model type: {type(artifacts.model).__name__}')
    if hasattr(artifacts, 'scaler'):
        print(f'   Scaler type: {type(artifacts.scaler).__name__}')
        
except Exception as e:
    print(f'❌ Error loading model: {e}')
    import traceback
    traceback.print_exc()
