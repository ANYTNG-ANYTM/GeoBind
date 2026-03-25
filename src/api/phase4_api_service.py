"""
Phase 4: FastAPI Service for GeoBind
Exposes the trained XGBoost binding prediction model via REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import os
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json

# Import GeoBind pipeline modules
from src.core.phase1_data_ingestion import ReceptorParser, LigandParser
from src.core.phase2_physics_geometry import ComplementarityVectorGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GeoBind Binding Prediction API",
    description="Predicts protein-ligand binding affinity using complementarity metrics and XGBoost",
    version="4.0.0"
)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    receptor_pdb_path: str = Field(..., description="Path to receptor PDB file")
    ligand_sdf_path: str = Field(..., description="Path to ligand SDF file")
    receptor_chain: str = Field(default="A", description="Protein chain ID")
    threshold: float = Field(default=0.5, description="Classification threshold (0-1)")
    
class PredictionResponse(BaseModel):
    """Response model for prediction"""
    pair_id: str
    binding_probability: float
    binding_prediction: int = Field(description="0=non-binder, 1=binder")
    confidence: float
    complementarity_vector: List[float] = Field(description="8 complementarity metrics")
    engineered_features: List[float] = Field(description="14 engineered features")
    predicted_at: str = Field(description="ISO timestamp")
    
class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    pairs: List[Dict[str, str]] = Field(..., description="List of {receptor_pdb_path, ligand_sdf_path, receptor_chain}")
    threshold: float = Field(default=0.5)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str

# ============================================================================
# Global State
# ============================================================================

MODEL_PATH = "models/master_optimized/geobind_xgb_optimized.pkl"
SCALER_KEY = "scaler"
MODEL_KEY = "model"

# Model cache
_model_cache = {MODEL_KEY: None, SCALER_KEY: None}

def load_model():
    """Load model and scaler from disk"""
    global _model_cache
    
    if _model_cache[MODEL_KEY] is not None:
        return _model_cache[MODEL_KEY], _model_cache[SCALER_KEY]
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    try:
        artifacts = joblib.load(MODEL_PATH)
        # Handle both dictionary format (legacy) and object format (current)
        if isinstance(artifacts, dict):
            model = artifacts[MODEL_KEY]
            scaler = artifacts[SCALER_KEY]
        else:
            # Assume it's a TrainingArtifacts object
            model = artifacts.model
            scaler = artifacts.scaler
        _model_cache[MODEL_KEY] = model
        _model_cache[SCALER_KEY] = scaler
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model, scaler
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def compute_complementarity_vector(receptor_path: str, ligand_path: str, chain: str = "A") -> tuple:
    """
    Compute complementarity vector for receptor-ligand pair
    
    Returns:
        Tuple of (raw_vector, full_feature_vector_with_engineering)
    """
    try:
        # Parse structures
        receptor, _ = ReceptorParser.parse_pdb(receptor_path, chain_id=chain)
        ligand, _ = LigandParser.parse_sdf(ligand_path)
        
        # Compute complementarity
        calc = ComplementarityVectorGenerator()
        vector, _ = calc.calculate_complementarity_vector(receptor, ligand)
        
        # Feature engineering: Create 6 additional engineered features from the 8 base features
        # Base feature order: [dist_score, angle_score, electrostatic_energy, hbond_count, 
        #                      hydrophobic_score, vdW_score, shape_match, pocket_fit]
        dist_score, angle_score, electrostatic_energy, hbond_count, \
            hydrophobic_score, vdW_score, shape_match, pocket_fit = vector
        
        # Create engineered features (6 features)
        engineered_features = np.array([
            shape_match * hydrophobic_score,           # shape_match_x_hydrophobic
            dist_score * angle_score,                  # dist_score_x_angle_score
            vdW_score * electrostatic_energy,          # vdW_x_electrostatic
            hbond_count * pocket_fit,                  # hbond_x_pocket_fit
            shape_match ** 2,                          # shape_match_squared
            hydrophobic_score ** 2,                    # hydrophobic_squared
        ])
        
        # Combine base and engineered features (14 total)
        full_features = np.concatenate([vector, engineered_features])
        
        return vector, full_features
    except Exception as e:
        logger.error(f"Error computing complementarity: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to compute features: {str(e)}")

def predict_pair(receptor_path: str, ligand_path: str, receptor_chain: str = "A", threshold: float = 0.5) -> Dict[str, Any]:
    """Make prediction for a single pair"""
    
    # Load model
    model, scaler = load_model()
    
    # Compute features (returns raw 8-feature and full 14-feature vectors)
    raw_vector, full_features = compute_complementarity_vector(receptor_path, ligand_path, receptor_chain)
    
    # Normalize with proper shape (1, 14)
    X = np.array([full_features])
    X_normalized = scaler.transform(X)
    
    # Predict
    prob = model.predict_proba(X_normalized)[0, 1]  # Probability of class 1 (binder)
    pred = 1 if prob >= threshold else 0
    confidence = max(prob, 1 - prob)
    
    pair_id = f"{Path(receptor_path).stem}_{Path(ligand_path).stem}"
    
    # Extract engineered features (last 6 of the 14 features)
    engineered_only = full_features[8:14]
    
    return {
        "pair_id": pair_id,
        "binding_probability": float(prob),
        "binding_prediction": int(pred),
        "confidence": float(confidence),
        "complementarity_vector": [float(x) for x in raw_vector],
        "engineered_features": [float(x) for x in engineered_only],
        "predicted_at": datetime.utcnow().isoformat()
    }

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = False
    try:
        load_model()
        model_loaded = True
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict binding for a single protein-ligand pair"""
    logger.info(f"Prediction request: {request.receptor_pdb_path} + {request.ligand_sdf_path}")
    
    # Validate paths exist
    if not os.path.exists(request.receptor_pdb_path):
        raise HTTPException(status_code=404, detail=f"Receptor file not found: {request.receptor_pdb_path}")
    if not os.path.exists(request.ligand_sdf_path):
        raise HTTPException(status_code=404, detail=f"Ligand file not found: {request.ligand_sdf_path}")
    
    # Make prediction
    result = predict_pair(
        request.receptor_pdb_path,
        request.ligand_sdf_path,
        request.receptor_chain,
        request.threshold
    )
    
    return PredictionResponse(**result)

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict binding for multiple protein-ligand pairs"""
    logger.info(f"Batch prediction request with {len(request.pairs)} pairs")
    
    results = []
    failed = []
    
    for i, pair in enumerate(request.pairs):
        try:
            result = predict_pair(
                pair.get("receptor_pdb_path"),
                pair.get("ligand_sdf_path"),
                pair.get("receptor_chain", "A"),
                request.threshold
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Pair {i} failed: {e}")
            failed.append({
                "pair_index": i,
                "error": str(e)
            })
    
    return JSONResponse({
        "total_pairs": len(request.pairs),
        "successful_predictions": len(results),
        "failed_predictions": len(failed),
        "predictions": results,
        "failures": failed,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.post("/predict_upload")
async def predict_upload(
    receptor: UploadFile = File(...),
    ligand: UploadFile = File(...),
    receptor_chain: str = "A",
    threshold: float = 0.5
):
    """Predict from uploaded PDB and SDF files"""
    logger.info(f"Upload prediction: {receptor.filename} + {ligand.filename}")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploads
        receptor_path = os.path.join(temp_dir, receptor.filename)
        ligand_path = os.path.join(temp_dir, ligand.filename)
        
        with open(receptor_path, "wb") as f:
            f.write(await receptor.read())
        with open(ligand_path, "wb") as f:
            f.write(await ligand.read())
        
        # Make prediction
        result = predict_pair(receptor_path, ligand_path, receptor_chain, threshold)
        
        return PredictionResponse(**result)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/model_info")
async def model_info():
    """Get information about loaded model"""
    try:
        model, scaler = load_model()
        
        return JSONResponse({
            "model_type": str(type(model).__name__),
            "model_path": MODEL_PATH,
            "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown",
            "feature_names": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else [],
            "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else "unknown",
            "max_depth": model.max_depth if hasattr(model, 'max_depth') else "unknown",
            "scaler_type": str(type(scaler).__name__),
            "loaded_at": datetime.utcnow().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/feature_importance")
async def feature_importance():
    """Get feature importance from model"""
    try:
        model, _ = load_model()
        
        if not hasattr(model, 'feature_importances_'):
            raise HTTPException(status_code=400, detail="Model does not have feature importances")
        
        importance = model.feature_importances_
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importance))]
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        
        return JSONResponse({
            "feature_importance": [
                {
                    "rank": int(i + 1),
                    "feature": str(feature_names[idx]),
                    "importance": float(importance[idx]),
                    "relative_importance": float(importance[idx] / importance.sum())
                }
                for i, idx in enumerate(sorted_indices)
            ],
            "total_sum": float(importance.sum())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """API documentation root"""
    return JSONResponse({
        "name": "GeoBind Binding Prediction API",
        "version": "4.0.0",
        "description": "Predicts protein-ligand binding affinity using physics-based metrics",
        "endpoints": {
            "GET /health": "Health check",
            "GET /model_info": "Get model information",
            "GET /feature_importance": "Get feature importance",
            "POST /predict": "Single prediction",
            "POST /predict_batch": "Batch predictions",
            "POST /predict_upload": "Prediction with file upload",
            "GET /": "This help"
        },
        "docs": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    })

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
