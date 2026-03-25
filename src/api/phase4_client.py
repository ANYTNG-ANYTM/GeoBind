"""
Phase 4: FastAPI Client Example
Test client for GeoBind API service
"""

import requests
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time

class GeobindClient:
    """Client for GeoBind API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
    
    def predict(self, receptor_pdb: str, ligand_sdf: str, chain: str = "A", threshold: float = 0.5) -> Dict:
        """Predict binding for a single pair"""
        payload = {
            "receptor_pdb_path": receptor_pdb,
            "ligand_sdf_path": ligand_sdf,
            "receptor_chain": chain,
            "threshold": threshold
        }
        resp = self.session.post(f"{self.base_url}/predict", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def predict_batch(self, pairs: List[Dict], threshold: float = 0.5) -> Dict:
        """Predict binding for multiple pairs"""
        payload = {
            "pairs": pairs,
            "threshold": threshold
        }
        resp = self.session.post(f"{self.base_url}/predict_batch", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def predict_upload(self, receptor_pdb: str, ligand_sdf: str, chain: str = "A", threshold: float = 0.5) -> Dict:
        """Predict with file upload"""
        with open(receptor_pdb, "rb") as f_rec, open(ligand_sdf, "rb") as f_lig:
            files = {
                "receptor": f_rec,
                "ligand": f_lig
            }
            params = {
                "receptor_chain": chain,
                "threshold": threshold
            }
            resp = self.session.post(f"{self.base_url}/predict_upload", files=files, params=params)
        resp.raise_for_status()
        return resp.json()
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        resp = self.session.get(f"{self.base_url}/model_info")
        resp.raise_for_status()
        return resp.json()
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance"""
        resp = self.session.get(f"{self.base_url}/feature_importance")
        resp.raise_for_status()
        return resp.json()

def main():
    parser = argparse.ArgumentParser(description="GeoBind API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--info", action="store_true", help="Get model info")
    parser.add_argument("--features", action="store_true", help="Get feature importance")
    parser.add_argument("--predict", nargs=2, metavar=("PDB", "SDF"), help="Predict single pair")
    parser.add_argument("--batch", type=str, help="Batch prediction from JSON file")
    parser.add_argument("--upload", nargs=2, metavar=("PDB", "SDF"), help="Predict with file upload")
    parser.add_argument("--chain", default="A", help="Protein chain ID")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    client = GeobindClient(args.url)
    
    try:
        # Health check
        if args.health:
            print("🔍 Checking API health...")
            health = client.health_check()
            print(json.dumps(health, indent=2))
            print()
        
        # Model info
        if args.info:
            print("📊 Model Information:")
            info = client.get_model_info()
            print(json.dumps(info, indent=2))
            print()
        
        # Feature importance
        if args.features:
            print("⭐ Feature Importance:")
            features = client.get_feature_importance()
            for feat in features["feature_importance"][:10]:
                print(f"  {feat['rank']:2d}. {feat['feature']:30s} {feat['importance']:.4f} ({feat['relative_importance']*100:5.2f}%)")
            print()
        
        # Single prediction
        if args.predict:
            print(f"🧬 Predicting: {args.predict[0]} + {args.predict[1]}")
            result = client.predict(args.predict[0], args.predict[1], args.chain, args.threshold)
            print(f"\n✅ {result['pair_id']}:")
            print(f"   Binding: {'YES' if result['binding_prediction'] == 1 else 'NO'}")
            print(f"   Probability: {result['binding_probability']:.4f}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print()
        
        # File upload prediction
        if args.upload:
            print(f"📤 Upload predicting: {args.upload[0]} + {args.upload[1]}")
            result = client.predict_upload(args.upload[0], args.upload[1], args.chain, args.threshold)
            print(f"\n✅ {result['pair_id']}:")
            print(f"   Binding: {'YES' if result['binding_prediction'] == 1 else 'NO'}")
            print(f"   Probability: {result['binding_probability']:.4f}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print()
        
        # Batch prediction
        if args.batch:
            print(f"📦 Batch predicting from {args.batch}...")
            with open(args.batch, "r") as f:
                pairs = json.load(f)
            
            result = client.predict_batch(pairs, args.threshold)
            print(f"\n📊 Results:")
            print(f"   Total pairs: {result['total_pairs']}")
            print(f"   Successful: {result['successful_predictions']}")
            print(f"   Failed: {result['failed_predictions']}")
            print(f"\n   Predictions:")
            for pred in result['predictions']:
                status = "✅ BINDER" if pred['binding_prediction'] == 1 else "❌ NON-BINDER"
                print(f"      {pred['pair_id']:40s} {status} ({pred['binding_probability']:.3f})")
            print()
        
        if not any([args.health, args.info, args.features, args.predict, args.upload, args.batch]):
            print("No action specified. Use --health, --info, --predict, etc.")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed. Is the API running at {args.url}?")
        print("   Start with: python -m uvicorn phase4_api_service:app --reload")
    except requests.exceptions.HTTPError as e:
        print(f"❌ API error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
