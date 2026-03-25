#!/usr/bin/env python3
"""Test the prediction endpoint."""

import requests
import json

url = "http://localhost:8000/predict"
data = {
    "receptor_pdb_path": "master_data/receptors/3FNU.pdb",
    "ligand_sdf_path": "master_data/ligands/CHEMBL231522.sdf"
}

print(f"Testing prediction endpoint: {url}")
print(f"Request data: {json.dumps(data, indent=2)}")
print()

try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCESS!")
        print(f"Pair ID: {result['pair_id']}")
        print(f"Binding Probability: {result['binding_probability']:.4f}")
        print(f"Binding Prediction: {result['binding_prediction']} (1=binder, 0=non-binder)")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Vector (8 features): {[f'{x:.6f}' for x in result['complementarity_vector'][:3]]}...")
        print(f"Engineered Features (6 features): {[f'{x:.6f}' for x in result['engineered_features'][:3]]}...")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
