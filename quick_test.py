#!/usr/bin/env python3
"""Quick test of the API"""

import requests
import json

url = 'http://localhost:8000/predict'
data = {
    'receptor_pdb_path': 'master_data/receptors/3FNU.pdb',
    'ligand_sdf_path': 'master_data/ligands/CHEMBL231522.sdf'
}

print("Testing API prediction...")
resp = requests.post(url, json=data)
print(f'Status: {resp.status_code}')

if resp.status_code == 200:
    result = resp.json()
    print(f'✅ SUCCESS!')
    print(f'   Prediction: {result.get("binding_prediction")} (1=binder, 0=non-binder)')
    print(f'   Probability: {result.get("binding_probability"):.3f}')
else:
    print(f'❌ Error: {resp.text[:200]}')
