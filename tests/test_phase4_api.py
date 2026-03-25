"""
Phase 4 API Integration Tests
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(json.dumps(data, indent=2))
    return resp.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info Endpoint ===")
    resp = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {resp.text}")
    return resp.status_code == 200

def test_feature_importance():
    """Test feature importance endpoint"""
    print("\n=== Testing Feature Importance Endpoint ===")
    resp = requests.get(f"{BASE_URL}/feature_importance")
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Top 5 Features:")
        for feat in data["feature_importance"][:5]:
            print(f"  {feat['rank']}. {feat['feature']:30s} {feat['importance']:.4f}")
    else:
        print(f"Error: {resp.text}")
    return resp.status_code == 200

def test_predict_single():
    """Test single prediction"""
    print("\n=== Testing Single Prediction ===")
    
    payload = {
        "receptor_pdb_path": "master_data/receptors/3FNU.pdb",
        "ligand_sdf_path": "master_data/ligands/CHEMBL231522.sdf",
        "receptor_chain": "A",
        "threshold": 0.5
    }
    
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Pair ID: {data['pair_id']}")
        print(f"Binding: {'YES' if data['binding_prediction'] == 1 else 'NO'}")
        print(f"Probability: {data['binding_probability']:.4f}")
        print(f"Confidence: {data['confidence']:.4f}")
    else:
        print(f"Error: {resp.text}")
    return resp.status_code == 200

def test_api_docs():
    """Test API documentation endpoints"""
    print("\n=== Testing API Documentation ===")
    
    # Swagger UI
    resp_swagger = requests.get(f"{BASE_URL}/docs")
    print(f"Swagger UI: {resp_swagger.status_code}")
    
    # ReDoc
    resp_redoc = requests.get(f"{BASE_URL}/redoc")
    print(f"ReDoc: {resp_redoc.status_code}")
    
    # OpenAPI Schema
    resp_openapi = requests.get(f"{BASE_URL}/openapi.json")
    print(f"OpenAPI Schema: {resp_openapi.status_code}")
    
    return all(r.status_code == 200 for r in [resp_swagger, resp_redoc, resp_openapi])

def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 4 API Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Feature Importance", test_feature_importance),
        ("Single Prediction", test_predict_single),
        ("API Documentation", test_api_docs),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API at http://localhost:8000")
        print("   Make sure the API is running: python -m uvicorn phase4_api_service:app --port 8000")
        sys.exit(1)
