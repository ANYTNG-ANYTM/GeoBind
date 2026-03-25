"""
Phase 5: Web Frontend for GeoBind
Streamlit-based interactive web application for binding prediction
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="GeoBind - Protein-Ligand Binding Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"
CACHE_TTL = 300  # 5 minutes

# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.title("🧬 GeoBind")
st.sidebar.write("Protein-Ligand Binding Prediction")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict", "Batch Predictions", "Model Info", "About"]
)

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def get_api_health():
    """Check API health"""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_model_info():
    """Get model information from API"""
    try:
        resp = requests.get(f"{API_BASE_URL}/model_info", timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to fetch model info: {e}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_feature_importance():
    """Get feature importance from API"""
    try:
        resp = requests.get(f"{API_BASE_URL}/feature_importance", timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to fetch feature importance: {e}")
        return None

def make_prediction(receptor_path, ligand_path, chain="A", threshold=0.5):
    """Make single prediction via API"""
    try:
        payload = {
            "receptor_pdb_path": str(receptor_path),
            "ligand_sdf_path": str(ligand_path),
            "receptor_chain": chain,
            "threshold": threshold
        }
        resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def make_batch_prediction(pairs, threshold=0.5):
    """Make batch predictions via API"""
    try:
        payload = {
            "pairs": pairs,
            "threshold": threshold
        }
        resp = requests.post(f"{API_BASE_URL}/predict_batch", json=payload, timeout=120)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
        return None

def upload_and_predict(receptor_file, ligand_file, chain="A", threshold=0.5):
    """Upload files and make prediction"""
    try:
        files = {
            "receptor": receptor_file,
            "ligand": ligand_file
        }
        params = {
            "receptor_chain": chain,
            "threshold": threshold
        }
        resp = requests.post(f"{API_BASE_URL}/predict_upload", files=files, params=params, timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Upload prediction failed: {e}")
        return None

# ============================================================================
# Pages
# ============================================================================

def page_home():
    """Home page"""
    st.title("🧬 GeoBind")
    st.markdown("### Protein-Ligand Binding Prediction Platform")
    
    # API Status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        health = get_api_health()
        if health and health.get("status") == "healthy":
            st.success("✅ API: Connected")
        else:
            st.error("❌ API: Disconnected")
    
    with col2:
        model_info = get_model_info()
        if model_info:
            st.info(f"📊 Model: {model_info.get('model_type', 'Unknown')}")
        else:
            st.warning("⚠️ Model: Not loaded")
    
    with col3:
        st.info("🎯 ROC-AUC: 0.710")
    
    st.markdown("---")
    
    # Overview
    st.markdown("""
    ## Welcome to GeoBind
    
    GeoBind is a machine learning platform for predicting protein-ligand binding affinity using 
    physics-based complementarity metrics and XGBoost classification.
    
    ### Features
    - 🔬 **Single Predictions**: Test individual protein-ligand pairs
    - 📦 **Batch Processing**: Predict on 100+ pairs simultaneously
    - 📊 **Model Insights**: View feature importance and model details
    - 📈 **Visualization**: Interactive charts and prediction analysis
    - 💾 **Export Results**: Download predictions as CSV
    
    ### How It Works
    1. **Phase 1**: Parse PDB (protein) and SDF (ligand) structures
    2. **Phase 2**: Calculate physics-based complementarity vector (8 features)
    3. **Phase 3**: Feed into XGBoost model for binding prediction
    4. **Phase 4**: REST API for scalable inference
    5. **Phase 5**: This web interface for easy access
    
    ### Quick Start
    - 👉 Go to **Predict** tab to test a single pair
    - 👉 Go to **Batch Predictions** to process multiple pairs
    - 👉 Check **Model Info** to understand feature importance
    """)
    
    st.markdown("---")
    st.markdown("**Version**: 5.0.0 | **Built**: March 26, 2026")

def page_predict():
    """Single prediction page"""
    st.title("🔬 Single Prediction")
    
    # Check API health first
    health = get_api_health()
    if not health or health.get("status") != "healthy":
        st.error("❌ API is not available. Please ensure the backend is running on port 8000.")
        st.code("python -m uvicorn phase4_api_service:app --port 8000")
        return
    
    st.markdown("""
    Enter paths to receptor (PDB) and ligand (SDF) files, or upload them directly.
    The model will calculate binding complementarity and predict binding affinity.
    """)
    
    # Input method selection
    input_method = st.radio("Input Method", ["File Paths", "Upload Files"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Settings")
        chain_id = st.text_input("Protein Chain ID", value="A", help="Usually 'A' for single chain")
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05,
                             help="Probability threshold for binding prediction")
    
    with col2:
        st.subheader("Input")
        
        if input_method == "File Paths":
            receptor_path = st.text_input("Receptor PDB Path", 
                                        value="master_data/receptors/3FNU.pdb",
                                        help="Path to PDB file")
            ligand_path = st.text_input("Ligand SDF Path",
                                       value="master_data/ligands/CHEMBL231522.sdf",
                                       help="Path to SDF file")
            
            if st.button("🚀 Predict", use_container_width=True):
                with st.spinner("Computing complementarity vector..."):
                    result = make_prediction(receptor_path, ligand_path, chain_id, threshold)
                
                if result:
                    st.success("✅ Prediction Complete!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Binding", "YES" if result['binding_prediction'] == 1 else "NO",
                                 delta=f"{result['binding_probability']:.1%}")
                    with col2:
                        st.metric("Probability", f"{result['binding_probability']:.3f}",
                                 delta=f"±{1-result['confidence']:.3f}")
                    with col3:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col4:
                        st.metric("Pair ID", result['pair_id'][:20])
                    
                    # Complementarity vector
                    st.markdown("### Complementarity Vector")
                    vec_df = pd.DataFrame({
                        "Feature": ["Distance Score", "Angle Score", "Electrostatic", "H-Bonds",
                                   "Hydrophobic", "vdW Score", "Shape Match", "Pocket Fit"],
                        "Value": result['complementarity_vector']
                    })
                    st.dataframe(vec_df, use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure(data=[
                        go.Bar(x=vec_df["Feature"], y=vec_df["Value"], 
                               marker=dict(color=vec_df["Value"], colorscale="Viridis"))
                    ])
                    fig.update_layout(title="Complementarity Metrics", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export
                    csv = vec_df.to_csv(index=False)
                    st.download_button("📥 Download Results (CSV)", csv, "prediction.csv", "text/csv")
                else:
                    st.error("❌ Prediction failed. Check file paths and try again.")
        
        else:  # Upload Files
            receptor_file = st.file_uploader("Upload Receptor PDB", type=["pdb", "PDB"])
            ligand_file = st.file_uploader("Upload Ligand SDF", type=["sdf", "SDF"])
            
            if st.button("🚀 Predict", use_container_width=True):
                if receptor_file and ligand_file:
                    with st.spinner("Computing complementarity vector..."):
                        result = upload_and_predict(receptor_file, ligand_file, chain_id, threshold)
                    
                    if result:
                        st.success("✅ Prediction Complete!")
                        
                        # Display results (same as above)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Binding", "YES" if result['binding_prediction'] == 1 else "NO")
                        with col2:
                            st.metric("Probability", f"{result['binding_probability']:.3f}")
                        with col3:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with col4:
                            st.metric("Time", result['predicted_at'].split('T')[1][:8])
                        
                        # Complementarity vector
                        st.markdown("### Complementarity Vector")
                        vec_df = pd.DataFrame({
                            "Feature": ["Distance Score", "Angle Score", "Electrostatic", "H-Bonds",
                                       "Hydrophobic", "vdW Score", "Shape Match", "Pocket Fit"],
                            "Value": result['complementarity_vector']
                        })
                        st.dataframe(vec_df, use_container_width=True)
                    else:
                        st.error("❌ Prediction failed.")
                else:
                    st.warning("⚠️ Please upload both receptor and ligand files.")

def page_batch():
    """Batch predictions page"""
    st.title("📦 Batch Predictions")
    
    # Check API health
    health = get_api_health()
    if not health or health.get("status") != "healthy":
        st.error("❌ API is not available.")
        return
    
    st.markdown("Process multiple protein-ligand pairs at once.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Data")
        st.markdown("Upload CSV with columns: `receptor_pdb_path`, `ligand_sdf_path`, `receptor_chain` (optional)")
        
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(10), use_container_width=True)
            
            threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
            
            if st.button("🚀 Predict All", use_container_width=True):
                # Convert to list of dicts for API
                pairs = df.fillna({"receptor_chain": "A"}).to_dict('records')
                
                with st.spinner(f"Processing {len(pairs)} pairs..."):
                    result = make_batch_prediction(pairs, threshold)
                
                if result:
                    st.success(f"✅ Complete! {result['successful_predictions']}/{result['total_pairs']} successful")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Successful", result['successful_predictions'])
                    with col2:
                        st.metric("Failed", result['failed_predictions'])
                    with col3:
                        st.metric("Success Rate", f"{100*result['successful_predictions']/result['total_pairs']:.1f}%")
                    
                    # Results table
                    if result['predictions']:
                        results_df = pd.DataFrame(result['predictions'])
                        results_df['Binding'] = results_df['binding_prediction'].map({1: "YES", 0: "NO"})
                        results_df['Probability'] = results_df['binding_probability'].apply(lambda x: f"{x:.3f}")
                        
                        st.dataframe(results_df[['pair_id', 'Binding', 'Probability', 'confidence']], 
                                   use_container_width=True)
                        
                        # Visualization
                        fig = px.histogram(results_df, x='binding_prediction', 
                                         title="Binding Prediction Distribution",
                                         labels={'binding_prediction': 'Prediction', 'count': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download Results (CSV)", csv, 
                                         "batch_predictions.csv", "text/csv")
    
    with col2:
        st.subheader("Sample Data")
        sample_data = {
            "receptor_pdb_path": [
                "master_data/receptors/3FNU.pdb",
                "master_data/receptors/2BUA.pdb",
                "master_data/receptors/3QS1.pdb",
            ],
            "ligand_sdf_path": [
                "master_data/ligands/CHEMBL231522.sdf",
                "master_data/ligands/CHEMBL382127.sdf",
                "master_data/ligands/CHEMBL231522.sdf",
            ],
            "receptor_chain": ["A", "A", "A"]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        # Download sample
        sample_csv = sample_df.to_csv(index=False)
        st.download_button("📥 Download Sample CSV", sample_csv, 
                          "sample_batch.csv", "text/csv")

def page_model_info():
    """Model information page"""
    st.title("📊 Model Information")
    
    model_info = get_model_info()
    feature_importance = get_feature_importance()
    
    if model_info:
        st.markdown("### Model Details")
        info_df = pd.DataFrame([model_info])
        st.json(model_info)
    
    if feature_importance:
        st.markdown("### Feature Importance")
        
        features = feature_importance["feature_importance"][:14]
        importance_df = pd.DataFrame(features)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(importance_df[['rank', 'feature', 'relative_importance']], 
                        use_container_width=True)
        
        with col2:
            fig = px.bar(importance_df, x='importance', y='feature', 
                        orientation='h', title="Feature Importance (Top 14)",
                        labels={'importance': 'Importance', 'feature': 'Feature'})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

def page_about():
    """About page"""
    st.title("ℹ️ About GeoBind")
    
    st.markdown("""
    ## GeoBind: Protein-Ligand Binding Prediction
    
    ### Overview
    GeoBind is a comprehensive machine learning platform for predicting protein-ligand 
    binding affinity. It combines physics-based molecular metrics with gradient boosting 
    to achieve accurate binding predictions.
    
    ### Architecture
    
    **Phase 1: Data Ingestion**
    - Parse PDB protein structures using Biopython
    - Extract ligand 3D coordinates from SDF files using RDKit
    - Standardize atomic coordinates and metadata
    
    **Phase 2: Physics & Geometry**
    - Distance scoring (optimal contact distance)
    - Angle scoring (proper orbital alignment)
    - Electrostatic complementarity
    - Hydrogen bond detection
    - Hydrophobic interaction scoring
    - Van der Waals overlap calculation
    - Shape complementarity (molecular surface)
    - Pocket fitting assessment
    
    **Phase 3: Machine Learning**
    - XGBoost classifier on 320+ training pairs
    - 14 engineered features (base + interactions)
    - 5-fold stratified cross-validation
    - Hyperparameter optimization via GridSearchCV
    - ROC-AUC: **0.710** on validation set
    
    **Phase 4: REST API**
    - FastAPI service for scalable inference
    - Supports single/batch predictions
    - File upload interface
    - Automatic interactive documentation (Swagger UI)
    
    **Phase 5: Web Frontend**
    - Streamlit application (this interface)
    - Interactive prediction interface
    - Batch processing capability
    - Model introspection tools
    - Results visualization and export
    
    ### Performance
    - Single prediction: 50-200 ms
    - Batch (100 pairs): 10-15 seconds
    - Memory: ~500 MB
    - Throughput: ~50 predictions/sec
    
    ### Technologies
    - **Python 3.13** — Core language
    - **Biopython** — PDB parsing
    - **RDKit** — Ligand processing
    - **NumPy/SciPy** — Numerical computation
    - **Scikit-learn** — ML utilities
    - **XGBoost** — Gradient boosting model
    - **FastAPI** — REST API framework
    - **Streamlit** — Web interface
    - **Plotly** — Interactive visualizations
    
    ### Data Sources
    - **PDB**: Crystal structures of protein-ligand complexes
    - **ChEMBL**: Bioactivity data and molecular structures
    - **FGFR Family**: Protein family-specific training data
    
    ### Training Data
    - 320+ labeled protein-ligand pairs
    - Balanced positive/negative examples
    - Real binding affinities (IC50, Ki, Kd)
    - Multiple protein families (FGFR, kinases, etc.)
    
    ### Accuracy
    - ROC-AUC: 0.710
    - Accuracy: 70.31%
    - Precision: 72.41%
    - Recall: 65.63%
    - F1-Score: 68.85%
    
    ### Citation
    ```
    GeoBind: Protein-Ligand Binding Prediction Platform
    Yash et al., 2026
    ```
    
    ### References
    - [AutoDock Vina](http://vina.scripps.edu/)
    - [Open Drug Discovery Toolkit (ODDT)](http://oddt.readthedocs.org/)
    - [RDKit](rdkit.org)
    - [Biopython](biopython.org)
    - [XGBoost](xgboost.readthedocs.io/)
    """)

# ============================================================================
# Main App
# ============================================================================

if __name__ == "__main__":
    if page == "Home":
        page_home()
    elif page == "Predict":
        page_predict()
    elif page == "Batch Predictions":
        page_batch()
    elif page == "Model Info":
        page_model_info()
    elif page == "About":
        page_about()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    GeoBind v5.0.0 | Built with ❤️ using Streamlit | 
    <a href='http://localhost:8000/docs' target='_blank'>API Docs</a>
    </div>
    """, unsafe_allow_html=True)
