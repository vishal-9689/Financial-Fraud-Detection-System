import streamlit as st
import joblib
import pandas as pd

# Set page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Load pipeline that includes preprocessing + model
try:
    pipeline = joblib.load("models/random_forest_pipeline.pkl")  # Make sure you saved the pipeline with this name
except Exception as e:
    st.error(f"❌ Failed to load model pipeline: {str(e)}")
    st.stop()

st.title("💳 Financial Fraud Detection System")
st.markdown("Enter transaction details to assess fraud risk using a trained ML model.")

# Default values based on real-world examples
default_values = {
    'V1': -1.23, 'V2': 0.45, 'V3': 1.12, 'V4': 0.88, 'V5': -0.56,
    'V6': 0.12, 'V7': -0.21, 'V8': 0.33, 'V9': -0.44, 'V10': 0.55,
    'V11': -0.66, 'V12': 0.77, 'V13': -0.88, 'V14': 0.99, 'V15': -0.11,
    'V16': 0.22, 'V17': -0.33, 'V18': 0.44, 'V19': -0.55, 'V20': 0.66,
    'V21': -0.77, 'V22': 0.88, 'V23': -0.99, 'V24': 0.11, 'V25': -0.22,
    'V26': 0.33, 'V27': -0.44, 'V28': 0.55
}

# Input layout
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)

with col2:
    features = {}
    for i in range(1, 29):
        feature_name = f"V{i}"
        features[feature_name] = st.number_input(
            f"{feature_name} (PCA Feature {i})",
            value=default_values.get(feature_name, 0.0),
            step=0.01,
            key=f"v_{i}"
        )

# Predict button
if st.button("🚦 Check Fraud Risk"):
    try:
        # Prepare input DataFrame
        input_dict = {f"V{i}": features[f"V{i}"] for i in range(1, 29)}
        input_dict["Amount"] = amount
        input_df = pd.DataFrame([input_dict])

        # Predict with pipeline
        proba = pipeline.predict_proba(input_df)[0][1]

        # Risk interpretation
        if proba > 0.2:
            st.error(f"🚨 High Fraud Risk: {proba:.2%}")
        elif proba > 0.1:
            st.warning(f"⚠️ Moderate Fraud Risk: {proba:.2%}")
        else:
            st.success(f"✅ Low Fraud Risk: {proba:.2%}")

        with st.expander("ℹ️ How to interpret this"):
            st.markdown("""
            - **<30%**: Low risk — Likely legitimate  
            - **30–70%**: Moderate risk — Manual review advised  
            - **>70%**: High risk — Strong fraud suspicion  
            """)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("📊 About the Model")
    st.markdown("""
    This app detects credit card fraud using a trained Random Forest model  
    integrated with a preprocessing pipeline.

    **Features used:**
    - 28 anonymized PCA features  
    - Transaction amount

    **Model Info:**
    - Classifier: Random Forest  
    - Accuracy: ~99% on test set  
    - Balanced for class imbalance  
    """)
