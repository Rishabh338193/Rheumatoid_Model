"""
Streamlit Web App for Rheumatoid Arthritis Prediction
Alternative frontend using Streamlit
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

# Page Configuration
st.set_page_config(
    page_title="RA Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = 'http://127.0.0.1:5001'

# Header
st.markdown('<h1 class="main-header">🏥 Rheumatoid Arthritis Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AutoImmune Disease Prediction Model</p>', unsafe_allow_html=True)

# Load model metrics
@st.cache_data
def load_model_metrics():
    try:
        response = requests.get(f'{API_BASE_URL}/metrics')
        if response.status_code == 200:
            return response.json()
    except:
        return None

metrics = load_model_metrics()

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    
    if metrics:
        st.metric("Model Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        st.metric("Recall", f"{metrics['recall']*100:.2f}%")
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        st.divider()
        st.caption("**Model Type:** Random Forest Classifier")
    else:
        st.error("⚠️ Cannot connect to API")
    
    st.divider()
    
    # Sample Data Button
    if st.button("📋 Load Sample Data"):
        st.session_state.load_sample = True

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Information")
    
    # Demographics
    with st.expander("👤 Demographics", expanded=True):
        if 'load_sample' in st.session_state and st.session_state.load_sample:
            age = st.number_input("Age", min_value=18, max_value=120, value=52, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"], index=1)
        else:
            age = st.number_input("Age", min_value=18, max_value=120, value=45, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Clinical Symptoms
    with st.expander("🩺 Clinical Symptoms", expanded=True):
        if 'load_sample' in st.session_state and st.session_state.load_sample:
            stiffness = st.number_input("Morning Stiffness Duration (minutes)", 
                                       min_value=0, max_value=300, value=90, step=5)
            joint_pain = st.slider("Joint Pain Score", 0, 10, 8)
            swollen_joints = st.number_input("Swollen Joint Count", 
                                            min_value=0, max_value=28, value=12, step=1)
            fatigue = st.slider("Fatigue Score", 0, 10, 9)
        else:
            stiffness = st.number_input("Morning Stiffness Duration (minutes)", 
                                       min_value=0, max_value=300, value=30, step=5)
            joint_pain = st.slider("Joint Pain Score", 0, 10, 5)
            swollen_joints = st.number_input("Swollen Joint Count", 
                                            min_value=0, max_value=28, value=0, step=1)
            fatigue = st.slider("Fatigue Score", 0, 10, 3)
    
    # Laboratory Tests
    with st.expander("🧪 Laboratory Tests", expanded=True):
        if 'load_sample' in st.session_state and st.session_state.load_sample:
            rf = st.number_input("Rheumatoid Factor (IU/ml)", 
                                min_value=0.0, max_value=300.0, value=95.5, step=0.1)
            anti_ccp = st.number_input("Anti-CCP (U/ml)", 
                                       min_value=0.0, max_value=500.0, value=150.3, step=0.1)
            esr = st.number_input("ESR (mm/hr)", 
                                 min_value=0.0, max_value=150.0, value=55.2, step=0.1)
            crp = st.number_input("CRP (mg/L)", 
                                 min_value=0.0, max_value=100.0, value=25.8, step=0.1)
        else:
            rf = st.number_input("Rheumatoid Factor (IU/ml)", 
                                min_value=0.0, max_value=300.0, value=10.0, step=0.1)
            anti_ccp = st.number_input("Anti-CCP (U/ml)", 
                                       min_value=0.0, max_value=500.0, value=5.0, step=0.1)
            esr = st.number_input("ESR (mm/hr)", 
                                 min_value=0.0, max_value=150.0, value=15.0, step=0.1)
            crp = st.number_input("CRP (mg/L)", 
                                 min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    
    # Medical History
    with st.expander("📋 Medical History", expanded=True):
        if 'load_sample' in st.session_state and st.session_state.load_sample:
            family_history = st.selectbox("Family History of RA", ["No", "Yes"], index=1)
            smoking = st.selectbox("Smoking Status", ["No", "Yes"], index=1)
        else:
            family_history = st.selectbox("Family History of RA", ["No", "Yes"])
            smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    
    # Reset sample data flag
    if 'load_sample' in st.session_state:
        del st.session_state.load_sample
    
    st.divider()
    
    # Predict Button
    predict_button = st.button("🔍 Predict RA Risk", type="primary")

# Results Column
with col2:
    st.header("Prediction Results")
    
    if predict_button:
        # Prepare data
        patient_data = {
            "Age": int(age),
            "Gender": gender,
            "Morning_Stiffness_Duration": int(stiffness),
            "Joint_Pain_Score": int(joint_pain),
            "Swollen_Joint_Count": int(swollen_joints),
            "Rheumatoid_Factor": float(rf),
            "Anti_CCP": float(anti_ccp),
            "ESR": float(esr),
            "CRP": float(crp),
            "Fatigue_Score": int(fatigue),
            "Family_History": family_history,
            "Smoking_Status": smoking
        }
        
        # Make prediction
        with st.spinner("Analyzing patient data..."):
            try:
                response = requests.post(
                    f'{API_BASE_URL}/predict',
                    json=patient_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Main Result
                    if result['prediction'] == 1:
                        st.error("### ⚠️ RA Positive")
                        result_color = "#ef4444"
                    else:
                        st.success("### ✅ No RA Detected")
                        result_color = "#10b981"
                    
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    # Risk Level
                    st.divider()
                    st.subheader("Risk Assessment")
                    
                    risk_level = result['risk_level']
                    if risk_level == "Low":
                        st.success(f"**Risk Level:** {risk_level}")
                    elif risk_level == "Medium":
                        st.warning(f"**Risk Level:** {risk_level}")
                    else:
                        st.error(f"**Risk Level:** {risk_level}")
                    
                    # Probability Gauge
                    prob = result['probability']['RA_Positive']
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "RA Probability (%)"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': result_color},
                            'steps': [
                                {'range': [0, 30], 'color': "#d1fae5"},
                                {'range': [30, 70], 'color': "#fed7aa"},
                                {'range': [70, 100], 'color': "#fee2e2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig)
                    
                    # Probabilities
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("RA Probability", 
                                 f"{result['probability']['RA_Positive']*100:.1f}%")
                    with col_b:
                        st.metric("No RA Probability", 
                                 f"{result['probability']['No_RA']*100:.1f}%")
                    
                    # Key Factors
                    st.divider()
                    st.subheader("Analysis & Key Factors")
                    
                    if result['explanation']['key_factors']:
                        for factor in result['explanation']['key_factors']:
                            st.info(f"🔸 {factor}")
                    else:
                        st.info("No significant risk factors detected.")
                    
                    # Recommendations
                    st.divider()
                    st.subheader("Recommendations")
                    
                    for rec in result['explanation']['recommendations']:
                        st.success(f"💡 {rec}")
                    
                    # Feature Importance
                    if 'top_features' in result['explanation']:
                        st.divider()
                        st.subheader("Most Important Features")
                        
                        features_df = pd.DataFrame(result['explanation']['top_features'])
                        features_df['importance'] = features_df['importance'] * 100
                        
                        fig = go.Figure(go.Bar(
                            x=features_df['importance'],
                            y=features_df['feature'],
                            orientation='h',
                            marker=dict(color='#2563eb')
                        ))
                        
                        fig.update_layout(
                            xaxis_title="Importance (%)",
                            yaxis_title="Feature",
                            height=300,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        
                        st.plotly_chart(fig)
                    
                else:
                    st.error(f"Error: {response.json().get('error', 'Prediction failed')}")
                    
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.info("Please ensure the backend server is running on http://127.0.0.1:5001")
    else:
        st.info("👈 Enter patient information and click 'Predict RA Risk' to see results")
        
        # Show placeholder
        st.image("https://via.placeholder.com/500x300/f8fafc/64748b?text=Prediction+Results+Will+Appear+Here")

# Footer
st.divider()
st.caption("🏥 Rheumatoid Arthritis Prediction System | Powered by Random Forest ML | Educational Project")
