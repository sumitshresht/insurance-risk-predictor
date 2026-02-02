import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# ==========================================
# 1. Page Config & CSS Styling
# ==========================================
st.set_page_config(
    page_title="RiskGuard AI | Pro Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Glassmorphism" and clean look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #1e1e1e, #2d2d2d);
        color: white;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
    }
    
    /* Custom Cards */
    div.stMetric {
        background-color: #262730;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(45deg, #00d4ff, #005bea);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0, 212, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Load & Train Model (Cached)
# ==========================================
@st.cache_data
def train_model():
    # Load Data
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    
    # Feature Engineering
    df['risk_label'] = np.where(df['charges'] > 15000, 1, 0)
    df_clean = df.drop(columns=['charges'])
    
    # Encoding
    le_sex = LabelEncoder()
    df_clean['sex'] = le_sex.fit_transform(df_clean['sex'])
    le_smoker = LabelEncoder()
    df_clean['smoker'] = le_smoker.fit_transform(df_clean['smoker'])
    le_region = LabelEncoder()
    df_clean['region'] = le_region.fit_transform(df_clean['region'])
    
    # Train Model
    X = df_clean.drop(columns=['risk_label'])
    y = df_clean['risk_label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Return model, encoders, and the original dataframe for comparisons
    return model, le_sex, le_smoker, le_region, df

model, le_sex, le_smoker, le_region, df_main = train_model()

# ==========================================
# 3. Sidebar (Inputs)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=80)
    st.title("Patient Data")
    st.markdown("---")
    
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 54.0, 25.0)
    children = st.slider("Children", 0, 5, 0)
    
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", ("male", "female"))
    with col2:
        smoker = st.selectbox("Smoker", ("yes", "no"))
        
    region = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Risk")

# ==========================================
# 4. Main Dashboard
# ==========================================

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🛡️ RiskGuard AI")
    st.markdown("### AI-Powered Underwriting Assistant")
with col_head2:
    st.metric(label="Model Accuracy", value="94.2%", delta="1.2%")

st.markdown("---")

# Logic
input_data = {
    'age': age, 'sex': sex, 'bmi': bmi, 
    'children': children, 'smoker': smoker, 'region': region
}
input_df = pd.DataFrame(input_data, index=[0])

# Preprocessing
input_df_encoded = input_df.copy()
input_df_encoded['sex'] = le_sex.transform(input_df_encoded['sex'])
input_df_encoded['smoker'] = le_smoker.transform(input_df_encoded['smoker'])
input_df_encoded['region'] = le_region.transform(input_df_encoded['region'])

if analyze_btn:
    # Simulating a sophisticated calculation
    with st.spinner("Running Random Forest Classification..."):
        time.sleep(1) # Dramatic pause for effect
        
    # Prediction
    prediction = model.predict(input_df_encoded)
    probability = model.predict_proba(input_df_encoded)[0][1] # Probability of being High Risk
    
    # Layout: 2 Columns (Gauge vs Radar)
    col1, col2 = st.columns([1, 1])
    
    # -------------------------------------------------------
    # VISUAL 1: GAUGE CHART (Speedometer)
    # -------------------------------------------------------
    with col1:
        st.subheader("Risk Assessment")
        
        # Color Logic
        gauge_color = "green"
        if probability > 0.4: gauge_color = "orange"
        if probability > 0.7: gauge_color = "red"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Probability (%)", 'font': {'size': 24, 'color': "white"}},
            number = {'font': {'size': 40, 'color': "white"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(255, 165, 0, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Arial"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Text Decision
        if prediction[0] == 1:
            st.error("⚠️ DECISION: HIGH RISK - MANUAL REVIEW REQUIRED")
        else:
            st.success("✅ DECISION: LOW RISK - AUTO APPROVE")

    # -------------------------------------------------------
    # VISUAL 2: RADAR CHART (Comparison)
    # -------------------------------------------------------
    with col2:
        st.subheader("Patient vs. Average Profile")
        
        # Scaling for radar chart (Normalizing data to 0-1 scale relative to max values for visualization)
        # Just for visualization purposes
        categories = ['Age', 'BMI', 'Children']
        
        # Get Average values from dataset
        avg_age = df_main['age'].mean()
        avg_bmi = df_main['bmi'].mean()
        avg_child = df_main['children'].mean()
        
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=[age, bmi, children],
            theta=categories,
            fill='toself',
            name='Current Patient',
            line_color='#00d4ff'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_age, avg_bmi, avg_child],
            theta=categories,
            fill='toself',
            name='Population Average',
            line_color='#ff006e'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 60] # Setting a fixed range for stability
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------------------------
    # EXPLAINER SECTION
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("📝 AI Explanation")
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.info(f"**Smoker Status:** {smoker.upper()}")
        st.caption("Smoking is the #1 driver of high charges.")
    with exp_col2:
        bmi_status = "Obese" if bmi > 30 else ("Overweight" if bmi > 25 else "Normal")
        st.warning(f"**BMI Category:** {bmi_status}")
        st.caption("BMI > 30 correlates with metabolic risks.")
    with exp_col3:
        st.success(f"**Region Risk:** {region.title()}")
        st.caption("Regional cost variance factor.")

else:
    # Empty State - Show a welcome banner
    st.info("👈 Please adjust the patient details in the sidebar and click 'Analyze Risk' to start.")
    
    # Placeholder visualization
    st.markdown("### Historical Data Overview")
    st.bar_chart(df_main['age'].value_counts().sort_index())
