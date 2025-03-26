import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="🏥 HealthGuard AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    return data

data = load_data()

# Train model with improvements
@st.cache_resource
def train_model():
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    feature_selector = ExtraTreesClassifier(n_estimators=700, random_state=42)
    feature_selector.fit(X, y)
    selected_features = X.columns[feature_selector.feature_importances_ > 0.02]
    X = X[selected_features]

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return model, scaler, model_accuracy, selected_features, feature_importance

# Train model and get results
model, scaler, model_accuracy, selected_features, feature_importance = train_model()

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            text-align: center;
        }
        .prediction-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }
        .health-tips {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Header with Title and Image
st.markdown('<h1 style="text-align: center; color: #2c3e50;">🏥 HealthGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #34495e;">Advanced Health Risk Assessment Powered by Machine Learning</h3>', unsafe_allow_html=True)

# Display Image Below Header
st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photo%2Fside-view-smiley-doctor-patient_34728735.htm&psig=AOvVaw1Ptb8maqlBuuE7XLRdi0Zm&ust=1743050085826000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCIilp4L2powDFQAAAAAdAAAAABAJ", use_column_width=True)

# Key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db;">Model Accuracy</h3>
            <h2>{model_accuracy:.1%}</h2>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db;">Features Analyzed</h3>
            <h2>{len(selected_features)}</h2>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db;">Dataset Size</h3>
            <h2>{len(data)}</h2>
        </div>
    """, unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: #3498db;'>🧑‍⚕️ Your Health Profile</h2>
    </div>
""", unsafe_allow_html=True)

user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.slider(
        f"{feature}",
        float(data[feature].min()),
        float(data[feature].max()),
        float(data[feature].mean()),
        help=f"Average {feature}: {data[feature].mean():.2f}"
    )

user_input_df = pd.DataFrame([user_input])
user_input_scaled = scaler.transform(user_input_df)

# Analyze Button
st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
assess_button = st.button("🚀 Analyze Health Risk")
st.markdown('</div>', unsafe_allow_html=True)

if assess_button:
    prediction = model.predict(user_input_scaled)[0]
    prediction_proba = model.predict_proba(user_input_scaled)[0][1]
    risk = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"

    st.markdown(f"""
        <div class="prediction-card">
            <h2 style='color: {"#e74c3c" if prediction == 1 else "#2ecc71"}; font-size: 2.5em;'>{risk}</h2>
            <p style='font-size: 1.2em; color: #666;'>Confidence Score: {prediction_proba:.1%}</p>
        </div>
    """, unsafe_allow_html=True)
