import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Train models and compare
@st.cache_resource
def train_models():
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

    models = {
        'XGBoost': XGBClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[model_name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']

    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return best_model, scaler, results, selected_features, feature_importance

model, scaler, model_results, selected_features, feature_importance = train_models()

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
    </style>
""", unsafe_allow_html=True)

# Header with Title and Image
st.markdown('<h1 style="text-align: center; color: #2c3e50;">🏥 HealthGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #34495e;">Advanced Health Risk Assessment Powered by Machine Learning</h3>', unsafe_allow_html=True)

st.image("https://raw.githubusercontent.com/yashkd-fresher/Personalized-Health-Risk-Assessment/main/premium_photo-1663054397533-2a3fb0cab5de.jpeg", use_container_width=True)

# Key Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db;">Best Model</h3>
            <h2>{max(model_results, key=lambda k: model_results[k]['accuracy'])}</h2>
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

# Correlation Heatmap
st.markdown("### 🔍 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature Importance Chart
st.markdown("### 📊 Feature Importance")
st.bar_chart(feature_importance.set_index('Feature'))

# Sidebar Inputs
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

if st.button("🚀 Analyze Health Risk"):
    prediction = model.predict(user_input_scaled)[0]
    risk = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"

    st.markdown(f"""
        <div class="prediction-card">
            <h2 style='color: {'#e74c3c' if prediction == 1 else '#2ecc71'}; font-size: 2.5em;'>{risk}</h2>
        </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("### 🛡️ Preventive Measures")
        st.write("- Maintain a balanced diet and regular exercise.")
        st.write("- Monitor blood sugar levels regularly.")
        st.write("- Consult a healthcare professional for personalized advice.")
