import streamlit as st
import joblib
import numpy as np

# Load saved models
scaler = joblib.load(rb"models\scaler.pkl")
kmeans = joblib.load(rb"models\kmeans.pkl")
best_models = joblib.load(rb"models\best_models_per_cluster.pkl")

columns = ['Cement (kg/m3)', 'Blast Furnace Slag (kg/m3)', 'Fly Ash (kg/m3)',
           'Water (kg/m3)', 'Superplasticizer (kg/m3)', 'Coarse Aggregate (kg/m3)',
           'Fine Aggregate (kg/m3)', 'Age (days)']

def predict_concrete(sample):
    sample = np.array([sample[col] for col in columns]).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    cluster = kmeans.predict(sample_scaled)[0]
    model = best_models[cluster]
    return float(model.predict(sample_scaled)[0])

st.title("Concrete Compressive Strength Predictor")
st.write("Enter the concrete mix components below:")

sample = {}
for col in columns:
    sample[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

if st.button("Predict"):
    pred = predict_concrete(sample)
    st.success(f"Predicted Compressive Strength: {pred:.2f} MPa")
