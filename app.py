import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor")

age = st.number_input("Enter Age", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Enter Fare", min_value=0.0, value=50.0)

if st.button("Predict"):
    input_data = np.array([[age, fare]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("🎉 Survived")
    else:
        st.error("💀 Did Not Survive")
