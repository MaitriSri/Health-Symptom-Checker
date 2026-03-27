import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("dataset.csv")

# Split data
X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Title
st.title("🩺 Health Symptom Checker")

st.write("Select your symptoms:")

# Inputs
fever = st.checkbox("Fever")
headache = st.checkbox("Headache")
fatigue = st.checkbox("Fatigue")
cough = st.checkbox("Cough")

# Button
if st.button("Predict"):
    prediction = model.predict([[fever, headache, fatigue, cough]])

    st.subheader("Result:")
    st.write("Possible Disease:", prediction[0])

    # Advice
    advice = {
        "Flu": "Take rest and drink fluids",
        "Cold": "Stay warm and hydrated",
        "Migraine": "Avoid stress and take rest"
    }

    st.write("Advice:", advice.get(prediction[0], "Consult doctor"))