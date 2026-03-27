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

# Take user input
fever = int(input("Fever (1/0): "))
headache = int(input("Headache (1/0): "))
fatigue = int(input("Fatigue (1/0): "))
cough = int(input("Cough (1/0): "))

# Predict
prediction = model.predict([[fever, headache, fatigue, cough]])

print("Predicted Disease:", prediction[0])

# Advice
advice = {
    "Flu": "Take rest and drink fluids",
    "Cold": "Stay warm and hydrated",
    "Migraine": "Avoid stress and take rest"
}

print("Advice:", advice.get(prediction[0], "Consult doctor"))