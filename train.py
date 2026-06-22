import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# Read data
data = pd.read_csv("data/symptoms_dataset.csv")

# Split symptoms
data["symptom_list"] = data["symptoms"].apply(lambda x: x.split(","))

# Change words to numbers
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(data["symptom_list"])

# Disease column
y = data["disease"]

# Machine learning brain
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save brain
joblib.dump((model, mlb), "model/disease_classifier.pkl")

# Print all available symptoms
print("\n" + "="*60)
print("📋 AVAILABLE SYMPTOMS FOR INPUT:")
print("="*60)
all_symptoms = sorted(set(symptom.strip() for symptoms in data["symptoms"] for symptom in symptoms.split(",")))
for i, symptom in enumerate(all_symptoms, 1):
    print(f"{i:3}. {symptom}")
print("="*60)
print(f"\n✅ Model trained successfully!")
print(f"📊 Total symptoms: {len(all_symptoms)}")
print(f"🏥 Total diseases: {data['disease'].nunique()}")
print(f"📝 Total training samples: {len(data)}")