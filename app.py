import os
from flask import Flask, render_template, request, jsonify
import joblib
from collections import Counter

app = Flask(__name__)

model, mlb = joblib.load("model/disease_classifier.pkl")

# 🧠 MEMORY
memory = {"symptoms": []}

# 🩺 FULL HEALTH DATA (all diseases)
health_advice = {
    "flu": {
        "severity": "MEDIUM",
        "medicine": ["Paracetamol", "Ibuprofen", "Oseltamivir (Tamiflu)"],
        "food_take": ["Chicken soup", "Warm fluids", "Ginger tea", "Citrus fruits"],
        "food_avoid": ["Cold drinks", "Alcohol", "Processed food"],
        "steps": ["Rest at home", "Drink plenty of fluids", "Monitor temperature"],
        "warning": "Consult doctor if fever exceeds 103°F or lasts more than 3 days",
        "emergency": ["chest pain", "difficulty breathing", "persistent vomiting"]
    },
    "cold": {
        "severity": "LOW",
        "medicine": ["Paracetamol", "Antihistamines", "Decongestants"],
        "food_take": ["Hot tea", "Warm soup", "Honey", "Ginger"],
        "food_avoid": ["Ice cream", "Cold drinks", "Dairy products"],
        "steps": ["Rest well", "Stay warm", "Use saline nasal spray"],
        "warning": "If symptoms persist beyond 10 days, see a doctor",
        "emergency": ["high fever", "severe headache", "difficulty swallowing"]
    },
    "heart_attack": {
        "severity": "CRITICAL",
        "medicine": ["Aspirin (chew immediately)", "Nitroglycerin if prescribed"],
        "food_take": ["None - seek emergency help immediately"],
        "food_avoid": ["Everything until medically cleared"],
        "steps": ["CALL AMBULANCE IMMEDIATELY", "Chew aspirin if available", "Loosen tight clothing", "Stay calm and sit/lie down"],
        "warning": "🚨 LIFE THREATENING - DO NOT DELAY",
        "emergency": ["chest pain", "left arm pain", "sweating", "jaw pain", "shortness of breath"]
    },
    "diabetes": {
        "severity": "HIGH",
        "medicine": ["Metformin", "Insulin (if prescribed)", "Glipizide"],
        "food_take": ["Leafy vegetables", "Whole grains", "Lean protein", "Water"],
        "food_avoid": ["Sugar", "White bread", "Sugary drinks", "Processed snacks"],
        "steps": ["Exercise daily", "Monitor blood sugar", "Take medications on time"],
        "warning": "Chronic condition - requires lifelong management",
        "emergency": ["blurred vision", "extreme thirst", "loss of consciousness"]
    },
    "migraine": {
        "severity": "MEDIUM",
        "medicine": ["Ibuprofen", "Sumatriptan", "Paracetamol"],
        "food_take": ["Water", "Ginger tea", "Magnesium-rich foods"],
        "food_avoid": ["Caffeine", "Alcohol", "Chocolate", "Aged cheese", "MSG"],
        "steps": ["Rest in a dark quiet room", "Apply cold/warm compress", "Stay hydrated"],
        "warning": "See neurologist if migraines are frequent or severe",
        "emergency": ["sudden severe headache", "vision changes", "confusion", "weakness"]
    },
    "food_poisoning": {
        "severity": "MEDIUM",
        "medicine": ["ORS (Oral Rehydration Solution)", "Loperamide", "Antiemetics"],
        "food_take": ["Clear fluids", "BRAT diet (banana, rice, applesauce, toast)", "Electrolytes"],
        "food_avoid": ["Dairy", "Fatty foods", "Spicy food", "Caffeine", "Alcohol"],
        "steps": ["Stay hydrated", "Rest", "Avoid solid food initially", "Gradually reintroduce bland food"],
        "warning": "Seek medical help if symptoms last more than 48 hours",
        "emergency": ["bloody stool", "high fever", "severe dehydration", "loss of consciousness"]
    },
    "allergy": {
        "severity": "LOW-MEDIUM",
        "medicine": ["Antihistamines (Cetirizine, Loratadine)", "Hydrocortisone cream", "Epinephrine (if severe)"],
        "food_take": ["Anti-inflammatory foods", "Vitamin C rich foods"],
        "food_avoid": ["Known allergens", "Processed food with additives"],
        "steps": ["Identify and avoid allergen", "Take prescribed antihistamines", "Keep an EpiPen if allergist recommends"],
        "warning": "Anaphylaxis is life-threatening - carry EpiPen if advised",
        "emergency": ["throat swelling", "difficulty breathing", "drop in blood pressure", "anaphylaxis"]
    },
    "chickenpox": {
        "severity": "MEDIUM",
        "medicine": ["Calamine lotion", "Antihistamines", "Acyclovir (antiviral)", "Paracetamol"],
        "food_take": ["Soft foods", "Cold yogurt", "Smoothies", "Water"],
        "food_avoid": ["Spicy food", "Salty food", "Acidic foods"],
        "steps": ["Avoid scratching blisters", "Trim nails short", "Wear loose clothing", "Isolate to prevent spread"],
        "warning": "Highly contagious - avoid contact with pregnant women and immunocompromised",
        "emergency": ["severe headache", "stiff neck", "difficulty breathing", "confusion"]
    },
    "pneumonia": {
        "severity": "HIGH",
        "medicine": ["Antibiotics (Amoxicillin)", "Cough suppressants", "Fever reducers"],
        "food_take": ["Warm broths", "Herbal teas", "Protein-rich foods", "Vitamin C"],
        "food_avoid": ["Cold foods", "Alcohol", "Smoking"],
        "steps": ["Seek medical care promptly", "Complete antibiotic course", "Rest fully", "Monitor oxygen levels"],
        "warning": "Requires medical diagnosis and treatment - do not self-medicate",
        "emergency": ["cyanosis (blue lips)", "confusion", "rapid breathing >30/min", "very low blood pressure"]
    },
    "malaria": {
        "severity": "HIGH",
        "medicine": ["Chloroquine", "Artemisinin-based therapy (ACT)", "Primaquine"],
        "food_take": ["High-calorie foods", "Fluids", "Fruits", "Soups"],
        "food_avoid": ["Alcohol", "Raw foods", "Oily food"],
        "steps": ["Seek immediate medical diagnosis", "Use mosquito nets", "Complete full medication course"],
        "warning": "Can be fatal if untreated - seek diagnosis via blood test",
        "emergency": ["seizures", "loss of consciousness", "severe anemia", "jaundice"]
    },
    "dengue": {
        "severity": "HIGH",
        "medicine": ["Paracetamol (only - NO aspirin/ibuprofen)", "ORS", "Platelet transfusion if needed"],
        "food_take": ["Papaya leaf juice", "Coconut water", "Pomegranate juice", "Warm fluids"],
        "food_avoid": ["Aspirin", "Ibuprofen", "Oily food", "Spicy food"],
        "steps": ["Hospitalize if severe", "Monitor platelet count", "Stay hydrated", "Rest completely"],
        "warning": "⚠️ DO NOT take aspirin or ibuprofen - can cause dangerous bleeding",
        "emergency": ["bleeding gums", "blood in urine/stool", "severe abdominal pain", "rapid breathing"]
    },
    "covid19": {
        "severity": "MEDIUM-HIGH",
        "medicine": ["Paracetamol", "Vitamins C & D", "Zinc", "Antivirals if prescribed (Paxlovid)"],
        "food_take": ["Warm fluids", "High-protein foods", "Citrus fruits", "Turmeric milk"],
        "food_avoid": ["Alcohol", "Tobacco", "Cold drinks"],
        "steps": ["Isolate immediately", "Monitor oxygen with pulse oximeter", "Rest", "Follow health authority guidelines"],
        "warning": "Isolate for at least 5 days - highly contagious",
        "emergency": ["oxygen saturation <94%", "persistent chest pain", "confusion", "bluish lips"]
    },
    "high_blood_pressure": {
        "severity": "HIGH",
        "medicine": ["Amlodipine", "Lisinopril", "Losartan", "Beta-blockers"],
        "food_take": ["Low-sodium foods", "Leafy greens", "Berries", "Oats", "Bananas"],
        "food_avoid": ["Salt", "Processed food", "Red meat", "Caffeine", "Alcohol"],
        "steps": ["Monitor BP regularly", "Exercise 30 min/day", "Reduce stress", "Maintain healthy weight"],
        "warning": "Silent killer - regular monitoring is essential",
        "emergency": ["severe headache", "vision changes", "chest pain", "BP >180/120"]
    },
    "meningitis": {
        "severity": "CRITICAL",
        "medicine": ["Antibiotics (IV Penicillin/Ceftriaxone)", "Corticosteroids", "Antiviral (if viral)"],
        "food_take": ["Fluids only until stabilized"],
        "food_avoid": ["Everything until medically cleared"],
        "steps": ["EMERGENCY HOSPITALIZATION REQUIRED", "IV antibiotics must be started immediately", "Isolate patient"],
        "warning": "🚨 MEDICAL EMERGENCY - Can cause death or permanent disability within hours",
        "emergency": ["stiff neck", "severe headache", "sensitivity to light", "purple rash", "seizures", "loss of consciousness"]
    }
}

# 🔥 SYMPTOM MAP - maps symptoms to diseases (priority rule-based system)
# More specific symptoms get higher priority
symptom_map = {
    # Heart Attack (CRITICAL - highest priority)
    "chest_pain": "heart_attack",
    "pain_in_arms": "heart_attack",
    "shortness_of_breath": "heart_attack",
    "sweating": "heart_attack",

    # Meningitis (CRITICAL)
    "stiff_neck": "meningitis",
    "seizures": "meningitis",

    # Dengue (specific symptoms)
    "pain_behind_eyes": "dengue",

    # COVID-19
    "loss_of_taste": "covid19",
    "loss_of_smell": "covid19",

    # Malaria
    "muscle_pain": "malaria",

    # Pneumonia
    "rapid_breathing": "pneumonia",
    "difficulty_breathing": "pneumonia",

    # Chickenpox
    "blisters": "chickenpox",

    # Migraine
    "severe_headache": "migraine",
    "throbbing_headache": "migraine",
    "frequent_headache": "high_blood_pressure",

    # Food Poisoning
    "vomiting": "food_poisoning",
    "diarrhea": "food_poisoning",
    "stomach_cramps": "food_poisoning",
    "abdominal_pain": "food_poisoning",

    # Allergy
    "skin_rash": "allergy",
    "hives": "allergy",
    "itching": "allergy",
    "swelling": "allergy",

    # High Blood Pressure
    "nosebleeds": "high_blood_pressure",

    # Flu
    "high_fever": "flu",
    "body_pain": "flu",
    "chills": "flu",
    "weakness": "flu",

    # Cold (lowest priority - very common symptoms)
    "runny_nose": "cold",
    "sneezing": "cold",
    "sore_throat": "cold",
    "congestion": "cold",
    "watery_eyes": "cold",

    # Shared symptoms (mapped to most likely)
    "fever": "flu",
    "cough": "cold",
    "headache": "migraine",
    "fatigue": "flu",
    "nausea": "food_poisoning",
    "dizziness": "migraine",
}

# Disease priority scores (higher = more serious/specific)
disease_priority = {
    "heart_attack": 100,
    "meningitis": 95,
    "pneumonia": 80,
    "dengue": 75,
    "malaria": 75,
    "covid19": 70,
    "high_blood_pressure": 65,
    "chickenpox": 60,
    "food_poisoning": 55,
    "migraine": 50,
    "flu": 40,
    "allergy": 35,
    "diabetes": 30,
    "cold": 10
}


def predict(symptoms):
    """
    Smart single-disease prediction:
    - Counts votes from symptom_map
    - Picks the disease with most symptom matches
    - Breaks ties by disease priority (more serious wins)
    - Falls back to ML if no rule matches
    """
    if not symptoms:
        return "cold", 0.5

    # Count votes for each disease
    vote_counter = Counter()
    for s in symptoms:
        s_clean = s.strip().lower().replace(" ", "_")
        if s_clean in symptom_map:
            vote_counter[symptom_map[s_clean]] += 1

    if vote_counter:
        # Find max votes
        max_votes = max(vote_counter.values())
        # Get all diseases with max votes
        top_diseases = [d for d, v in vote_counter.items() if v == max_votes]

        # Break tie by priority (highest priority wins)
        best_disease = max(top_diseases, key=lambda d: disease_priority.get(d, 0))
        confidence = min(0.6 + (max_votes * 0.1), 0.99)
        return best_disease, confidence

    # ML fallback
    try:
        vector = mlb.transform([symptoms])
        d = model.predict(vector)[0]
        c = model.predict_proba(vector).max()
        return d, c
    except:
        return "cold", 0.5


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("message", "").lower()

    symptoms = text.split(",")
    symptoms = [s.strip().replace(" ", "_") for s in symptoms if s.strip()]

    memory["symptoms"] = symptoms

    disease, confidence = predict(memory["symptoms"])
    advice = health_advice.get(disease, {
        "severity": "UNKNOWN",
        "medicine": ["Please consult a doctor"],
        "food_take": ["Balanced diet"],
        "food_avoid": ["Junk food"],
        "steps": ["See a healthcare professional"],
        "warning": "Consult a doctor for proper diagnosis",
        "emergency": ["Seek medical help if condition worsens"]
    })

    # 🩺 FORMATTED OUTPUT
    output = f"""
🩺 ADVANCED AI HEALTHCARE CHATBOT
--------------------------------------------------

📝 Symptoms Entered: {', '.join([s.replace('_', ' ') for s in memory['symptoms']])}

🎯 Predicted Disease: {disease.upper().replace('_', ' ')}
📊 Confidence: {round(confidence * 100)}%

🏥 DIAGNOSIS: {disease.upper().replace('_', ' ')}
📊 SEVERITY: {advice.get('severity')}

💊 Medicines:
{chr(10).join(['• ' + m for m in advice.get('medicine', [])])}

🥗 Food to Take:
{chr(10).join(['✓ ' + f for f in advice.get('food_take', [])])}

🚫 Food to Avoid:
{chr(10).join(['✗ ' + f for f in advice.get('food_avoid', [])])}

📋 Steps:
{chr(10).join(['→ ' + s for s in advice.get('steps', [])])}

⚠️ Warning:
{advice.get('warning')}

🚨 Emergency Signs (seek help immediately):
{chr(10).join(['⚡ ' + e for e in advice.get('emergency', [])])}
"""

    return jsonify({
        "reply": output,
        "disease": disease
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
