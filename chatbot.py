"""
=============================================================================
ADVANCED AI HEALTHCARE CHATBOT WITH VOICE RECOGNITION
=============================================================================
Features:
✓ Voice Input (Speech-to-Text)
✓ Voice Output (Text-to-Speech)
✓ ML-Based Disease Prediction
✓ Detailed Health Advice
✓ Emergency Detection
✓ Severity Assessment
✓ Confidence Score
✓ User-Friendly Interface
=============================================================================
"""

import joblib
import speech_recognition as sr
import pyttsx3
import time
import os

# ===============================
# LOAD TRAINED MODEL
# ===============================
model, mlb = joblib.load("model/disease_classifier.pkl")

# ===============================
# DETAILED HEALTH ADVICE DATABASE
# ===============================
health_advice = {
    "cold": {
        "severity": "low",
        "description": "A common viral infection of the nose and throat.",
        "medicine": [
            "Paracetamol 500mg (for fever)",
            "Antihistamine (for sneezing)",
            "Decongestant nasal spray",
            "Vitamin C supplements"
        ],
        "food_take": [
            "Warm chicken soup",
            "Ginger tea with honey",
            "Hot water with lemon",
            "Orange and citrus fruits",
            "Turmeric milk",
            "Light porridge"
        ],
        "food_avoid": [
            "Cold drinks and ice water",
            "Ice cream and cold desserts",
            "Fried and oily foods",
            "Dairy products (if congestion)",
            "Alcohol and caffeine"
        ],
        "steps": [
            "Get complete bed rest for 2-3 days",
            "Drink 8-10 glasses of warm water daily",
            "Do steam inhalation 2-3 times daily",
            "Gargle with warm salt water",
            "Cover mouth and nose while coughing/sneezing",
            "Use a humidifier in your room"
        ],
        "warning": "Consult doctor if symptoms persist beyond 7 days or fever exceeds 103°F",
        "emergency_signs": ["high_fever", "difficulty_breathing", "chest_pain"]
    },

    "flu": {
        "severity": "medium",
        "description": "Influenza - a contagious respiratory illness caused by influenza viruses.",
        "medicine": [
            "Oseltamivir (Tamiflu) - within 48 hours",
            "Paracetamol 650mg for fever",
            "Ibuprofen for body pain",
            "Antihistamine for runny nose",
            "Cough suppressant if needed"
        ],
        "food_take": [
            "Warm chicken or vegetable soup",
            "Honey and warm water",
            "Fresh fruit juices",
            "Herbal teas (ginger, peppermint)",
            "Light rice and boiled vegetables",
            "Yogurt with probiotics"
        ],
        "food_avoid": [
            "Cold beverages",
            "Fried and fatty foods",
            "Sugary drinks",
            "Alcohol and coffee",
            "Hard to digest foods"
        ],
        "steps": [
            "Stay home and rest completely",
            "Isolate from family members",
            "Drink plenty of fluids",
            "Monitor temperature every 4 hours",
            "Wear a surgical mask",
            "Take prescribed antiviral medications"
        ],
        "warning": "Seek immediate medical care if breathing becomes difficult or fever is very high",
        "emergency_signs": ["difficulty_breathing", "chest_pain", "confusion", "severe_vomiting"]
    },

    "migraine": {
        "severity": "medium",
        "description": "A neurological condition causing severe headaches and other symptoms.",
        "medicine": [
            "Sumatriptan (prescription)",
            "Paracetamol or Ibuprofen",
            "Anti-nausea medication",
            "Beta-blockers (preventive)",
            "Topiramate (preventive)"
        ],
        "food_take": [
            "Plenty of water (dehydration causes migraines)",
            "Magnesium-rich foods (spinach, almonds)",
            "Complex carbohydrates",
            "Fresh fruits and vegetables",
            "Ginger tea"
        ],
        "food_avoid": [
            "Caffeine (excess)",
            "Alcohol (especially red wine)",
            "Aged cheeses",
            "Processed meats (nitrates)",
            "Artificial sweeteners",
            "Chocolate",
            "Monosodium glutamate (MSG)"
        ],
        "steps": [
            "Rest in a dark, quiet room",
            "Apply cold compress to forehead",
            "Practice deep breathing exercises",
            "Massage temples gently",
            "Stay hydrated",
            "Maintain regular sleep schedule",
            "Identify and avoid triggers"
        ],
        "warning": "Consult neurologist if headaches occur more than twice weekly or vision changes",
        "emergency_signs": ["sudden_severe_headache", "vision_loss", "speech_difficulty", "weakness"]
    },

    "food_poisoning": {
        "severity": "medium",
        "description": "Illness caused by consuming contaminated food or beverages.",
        "medicine": [
            "ORS (Oral Rehydration Solution)",
            "Antacid for stomach relief",
            "Anti-diarrheal medication (loperamide)",
            "Antibiotics (if bacterial - prescribed)",
            "Probiotics after recovery"
        ],
        "food_take": [
            "ORS solution every 15-30 minutes",
            "Bananas (potassium)",
            "Plain rice or toast",
            "Applesauce",
            "Boiled potatoes",
            "Coconut water",
            "Plain yogurt after 24 hours"
        ],
        "food_avoid": [
            "Spicy and fatty foods",
            "Dairy products initially",
            "Caffeine and alcohol",
            "High-fiber foods",
            "Raw vegetables",
            "Sugary drinks"
        ],
        "steps": [
            "Stop eating solid food immediately",
            "Drink ORS slowly and frequently",
            "Rest and avoid physical exertion",
            "Do not take anti-diarrheal if bloody stool",
            "Monitor for dehydration signs",
            "Maintain hygiene to prevent spread",
            "Wash hands frequently"
        ],
        "warning": "Go to hospital if vomiting blood, bloody diarrhea, or severe dehydration occurs",
        "emergency_signs": ["blood_in_vomit", "blood_in_stool", "severe_dehydration", "high_fever"]
    },

    "diabetes": {
        "severity": "high",
        "description": "A chronic condition affecting how your body processes blood sugar (glucose).",
        "medicine": [
            "Metformin (first-line medication)",
            "Insulin therapy (as prescribed)",
            "GLP-1 receptor agonists",
            "SGLT2 inhibitors",
            "Regular blood sugar monitoring"
        ],
        "food_take": [
            "Leafy green vegetables",
            "Whole grains (oats, quinoa, brown rice)",
            "High-fiber foods",
            "Lean proteins (fish, chicken, tofu)",
            "Nuts and seeds",
            "Berries and low-sugar fruits",
            "Healthy fats (olive oil, avocado)"
        ],
        "food_avoid": [
            "Sugar and sweets",
            "Soft drinks and fruit juices",
            "White bread and refined carbs",
            "Fried foods",
            "Processed snacks",
            "High-sodium foods",
            "Alcohol (especially beer and sweet wines)"
        ],
        "steps": [
            "Check blood sugar levels regularly",
            "Take medications as prescribed",
            "Exercise 30 minutes daily",
            "Eat small, frequent meals",
            "Maintain healthy weight",
            "Keep a food and sugar diary",
            "Regular check-ups with endocrinologist",
            "Foot care daily inspection"
        ],
        "warning": "This is a chronic condition requiring lifelong management. Always consult your doctor.",
        "emergency_signs": ["extreme_thirst", "frequent_urination", "confusion", "fruity_breath", "nausea"]
    },

    "heart_attack": {
        "severity": "critical",
        "description": "A medical emergency where blood flow to the heart is blocked.",
        "medicine": [
            "Aspirin 325mg (chew immediately)",
            "Nitroglycerin (if prescribed)",
            "Emergency medical services ONLY"
        ],
        "food_take": [
            "None - seek emergency help first",
            "After recovery: Heart-healthy diet"
        ],
        "food_avoid": [
            "None - emergency first"
        ],
        "steps": [
            "CALL EMERGENCY SERVICES IMMEDIATELY",
            "Chew aspirin 325mg (if not allergic)",
            "Sit or lie down in comfortable position",
            "Loosen tight clothing",
            "Stay calm and wait for ambulance",
            "Do NOT drive yourself to hospital",
            "If trained, perform CPR if person becomes unresponsive"
        ],
        "warning": "⚠️ THIS IS A LIFE-THREATENING EMERGENCY! Every minute counts!",
        "emergency_signs": ["chest_pain", "shortness_of_breath", "pain_in_arms", "sweating", "nausea"]
    },

    "allergy": {
        "severity": "low",
        "description": "An immune response to substances that are usually harmless.",
        "medicine": [
            "Antihistamine (Cetirizine, Loratadine)",
            "Decongestant nasal spray",
            "Corticosteroid cream for skin",
            "Epinephrine auto-injector (severe allergies)",
            "Eye drops for allergic conjunctivitis"
        ],
        "food_take": [
            "Anti-inflammatory foods",
            "Quercetin-rich foods (onions, apples)",
            "Vitamin C rich foods",
            "Probiotic-rich foods",
            "Omega-3 fatty acids"
        ],
        "food_avoid": [
            "Known allergens",
            "Processed foods with additives",
            "Sulfites (wine, dried fruits)",
            "Shellfish and nuts (if allergic)",
            "Cross-reactive foods"
        ],
        "steps": [
            "Identify and avoid triggers",
            "Keep windows closed during high pollen",
            "Use air purifiers",
            "Shower after being outdoors",
            "Wash bedding frequently",
            "Keep emergency medication handy",
            "Wear medical alert bracelet"
        ],
        "warning": "Seek emergency care for anaphylaxis signs: difficulty breathing, throat swelling",
        "emergency_signs": ["difficulty_breathing", "throat_swelling", "dizziness", "rapid_heartbeat"]
    },

    "chickenpox": {
        "severity": "medium",
        "description": "A highly contagious viral infection, mainly affecting children.",
        "medicine": [
            "Paracetamol for fever",
            "Antihistamine for itching",
            "Calamine lotion for skin",
            "Acyclovir (for adults/immunocompromised)",
            "Antibiotics if infected sores"
        ],
        "food_take": [
            "Soft, easy-to-digest foods",
            "Cool foods (yogurt, smoothies)",
            "Plenty of fluids",
            "Blended soups",
            "Fresh fruits"
        ],
        "food_avoid": [
            "Hard, crunchy foods",
            "Spicy and acidic foods",
            "Salty foods (increases itching)",
            "Hot beverages"
        ],
        "steps": [
            "Isolate until all blisters scab over",
            "Keep skin clean and dry",
            "Do NOT scratch (causes scars)",
            "Apply calamine lotion",
            "Take lukewarm baths with oatmeal",
            "Trim fingernails short",
            "Use gloves at night if scratching"
        ],
        "warning": "Consult doctor if pregnant, immunocompromised, or symptoms are severe",
        "emergency_signs": ["high_fever", "difficulty_breathing", "severe_headache", "confusion"]
    },

    "tonsillitis": {
        "severity": "medium",
        "description": "Inflammation of the tonsils, usually due to infection.",
        "medicine": [
            "Paracetamol or Ibuprofen",
            "Antibiotics (if bacterial - prescribed)",
            "Throat lozenges",
            "Gargle with antiseptic mouthwash"
        ],
        "food_take": [
            "Warm soups and broths",
            "Soft fruits (banana, avocado)",
            "Honey and warm water",
            "Smoothies",
            "Mashed potatoes",
            "Yogurt"
        ],
        "food_avoid": [
            "Hard and crunchy foods",
            "Spicy foods",
            "Acidic foods (oranges, tomatoes)",
            "Very hot foods",
            "Rough textures"
        ],
        "steps": [
            "Get plenty of rest",
            "Gargle with warm salt water 3-4 times daily",
            "Drink warm fluids",
            "Use humidifier in room",
            "Avoid smoking and irritants",
            "Complete full course of antibiotics if prescribed"
        ],
        "warning": "See doctor if difficulty breathing, severe pain, or fever persists beyond 3 days",
        "emergency_signs": ["difficulty_breathing", "difficulty_swallowing", "excessive_drooling"]
    },

    "ear_infection": {
        "severity": "medium",
        "description": "Infection of the middle ear, common in children but can affect adults.",
        "medicine": [
            "Pain relievers (Paracetamol, Ibuprofen)",
            "Antibiotic ear drops",
            "Oral antibiotics (if bacterial)",
            "Decongestants"
        ],
        "food_take": [
            "Anti-inflammatory foods",
            "Omega-3 rich foods",
            "Fresh fruits and vegetables",
            "Warm soups"
        ],
        "food_avoid": [
            "Dairy (may increase mucus)",
            "Sugar and processed foods",
            "Allergenic foods if related to allergies"
        ],
        "steps": [
            "Apply warm compress to ear",
            "Keep head elevated",
            "Use prescribed ear drops correctly",
            "Avoid water in ear",
            "Do NOT insert objects in ear",
            "Complete antibiotic course"
        ],
        "warning": "Consult ENT specialist if recurrent infections or hearing loss",
        "emergency_signs": ["severe_pain", "discharge_from_ear", "hearing_loss", "dizziness"]
    },

    "pneumonia": {
        "severity": "high",
        "description": "Infection that inflames air sacs in one or both lungs.",
        "medicine": [
            "Antibiotics (prescribed based on type)",
            "Fever reducers",
            "Cough medicine",
            "Oxygen therapy (severe cases)"
        ],
        "food_take": [
            "Light, nutritious meals",
            "Warm soups and broths",
            "Fresh fruits high in Vitamin C",
            "Vegetables",
            "Plenty of fluids",
            "Protein-rich foods"
        ],
        "food_avoid": [
            "Dairy (if increasing mucus)",
            "Processed foods",
            "Sugar",
            "Alcohol",
            "Caffeine"
        ],
        "steps": [
            "Complete bed rest",
            "Take all prescribed antibiotics",
            "Drink 8-10 glasses of water daily",
            "Use humidifier",
            "Practice deep breathing exercises",
            "Monitor temperature",
            "Follow up with doctor"
        ],
        "warning": "Hospitalization may be needed for severe cases. Seek immediate care.",
        "emergency_signs": ["difficulty_breathing", "chest_pain", "confusion", "high_fever", "blue_lips"]
    },

    "hepatitis": {
        "severity": "high",
        "description": "Inflammation of the liver, caused by viral infection.",
        "medicine": [
            "Antiviral medications (type-specific)",
            "Liver support supplements",
            "Pain relievers (avoid acetaminophen)",
            "Rest and hydration"
        ],
        "food_take": [
            "Easily digestible foods",
            "Fresh fruits and vegetables",
            "Whole grains",
            "Lean proteins",
            "Plenty of water",
            "Herbal teas"
        ],
        "food_avoid": [
            "Alcohol (strictly prohibited)",
            "Fatty and fried foods",
            "Raw or undercooked shellfish",
            "Processed foods",
            "Excessive sugar",
            "Salt"
        ],
        "steps": [
            "Complete rest",
            "Avoid alcohol completely",
            "Eat small, frequent meals",
            "Avoid medications affecting liver",
            "Practice safe sex",
            "Do not share personal items",
            "Regular liver function tests"
        ],
        "warning": "Chronic hepatitis requires long-term management. Consult hepatologist.",
        "emergency_signs": ["severe_abdominal_pain", "confusion", "jaundice_worsening", "bleeding"]
    },

    "malaria": {
        "severity": "critical",
        "description": "Life-threatening disease transmitted through mosquito bites.",
        "medicine": [
            "Artemisinin-based combination therapy (ACT)",
            "Chloroquine (if sensitive strain)",
            "Primaquine for radical cure",
            "Supportive care"
        ],
        "food_take": [
            "Easily digestible foods",
            "Plenty of fluids"
            "Fruits and vegetables",
            "Light meals"
        ],
        "food_avoid": [
            "Heavy meals",
            "Alcohol",
            "Caffeine"
        ],
        "steps": [
            "SEEK EMERGENCY MEDICAL CARE IMMEDIATELY",
            "Complete full course of treatment",
            "Use mosquito nets",
            "Apply insect repellent",
            "Eliminate standing water",
            "Wear protective clothing"
        ],
        "warning": "Malaria can be fatal within 24 hours. Immediate treatment is critical!",
        "emergency_signs": ["high_fever", "seizures", "difficulty_breathing", "organ_failure"]
    },

    "dengue": {
        "severity": "critical",
        "description": "Viral infection transmitted by Aedes mosquitoes.",
        "medicine": [
            "Paracetamol ONLY (avoid ibuprofen/aspirin)",
            "ORS for hydration",
            "IV fluids if severe",
            "Blood transfusion (if needed)"
        ],
        "food_take": [
            "Plenty of fluids (water, ORS, coconut water)",
            "Fruit juices",
            "Light foods",
            "Papaya leaf extract (traditional remedy)",
        ],
        "food_avoid": [
            "Ibuprofen and aspirin (risk of bleeding)",
            "NSAIDs",
            "Alcohol",
            "Caffeine",
            "Spicy foods"
        ],
        "steps": [
            "Rest completely",
            "Drink 5-6 liters of fluids daily",
            "Monitor platelet count",
            "Use mosquito nets to prevent spread",
            "Avoid mosquito bites",
            "Regular doctor visits"
        ],
        "warning": "Dengue can become severe (Dengue Hemorrhagic Fever). Watch for warning signs!",
        "emergency_signs": ["severe_abdominal_pain", "bleeding", "vomiting_blood", "difficulty_breathing", "fatigue"]
    },

    "chikungunya": {
        "severity": "medium",
        "description": "Viral disease transmitted by mosquitoes causing severe joint pain.",
        "medicine": [
            "Paracetamol for fever and pain",
            "Anti-inflammatory drugs",
            "Rest",
            "Physical therapy for joints"
        ],
        "food_take": [
            "Anti-inflammatory foods",
            "Omega-3 rich foods",
            "Fresh fruits and vegetables",
            "Plenty of water"
        ],
        "food_avoid": [
            "Processed foods",
            "Sugar",
            "Alcohol",
            "Caffeine"
        ],
        "steps": [
            "Rest completely",
            "Take pain relievers as prescribed",
            "Apply cold compress to joints",
            "Gentle stretching exercises",
            "Use mosquito nets",
            "Drink plenty of fluids"
        ],
        "warning": "Joint pain may persist for months. Consult doctor if severe.",
        "emergency_signs": ["severe_joint_pain", "difficulty_moving", "high_fever"]
    },

    "covid19": {
        "severity": "high",
        "description": "Respiratory illness caused by coronavirus SARS-CoV-2.",
        "medicine": [
            "Paracetamol for fever",
            "Isolation and rest",
            "Oxygen therapy if needed",
            "Antiviral medications (as prescribed)",
            "Corticosteroids for severe cases"
        ],
        "food_take": [
            "Nutritious, easy-to-digest foods",
            "Hot soups and broths",
            "Fresh fruits and vegetables",
            "Protein-rich foods",
            "Plenty of fluids",
            "Honey and ginger"
        ],
        "food_avoid": [
            "Processed foods",
            "Sugar",
            "Alcohol",
            "Heavy meals"
        ],
        "steps": [
            "Isolate in separate room",
            "Wear mask at all times",
            "Monitor oxygen levels (pulse oximeter)",
            "Stay hydrated",
            "Track symptoms daily",
            "Seek help if oxygen drops below 94%",
            "Follow vaccination guidelines"
        ],
        "warning": "Seek immediate care if oxygen saturation drops or breathing difficulty increases.",
        "emergency_signs": ["shortness_of_breath", "chest_pain", "confusion", "blue_lips", "inability_to_awake"]
    },

    "high_blood_pressure": {
        "severity": "high",
        "description": "A condition in which the force of blood against artery walls is too high.",
        "medicine": [
            "ACE inhibitors",
            "Beta-blockers",
            "Calcium channel blockers",
            "Diuretics",
            "Lifestyle changes"
        ],
        "food_take": [
            "Fruits (bananas, berries)",
            "Vegetables (leafy greens)",
            "Whole grains",
            "Lean proteins",
            "Low-fat dairy",
            "Potassium-rich foods"
        ],
        "food_avoid": [
            "Salt and sodium",
            "Processed foods",
            "Fried foods",
            "Red meat",
            "Alcohol",
            "Caffeine",
            "Sugar"
        ],
        "steps": [
            "Check blood pressure regularly",
            "Exercise 30 minutes daily",
            "Maintain healthy weight",
            "Reduce salt intake",
            "Manage stress",
            "Quit smoking",
            "Limit alcohol",
            "Take medications as prescribed"
        ],
        "warning": "Uncontrolled hypertension leads to heart disease, stroke, kidney damage. Regular monitoring essential.",
        "emergency_signs": ["severe_headache", "chest_pain", "difficulty_breathing", "vision_problems", "nosebleed"]
    },

    "meningitis": {
        "severity": "critical",
        "description": "Inflammation of the membranes surrounding the brain and spinal cord.",
        "medicine": [
            "IV antibiotics (bacterial)",
            "IV corticosteroids",
            "Antiviral medications (viral)",
            "Supportive care"
        ],
        "food_take": [
            "Easily digestible foods",
            "Light meals",
            "Plenty of fluids",
            "Nutritious soups"
        ],
        "food_avoid": [
            "Heavy meals",
            "Difficult to digest foods"
        ],
        "steps": [
            "SEEK EMERGENCY MEDICAL CARE IMMEDIATELY",
            "Do NOT wait for symptoms to worsen",
            "Hospitalization required",
            "Complete all prescribed treatment",
            "Rest during recovery"
        ],
        "warning": "Meningitis can be fatal within hours. This is a medical emergency!",
        "emergency_signs": ["high_fever", "stiff_neck", "severe_headache", "confusion", "seizures", "sensitivity_to_light"]
    }
}

# ===============================
# VOICE ENGINE SETUP
# ===============================
class VoiceEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Configure voice
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)  # Default voice
        self.engine.setProperty('rate', 150)  # Speaking rate
        self.engine.setProperty('volume', 1.0)  # Volume level

    def speak(self, text):
        """Convert text to speech"""
        print(f"\n🤖 AI: {text}\n")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Convert speech to text"""
        with sr.Microphone() as source:
            print("\n🎤 Listening... (Speak your symptoms)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                print(f"👤 You said: {text}")
                return text.lower()
            except sr.UnknownValueError:
                print("❌ Sorry, I couldn't understand. Please try again.")
                return None
            except sr.RequestError:
                print("❌ Speech service is unavailable.")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                return None

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_available_symptoms():
    """Get list of all symptoms the model can recognize"""
    all_symptoms = sorted(set(
        symptom.strip() 
        for symptoms in open("data/symptoms_dataset.csv").read().split("\n")[1:] 
        for symptom in symptoms.split(",")
        if symptom.strip()
    ))
    return all_symptoms

def display_symptom_list():
    """Display all available symptoms"""
    symptoms = get_available_symptoms()
    print("\n" + "="*70)
    print("📋 LIST OF RECOGNIZABLE SYMPTOMS (comma-separated):")
    print("="*70)
    for i in range(0, len(symptoms), 4):
        row = symptoms[i:i+4]
        print("  " + " | ".join(f"{s:25}" for s in row))
    print("="*70)

def predict_disease(symptoms_text):
    """Predict disease from symptoms text"""
    symptoms = symptoms_text.replace(" ", "").split(",")
    symptoms = [s.strip() for s in symptoms if s.strip()]
    
    try:
        vector = mlb.transform([symptoms])
        disease = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()
        return disease, confidence, symptoms
    except Exception as e:
        return None, 0.0, symptoms

def display_advice(disease):
    """Display detailed health advice"""
    advice = health_advice.get(disease, {})
    
    if not advice:
        print(f"\n⚠️ No detailed advice available for {disease}")
        return

    # Severity indicator
    severity_colors = {
        "low": "🟢 LOW",
        "medium": "🟡 MEDIUM",
        "high": "🟠 HIGH",
        "critical": "🔴 CRITICAL"
    }
    severity = advice.get("severity", "unknown")
    
    print("\n" + "="*70)
    print(f"🏥 DIAGNOSIS: {disease.upper().replace('_', ' ')}")
    print(f"📊 SEVERITY: {severity_colors.get(severity, severity)}")
    print("="*70)
    
    print(f"\n📝 Description: {advice.get('description', 'N/A')}")
    
    print("\n" + "-"*70)
    print("💊 RECOMMENDED MEDICATIONS:")
    print("-"*70)
    for med in advice.get("medicine", []):
        print(f"  • {med}")
    
    print("\n" + "-"*70)
    print("🥗 FOOD TO EAT:")
    print("-"*70)
    for food in advice.get("food_take", []):
        print(f"  ✓ {food}")
    
    print("\n" + "-"*70)
    print("🚫 FOOD TO AVOID:")
    print("-"*70)
    for food in advice.get("food_avoid", []):
        print(f"  ✗ {food}")
    
    print("\n" + "-"*70)
    print("📋 STEPS TO FOLLOW:")
    print("-"*70)
    for step in advice.get("steps", []):
        print(f"  → {step}")
    
    print("\n" + "-"*70)
    print("⚠️ WARNING:")
    print("-"*70)
    print(f"  {advice.get('warning', 'Consult a doctor for proper diagnosis')}")
    
    # Emergency signs
    emergency = advice.get("emergency_signs", [])
    if emergency:
        print("\n" + "-"*70)
        print("🚨 EMERGENCY SIGNS TO WATCH FOR:")
        print("-"*70)
        for sign in emergency:
            print(f"  ⚡ {sign.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    print("❗ DISCLAIMER: This is an AI-generated suggestion. Please consult")
    print("   a qualified healthcare professional for proper diagnosis and treatment.")
    print("="*70)

# ===============================
# MAIN CHATBOT
# ===============================
def main():
    voice = VoiceEngine()
    
    print("\n" + "="*70)
    print("🩺 ADVANCED AI HEALTHCARE CHATBOT")
    print("="*70)
    print("Features: Voice Input | Disease Prediction | Health Advice")
    print("="*70)
    
    voice.speak("Welcome to the AI Healthcare Chatbot. How can I help you today?")
    
    while True:
        print("\n" + "-"*70)
        print("📋 MAIN MENU")
        print("-"*70)
        print("  1. 🔤 Type symptoms")
        print("  2. 🎤 Speak symptoms")
        print("  3. 📋 View all symptoms")
        print("  4. ℹ️ About this chatbot")
        print("  5. 🚪 Exit")
        print("-"*70)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "5" or choice.lower() in ["exit", "quit"]:
            print("\n🙏 Thank you for using Healthcare Chatbot. Stay healthy!")
            voice.speak("Thank you for using Healthcare Chatbot. Stay healthy and take care!")
            break
        
        elif choice == "1":
            print("\n📝 Enter your symptoms separated by commas:")
            print("   Example: fever, cough, headache")
            user_input = input("Symptoms: ").strip().lower()
            
            if not user_input:
                print("❌ No symptoms entered. Please try again.")
                continue
            
            disease, confidence, symptoms = predict_disease(user_input)
            
            if disease:
                print(f"\n🎯 Predicted Disease: {disease.upper().replace('_', ' ')}")
                print(f"📊 Confidence: {round(confidence * 100, 2)}%")
                print(f"📝 Symptoms Analyzed: {', '.join(symptoms)}")
                
                display_advice(disease)
            else:
                print("❌ Could not identify disease from symptoms.")
                print("💡 Try using different symptom combinations.")
        
        elif choice == "2":
            user_input = voice.listen()
            
            if user_input:
                disease, confidence, symptoms = predict_disease(user_input)
                
                if disease:
                    print(f"\n🎯 Predicted Disease: {disease.upper().replace('_', ' ')}")
                    print(f"📊 Confidence: {round(confidence * 100, 2)}%")
                    
                    display_advice(disease)
                    
                    # Speak the diagnosis
                    voice.speak(f"Based on your symptoms, you may have {disease.replace('_', ' ')}.")
                    voice.speak(health_advice[disease]["warning"])
                else:
                    print("❌ Could not identify disease from symptoms.")
                    voice.speak("I couldn't identify a disease from your symptoms. Please try again.")
        
        elif choice == "3":
            display_symptom_list()
        
        elif choice == "4":
            print("\n" + "="*70)
            print("ℹ️ ABOUT THIS CHATBOT")
            print("="*70)
            print("""
  🩺 AI Healthcare Chatbot
  
  This chatbot uses Machine Learning to predict possible diseases
  based on your symptoms and provides health advice.
  
  Features:
  - Voice Recognition (Speech-to-Text)
  - Text-to-Speech Response
  - ML-Based Disease Prediction
  - Detailed Health Advice
  - Emergency Detection
  
  How to Use:
  - Type or speak your symptoms
  - Separate multiple symptoms with commas
  - Get instant diagnosis and advice
  
  ⚠️ IMPORTANT: This is NOT a replacement for professional
     medical advice. Always consult a doctor for serious conditions.
  
  Created with ❤️ using Python, Scikit-learn, and SpeechRecognition.
            """)
            print("="*70)
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot closed by user. Stay healthy!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Please make sure the model is trained and all dependencies are installed.")