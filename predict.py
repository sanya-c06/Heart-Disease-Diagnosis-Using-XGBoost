import xgboost as xgb
import pandas as pd

# 1. Initialize and load your saved AI "Brain"
print("Loading the AI Diagnostic Model...")
model = xgb.XGBClassifier()
model.load_model("chd_xgboost_model.json")

def diagnose_patient(patient_data):
    # 2. Convert our dictionary into a Pandas DataFrame (1 row of data)
    # The columns MUST match the exact order and spelling from your preprocessing phase
    df = pd.DataFrame([patient_data])
    
    # 3. Ask the AI for a diagnosis
    # predict() gives a 0 or 1. predict_proba() gives the actual percentage of certainty.
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] * 100 
    
    # 4. Print the results nicely
    print("\n--- Diagnostic Results ---")
    if prediction == 1:
        print(f"HIGH RISK: The model predicts Coronary Heart Disease (Confidence: {probability:.1f}%)")
    else:
        print(f"LOW RISK: No Coronary Heart Disease detected (Confidence: {(100 - probability):.1f}%)")
    print("--------------------------\n")

# ==========================================
# Let's test it on two fake patients!
# ==========================================

# Patient A: A 65-year-old male smoker with high blood pressure, high cholesterol, and diabetes.
patient_a = {
    'IMF_Energy': 0.85, 'IMF_Amplitude': 1.5, 'IMF_Frequency': 0.28, 
    'HRV': 60,          # Lower Heart Rate Variability
    'BP_Systolic': 160, # High Blood Pressure
    'BP_Diastolic': 100, 
    'Age': 65,          # Older
    'Cholesterol': 280, # High Cholesterol
    'Diabetes': 1,      # 1 = Yes
    'Smoking': 1,       # 1 = Yes
    'Sex_M': 1          # 1 = Male
}

# Patient B: A 32-year-old female non-smoker with normal vitals.
patient_b = {
    'IMF_Energy': 0.90, 'IMF_Amplitude': 1.2, 'IMF_Frequency': 0.22, 
    'HRV': 95,          # Healthy Heart Rate Variability
    'BP_Systolic': 115, # Normal Blood Pressure
    'BP_Diastolic': 75, 
    'Age': 32,          # Younger
    'Cholesterol': 170, # Normal Cholesterol
    'Diabetes': 0,      # 0 = No
    'Smoking': 0,       # 0 = No
    'Sex_M': 0          # 0 = Female
}

print("\nEvaluating Patient A (Expected: High Risk)...")
diagnose_patient(patient_a)

print("Evaluating Patient B (Expected: Low Risk)...")
diagnose_patient(patient_b)