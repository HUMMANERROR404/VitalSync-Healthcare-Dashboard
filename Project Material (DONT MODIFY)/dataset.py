import pandas as pd
import numpy as np

# ==========================================================
# Heart Disease Risk Prediction Dataset Generator
# ==========================================================
# Author  : Mobeen Fatima
# Version : 1.0
# Dataset : Synthetic Heart Disease Dataset
# Records : 10,000
# ==========================================================

# ----------------------------------------------------------
# Random Seed
# ----------------------------------------------------------

np.random.seed(42)

# ----------------------------------------------------------
# Number of Patients
# ----------------------------------------------------------

NUM_PATIENTS = 10000
# ==========================================================
# Generate Patient Information
# ==========================================================

# -----------------------------
# Patient ID
# -----------------------------

patient_id = np.arange(1, NUM_PATIENTS + 1)

# -----------------------------
# Age
# -----------------------------

age = np.random.randint(18, 91, NUM_PATIENTS)

# -----------------------------
# Gender
# -----------------------------

gender = np.random.choice(
    ["Male", "Female"],
    size=NUM_PATIENTS,
    p=[0.52, 0.48]
)

# -----------------------------
# Height (cm)
# -----------------------------

height = []

for g in gender:

    if g == "Male":
        h = np.random.normal(175, 7)

    else:
        h = np.random.normal(162, 6)

    h = np.clip(h, 145, 200)

    height.append(round(h, 1))

# -----------------------------
# Weight (kg)
# -----------------------------

weight = []

for h in height:

    ideal_weight = (h - 100) * 0.9

    w = np.random.normal(ideal_weight, 12)

    w = np.clip(w, 40, 160)

    weight.append(round(w, 1))

# -----------------------------
# BMI
# -----------------------------

bmi = []

for h, w in zip(height, weight):

    bmi_value = w / ((h / 100) ** 2)

    bmi.append(round(bmi_value, 1))

# ==========================================================
# Create DataFrame
# ==========================================================

df = pd.DataFrame({

    "Patient_ID": patient_id,
    "Age": age,
    "Gender": gender,
    "Height_cm": height,
    "Weight_kg": weight,
    "BMI": bmi

})
# ==========================================================
# Lifestyle & Medical History
# ==========================================================

smoking = []
alcohol = []
physical_activity = []
stress_level = []
sleep_hours = []
family_history = []
diabetes = []
hypertension = []
for i in range(NUM_PATIENTS):

    age = df.loc[i, "Age"]
    bmi = df.loc[i, "BMI"]

    # -----------------------------
    # Smoking
    # -----------------------------
    smoke_prob = 0.18

    if age >= 50:
        smoke_prob += 0.10

    if bmi >= 30:
        smoke_prob += 0.05

    smoking.append(
        np.random.choice(
            ["Yes", "No"],
            p=[min(smoke_prob, 0.60), 1 - min(smoke_prob, 0.60)]
        )
    )

    # -----------------------------
    # Alcohol
    # -----------------------------
    alcohol.append(
        np.random.choice(
            ["Yes", "No"],
            p=[0.35, 0.65]
        )
    )

    # -----------------------------
    # Physical Activity
    # -----------------------------
    if bmi >= 30:

        activity = np.random.choice(
            ["Low", "Moderate", "High"],
            p=[0.55, 0.35, 0.10]
        )

    else:

        activity = np.random.choice(
            ["Low", "Moderate", "High"],
            p=[0.20, 0.55, 0.25]
        )

    physical_activity.append(activity)

    # -----------------------------
    # Stress Level
    # -----------------------------
    stress = np.random.choice(
        ["Low", "Medium", "High"],
        p=[0.30, 0.45, 0.25]
    )

    stress_level.append(stress)

    # -----------------------------
    # Sleep Hours
    # -----------------------------
    if stress == "High":

        sleep = np.random.randint(4, 7)

    elif stress == "Medium":

        sleep = np.random.randint(6, 8)

    else:

        sleep = np.random.randint(7, 10)

    sleep_hours.append(sleep)

    # -----------------------------
    # Family History
    # -----------------------------
    family_history.append(
        np.random.choice(
            ["Yes", "No"],
            p=[0.30, 0.70]
        )
    )

    # -----------------------------
    # Diabetes
    # -----------------------------
    diabetes_risk = 0.05

    if age >= 50:
        diabetes_risk += 0.15

    if bmi >= 30:
        diabetes_risk += 0.20

    diabetes.append(
        np.random.choice(
            ["Yes", "No"],
            p=[
                min(diabetes_risk, 0.80),
                1 - min(diabetes_risk, 0.80)
            ]
        )
    )

    # -----------------------------
    # Hypertension
    # -----------------------------
    bp_risk = 0.08

    if age >= 50:
        bp_risk += 0.20

    if bmi >= 30:
        bp_risk += 0.10

    hypertension.append(
        np.random.choice(
            ["Yes", "No"],
            p=[
                min(bp_risk, 0.80),
                1 - min(bp_risk, 0.80)
            ]
        )
    )
# ==========================================================
# Add Lifestyle Columns
# ==========================================================

df["Smoking"] = smoking
df["Alcohol"] = alcohol
df["Physical_Activity"] = physical_activity
df["Stress_Level"] = stress_level
df["Sleep_Hours"] = sleep_hours
df["Family_History"] = family_history
df["Diabetes"] = diabetes
df["Hypertension"] = hypertension
# ==========================================================
# Clinical Measurements
# ==========================================================

cholesterol = []
hdl = []
ldl = []
triglycerides = []
resting_bp = []
fasting_blood_sugar = []
max_heart_rate = []
oxygen_saturation = []
for i in range(NUM_PATIENTS):

    age = df.loc[i, "Age"]
    bmi = df.loc[i, "BMI"]
    smoker = df.loc[i, "Smoking"]
    diabetic = df.loc[i, "Diabetes"]
    high_bp = df.loc[i, "Hypertension"]

    # -------------------------
    # Total Cholesterol
    # -------------------------

    chol = np.random.randint(150, 220)

    if smoker == "Yes":
        chol += np.random.randint(10, 25)

    if diabetic == "Yes":
        chol += np.random.randint(15, 35)

    if age >= 55:
        chol += np.random.randint(10, 25)

    chol = min(chol, 350)

    cholesterol.append(chol)

    # -------------------------
    # HDL
    # -------------------------

    if smoker == "Yes":
        h = np.random.randint(30, 50)
    else:
        h = np.random.randint(45, 75)

    hdl.append(h)

    # -------------------------
    # LDL
    # -------------------------

    l = chol - h - np.random.randint(20, 40)

    l = max(50, min(l, 220))

    ldl.append(l)

    # -------------------------
    # Triglycerides
    # -------------------------

    tri = np.random.randint(80, 160)

    if diabetic == "Yes":
        tri += np.random.randint(30, 70)

    triglycerides.append(min(tri, 400))

    # -------------------------
    # Resting Blood Pressure
    # -------------------------

    bp = np.random.randint(100, 125)

    if high_bp == "Yes":
        bp += np.random.randint(15, 35)

    resting_bp.append(min(bp, 200))

    # -------------------------
    # Fasting Blood Sugar
    # -------------------------

    if diabetic == "Yes":
        fasting_blood_sugar.append("High")
    else:
        fasting_blood_sugar.append(
            np.random.choice(
                ["Normal", "High"],
                p=[0.90, 0.10]
            )
        )

    # -------------------------
    # Maximum Heart Rate
    # -------------------------

    hr = 220 - age + np.random.randint(-8, 9)

    hr = max(70, min(hr, 205))

    max_heart_rate.append(hr)

    # -------------------------
    # Oxygen Saturation
    # -------------------------

    oxygen = round(np.random.uniform(95.0, 100.0), 1)

    oxygen_saturation.append(oxygen)

# ==========================================================
# Add Clinical Columns
# ==========================================================

df["Cholesterol"] = cholesterol
df["HDL_Cholesterol"] = hdl
df["LDL_Cholesterol"] = ldl
df["Triglycerides"] = triglycerides
df["Resting_BP"] = resting_bp
df["Fasting_Blood_Sugar"] = fasting_blood_sugar
df["Max_Heart_Rate"] = max_heart_rate
df["Oxygen_Saturation"] = oxygen_saturation

# ==========================================================
# Cardiology Features & Target
# ==========================================================

resting_ecg = []
exercise_angina = []
st_depression = []
chest_pain_type = []
daily_steps = []
water_intake = []
medication = []
heart_disease = []
for i in range(NUM_PATIENTS):

    age = df.loc[i, "Age"]
    smoker = df.loc[i, "Smoking"]
    diabetic = df.loc[i, "Diabetes"]
    hypertension = df.loc[i, "Hypertension"]
    activity = df.loc[i, "Physical_Activity"]
    stress = df.loc[i, "Stress_Level"]
    sleep = df.loc[i, "Sleep_Hours"]

    chol = df.loc[i, "Cholesterol"]
    bp = df.loc[i, "Resting_BP"]
    hr = df.loc[i, "Max_Heart_Rate"]

    # -------------------------
    # Resting ECG
    # -------------------------

    if hypertension == "Yes":

        ecg = np.random.choice(
            [
                "Normal",
                "ST-T Abnormality",
                "Left Ventricular Hypertrophy"
            ],
            p=[0.45,0.30,0.25]
        )

    else:

        ecg = np.random.choice(
            [
                "Normal",
                "ST-T Abnormality",
                "Left Ventricular Hypertrophy"
            ],
            p=[0.80,0.15,0.05]
        )

    resting_ecg.append(ecg)

    # -------------------------
    # Exercise Angina
    # -------------------------

    risk = 0.08

    if age >= 55:
        risk += 0.15

    if smoker == "Yes":
        risk += 0.10

    if hypertension == "Yes":
        risk += 0.15

    angina = np.random.choice(
        ["Yes","No"],
        p=[min(risk,0.80),1-min(risk,0.80)]
    )

    exercise_angina.append(angina)

    # -------------------------
    # ST Depression
    # -------------------------

    if angina == "Yes":
        st = round(np.random.uniform(1.0,4.5),1)
    else:
        st = round(np.random.uniform(0.0,2.0),1)

    st_depression.append(st)

    # -------------------------
    # Chest Pain
    # -------------------------

    if angina == "Yes":

        chest = np.random.choice(
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ],
            p=[0.45,0.20,0.15,0.20]
        )

    else:

        chest = np.random.choice(
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ],
            p=[0.10,0.25,0.50,0.15]
        )

    chest_pain_type.append(chest)

    # -------------------------
    # Daily Steps
    # -------------------------

    if activity == "High":

        steps = np.random.randint(9000,15000)

    elif activity == "Moderate":

        steps = np.random.randint(5000,9000)

    else:

        steps = np.random.randint(1000,5000)

    daily_steps.append(steps)

    # -------------------------
    # Water Intake
    # -------------------------

    water = round(np.random.uniform(1.5,4.0),1)

    water_intake.append(water)

    # -------------------------
    # Medication
    # -------------------------

    med = "No"

    if diabetic == "Yes" or hypertension == "Yes" or age >= 60:
        med = np.random.choice(
            ["Yes","No"],
            p=[0.75,0.25]
        )

    medication.append(med)

    # -------------------------
    # Heart Disease Target
    # -------------------------

    probability = 0.02

    if age >= 60:
        probability += 0.15

    if smoker == "Yes":
        probability += 0.10

    if diabetic == "Yes":
        probability += 0.12

    if hypertension == "Yes":
        probability += 0.12

    if chol >= 240:
        probability += 0.10

    if bp >= 140:
        probability += 0.10

    if angina == "Yes":
        probability += 0.15

    if st >= 2:
        probability += 0.08

    if activity == "Low":
        probability += 0.05

    if stress == "High":
        probability += 0.03

    if sleep <= 5:
        probability += 0.03

    probability = min(probability,0.95)

    target = np.random.choice(
        [1,0],
        p=[probability,1-probability]
    )

    heart_disease.append(target)

df["Resting_ECG"] = resting_ecg
df["Exercise_Angina"] = exercise_angina
df["ST_Depression"] = st_depression
df["Chest_Pain_Type"] = chest_pain_type
df["Daily_Steps"] = daily_steps
df["Water_Intake"] = water_intake
df["Medication"] = medication
df["Heart_Disease"] = heart_disease
# ==========================================================
# Preview Dataset
# ==========================================================

print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================================================
# Save Dataset
# ==========================================================

output_file = "heart_disease_risk_prediction.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("Heart Disease Dataset Created Successfully!")
print("=" * 60)

print(f"File Name : {output_file}")
print(f"Rows      : {df.shape[0]}")
print(f"Columns   : {df.shape[1]}")

print("=" * 60)

# ==========================================================
# Dataset Information
# ==========================================================

print("\nDataset Shape")
print(df.shape)

print("\nFirst Five Records")
print(df.head())

print("\nLast Five Records")
print(df.tail())

# ==========================================================
# Target Distribution
# ==========================================================

print("\nHeart Disease Distribution")

print(df["Heart_Disease"].value_counts())

print("\nHeart Disease Percentage")

print(
    round(
        df["Heart_Disease"].value_counts(normalize=True) * 100,
        2
    )
)
# ==========================================================
# Data Quality
# ==========================================================

print("\nMissing Values")

print(df.isnull().sum())

print("\nDuplicate Rows")

print(df.duplicated().sum())

print("\nDuplicate Patient IDs")

print(df["Patient_ID"].duplicated().sum())
# ==========================================================
# Data Types
# ==========================================================

print("\nColumn Data Types")

print(df.dtypes)
# ==========================================================
# Summary Statistics
# ==========================================================

print("\nSummary Statistics")

print(df.describe(include="all"))