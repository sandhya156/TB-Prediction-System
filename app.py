import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="TB Prediction System", layout="centered")

# --- Load Models (cached so they load only once) ---
@st.cache_resource
def load_xray_model():
    return load_model('xray_module/tb_xray_model.h5')

@st.cache_resource
def load_clinical_model():
    model = load_model('clinical_module/tb_clinical_model.h5')
    scaler = joblib.load('clinical_module/clinical_scaler.pkl')
    features = joblib.load('clinical_module/clinical_features.pkl')
    return model, scaler, features

xray_model = load_xray_model()
clinical_model, clinical_scaler, clinical_features = load_clinical_model()

# --- Fusion Logic ---
def combined_prediction(xray_prob=None, clinical_prob=None):
    probs = []
    weights = []

    if xray_prob is not None:
        probs.append(xray_prob)
        weights.append(0.6)   # X-ray is generally more reliable
    if clinical_prob is not None:
        probs.append(clinical_prob)
        weights.append(0.4)

    if not probs:
        return None, "No input provided"

    final_score = sum(p * w for p, w in zip(probs, weights)) / sum(weights)
    label = "TB Positive" if final_score > 0.5 else "Normal"
    return round(float(final_score), 3), label

# --- UI ---
st.title("🩺 Tuberculosis Prediction System")
st.write("Upload a chest X-ray and/or enter symptom details to get a TB risk prediction. You can provide either or both.")

xray_prob = None
clinical_prob = None

# --- X-ray Section ---
st.subheader("1. Chest X-ray (optional)")
xray_file = st.file_uploader("Upload chest X-ray image", type=['jpg', 'jpeg', 'png'])
if xray_file:
    img = image.load_img(xray_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    xray_prob = float(xray_model.predict(img_array)[0][0])
    st.image(img, caption="Uploaded X-ray", width=250)
    st.write(f"**X-ray model risk score:** {xray_prob:.2f}")

# --- Clinical Section ---
st.subheader("2. Symptoms & Clinical Data (optional)")
use_clinical = st.checkbox("Enter symptom data")

if use_clinical:
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Age", 0, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cough_duration = st.number_input("Cough duration (weeks)", 0, 52, 0)
        fever = st.selectbox("Fever?", ["No", "Yes"])
        weight_loss = st.selectbox("Weight loss?", ["No", "Yes"])
        night_sweats = st.selectbox("Night sweats?", ["No", "Yes"])
    with col2:
        chest_pain = st.selectbox("Chest pain?", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of breath?", ["No", "Yes"])
        hemoptysis = st.selectbox("Coughing blood (hemoptysis)?", ["No", "Yes"])
        smoking_status = st.selectbox("Smoking status?", ["No", "Yes"])
        bmi = st.number_input("BMI", 10.0, 45.0, 22.0)
        contact_tb = st.selectbox("Contact with TB patient?", ["No", "Yes"])

    input_dict = {
        'patient_age': patient_age,
        'gender': 1 if gender == "Male" else 0,
        'cough_duration': cough_duration,
        'fever': 1 if fever == "Yes" else 0,
        'weight_loss': 1 if weight_loss == "Yes" else 0,
        'night_sweats': 1 if night_sweats == "Yes" else 0,
        'chest_pain': 1 if chest_pain == "Yes" else 0,
        'shortness_breath': 1 if shortness_breath == "Yes" else 0,
        'hemoptysis': 1 if hemoptysis == "Yes" else 0,
        'smoking_status': 1 if smoking_status == "Yes" else 0,
        'bmi': bmi,
        'contact_tb': 1 if contact_tb == "Yes" else 0,
    }

    input_data = np.array([[input_dict[f] for f in clinical_features]])
    input_scaled = clinical_scaler.transform(input_data)
    clinical_prob = float(clinical_model.predict(input_scaled)[0][0])
    st.write(f"**Clinical model risk score:** {clinical_prob:.2f}")

# --- Final Combined Prediction ---
st.subheader("Final Prediction")
if st.button("Get Combined Prediction"):
    final_score, label = combined_prediction(xray_prob, clinical_prob)
    if final_score is not None:
        if label == "TB Positive":
            st.error(f"**Result: {label}**  (confidence score: {final_score})")
        else:
            st.success(f"**Result: {label}**  (confidence score: {final_score})")
    else:
        st.warning("Please upload an X-ray or enter symptom data first.")

st.caption("⚠️ This is a student/research project and NOT a certified medical diagnostic tool. Please consult a doctor for actual diagnosis.")
