import os
import tempfile

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

@st.cache_resource
def load_cough_predictor():
    # Heavy (torch + wav2vec2), so it is loaded only when a cough file is uploaded.
    # Returns (function, error_message).
    try:
        from cough_audio_module.cough_module import predict_cough
        return predict_cough, None
    except Exception as e:
        return None, str(e)

xray_model = load_xray_model()
clinical_model, clinical_scaler, clinical_features = load_clinical_model()

# --- Fusion Logic ---
# Cough gets a small weight: on its own it is a weak signal (patient-level AUC ~0.69).
WEIGHTS = {"xray": 0.6, "clinical": 0.4, "cough": 0.1}

def combined_prediction(xray_prob=None, clinical_prob=None, cough_prob=None):
    inputs = {"xray": xray_prob, "clinical": clinical_prob, "cough": cough_prob}
    available = {k: v for k, v in inputs.items() if v is not None}

    if not available:
        return None, "No input provided"

    total_weight = sum(WEIGHTS[k] for k in available)
    final_score = sum(v * WEIGHTS[k] for k, v in available.items()) / total_weight
    label = "TB Positive" if final_score > 0.5 else "Normal"
    return round(float(final_score), 3), label

# --- UI ---
st.title("🩺 Tuberculosis Prediction System")
st.write("Upload a chest X-ray, enter symptom details and/or upload a cough recording to get a TB risk prediction. You can provide any of them.")

xray_prob = None
clinical_prob = None
cough_prob = None

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

# --- Cough Section ---
st.subheader("3. Cough Audio (optional)")
cough_files = st.file_uploader(
    "Upload one or more cough recordings (same patient)",
    type=['wav', 'mp3', 'ogg', 'flac'],
    accept_multiple_files=True,
)
if cough_files:
    predict_cough, cough_error = load_cough_predictor()
    if predict_cough is None:
        st.warning(f"Cough model could not be loaded: {cough_error}")
    else:
        tmp_paths = []
        try:
            for f in cough_files:
                suffix = os.path.splitext(f.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getbuffer())
                    tmp_paths.append(tmp.name)
            with st.spinner("Analysing cough..."):
                cough_prob = float(predict_cough(tmp_paths))
            st.write(f"**Cough model risk score:** {cough_prob:.2f}")
            st.caption("Cough alone is a weak TB signal (research result: patient-level AUC ~0.69), so it has a small weight in the final score.")
        except Exception as e:
            st.warning(f"Could not process the cough audio: {e}")
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)

# --- Final Combined Prediction ---
st.subheader("Final Prediction")
if st.button("Get Combined Prediction"):
    final_score, label = combined_prediction(xray_prob, clinical_prob, cough_prob)
    if final_score is not None:
        if xray_prob is None and clinical_prob is None and cough_prob is not None:
            st.warning("Only cough audio was provided. This is a low-confidence result; add an X-ray or symptoms for a more reliable prediction.")
        if label == "TB Positive":
            st.error(f"**Result: {label}**  (confidence score: {final_score})")
        else:
            st.success(f"**Result: {label}**  (confidence score: {final_score})")
    else:
        st.warning("Please upload an X-ray, enter symptom data or upload a cough recording first.")

st.caption("⚠️ This is a student/research project and NOT a certified medical diagnostic tool. Please consult a doctor for actual diagnosis.")        
