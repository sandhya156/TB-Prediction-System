from pathlib import Path
import tempfile

import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TB Prediction System",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# ---------------- BLUE-WHITE DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #f3f7ff;
    color: #16345b;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1250px;
}
.hero {
    background: linear-gradient(120deg, #dceaff, #ffffff);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #d2e2ff;
    margin-bottom: 24px;
}
.hero h1 {
    color: #163c72;
    margin-bottom: 8px;
}
.hero p {
    color: #58749b;
    font-size: 16px;
}
.section {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #dce7f7;
    margin-bottom: 18px;
}
.result-card {
    background: white;
    border: 1px solid #dce7f7;
    border-radius: 14px;
    padding: 18px;
    min-height: 175px;
}
.score {
    color: #1769c2;
    font-size: 30px;
    font-weight: 700;
}
.small-note {
    color: #647c9c;
    font-size: 13px;
}
div.stButton > button {
    background: #1769c2;
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 45px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #104f96;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_xray_model():
    path = BASE_DIR / "xray_module" / "tb_xray_model.h5"
    return load_model(str(path))


@st.cache_resource
def load_clinical_model():
    folder = BASE_DIR / "clinical_module"

    model = load_model(str(folder / "tb_clinical_model.h5"))
    scaler = joblib.load(folder / "clinical_scaler.pkl")
    features = joblib.load(folder / "clinical_features.pkl")

    return model, scaler, features


@st.cache_resource
def load_sputum_model():
    path = BASE_DIR / "sputum_module" / "sputum_model_finetuned.keras"
    return load_model(str(path))


# Load cough model only when cough analysis is requested.
@st.cache_resource
def load_cough_predictor():
    import sys

    folder = str(BASE_DIR / "cough_audio_module")
    if folder not in sys.path:
        sys.path.insert(0, folder)

    from cough_module import predict_cough
    return predict_cough


# ---------------- PREDICTION FUNCTIONS ----------------
def predict_xray(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    score = float(load_xray_model().predict(arr, verbose=0)[0][0])
    return score, img


def predict_clinical(input_dict):
    model, scaler, features = load_clinical_model()

    values = np.array([[input_dict[f] for f in features]])
    scaled = scaler.transform(values)

    score = float(model.predict(scaled, verbose=0)[0][0])
    return score


def predict_sputum(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # Must match the preprocessing used during model training.
    score = float(load_sputum_model().predict(arr, verbose=0)[0][0])
    return score, img


def predict_cough(uploaded_file):
    predictor = load_cough_predictor()

    # The existing cough model accepts audio file paths.
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"

    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=True
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file.flush()

        score = float(predictor([temp_file.name]))

    return score


def result_label(score):
    return "TB-positive class" if score >= 0.5 else "Normal/negative class"


def show_result(title, score):
    st.markdown(f"""
    <div class="result-card">
        <h3>{title}</h3>
        <div class="score">{score * 100:.1f}%</div>
        <p>{result_label(score)}</p>
        <p class="small-note">
            Model output only — not a confirmed diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🩺 TB Prediction System</h1>
    <p>
        Multimodal research dashboard for chest X-ray, clinical data,
        sputum images and cough audio.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUTS ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📥 Prediction Center")
st.write("Upload one or more inputs. Then click Analyze to run the available models.")
st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🩻 1. Chest X-ray")
    xray_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        key="xray_upload"
    )
    if xray_file:
        st.image(xray_file, caption="Uploaded chest X-ray", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🔬 2. Sputum Image")
    sputum_file = st.file_uploader(
        "Upload sputum smear image",
        type=["jpg", "jpeg", "png"],
        key="sputum_upload"
    )
    if sputum_file:
        st.image(sputum_file, caption="Uploaded sputum image", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🎧 3. Cough Audio")
    st.write("Upload a cough recording for the existing audio model.")
    cough_file = st.file_uploader(
        "Upload cough audio (WAV recommended)",
        type=["wav"],
        key="cough_upload"
    )
    if cough_file:
        st.audio(cough_file, format="audio/wav")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🧬 4. Clinical Data")
    use_clinical = st.checkbox(
        "Enter clinical and symptom details",
        key="use_clinical"
    )

    clinical_values = None

    if use_clinical:
        with st.form("clinical_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 0, 120, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                cough_duration = st.number_input(
                    "Cough duration (weeks)", 0, 52, 0
                )
                fever = st.selectbox("Fever?", ["No", "Yes"])
                weight_loss = st.selectbox("Weight loss?", ["No", "Yes"])
                night_sweats = st.selectbox(
                    "Night sweats?", ["No", "Yes"]
                )

            with col2:
                chest_pain = st.selectbox("Chest pain?", ["No", "Yes"])
                breathlessness = st.selectbox(
                    "Shortness of breath?", ["No", "Yes"]
                )
                hemoptysis = st.selectbox(
                    "Coughing blood?", ["No", "Yes"]
                )
                smoking = st.selectbox("Smoking status?", ["No", "Yes"])
                bmi = st.number_input("BMI", 10.0, 45.0, 22.0)
                contact_tb = st.selectbox(
                    "Contact with TB patient?", ["No", "Yes"]
                )

            submitted = st.form_submit_button("Save Clinical Inputs")

        if submitted:
            clinical_values = {
                "patient_age": age,
                "gender": 1 if gender == "Male" else 0,
                "cough_duration": cough_duration,
                "fever": int(fever == "Yes"),
                "weight_loss": int(weight_loss == "Yes"),
                "night_sweats": int(night_sweats == "Yes"),
                "chest_pain": int(chest_pain == "Yes"),
                "shortness_breath": int(breathlessness == "Yes"),
                "hemoptysis": int(hemoptysis == "Yes"),
                "smoking_status": int(smoking == "Yes"),
                "bmi": bmi,
                "contact_tb": int(contact_tb == "Yes"),
            }
            st.session_state["clinical_values"] = clinical_values

        clinical_values = st.session_state.get("clinical_values")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- ANALYZE ----------------
st.markdown("---")

if st.button("🔍 Analyze Available Inputs", use_container_width=True):
    if not any([xray_file, sputum_file, cough_file, clinical_values]):
        st.warning("Please upload at least one file or save clinical inputs.")
    else:
        results = {}
        errors = {}

        with st.spinner("Analyzing available inputs..."):

            if xray_file:
                try:
                    results["Chest X-ray"] = predict_xray(xray_file)[0]
                except Exception as e:
                    errors["Chest X-ray"] = str(e)

            if sputum_file:
                try:
                    results["Sputum Image"] = predict_sputum(sputum_file)[0]
                except Exception as e:
                    errors["Sputum Image"] = str(e)

            if cough_file:
                try:
                    results["Cough Audio"] = predict_cough(cough_file)
                except Exception as e:
                    errors["Cough Audio"] = str(e)

            if clinical_values:
                try:
                    results["Clinical Data"] = predict_clinical(clinical_values)
                except Exception as e:
                    errors["Clinical Data"] = str(e)

        st.session_state["prediction_results"] = results
        st.session_state["prediction_errors"] = errors


# ---------------- DISPLAY RESULTS ----------------
results = st.session_state.get("prediction_results", {})
errors = st.session_state.get("prediction_errors", {})

if results or errors:
    st.markdown("---")
    st.header("📊 Individual Model Results")

    if results:
        cols = st.columns(min(2, len(results)))

        for index, (name, score) in enumerate(results.items()):
            with cols[index % len(cols)]:
                show_result(name, score)

    if errors:
        st.subheader("⚠️ Modules needing attention")
        for name, error in errors.items():
            st.error(f"{name}: {error}")

    if "Chest X-ray" in results and "Clinical Data" in results:
        st.markdown("---")
        st.subheader("🎯 Combined X-ray + Clinical Score")

        combined = (
            0.6 * results["Chest X-ray"]
            + 0.4 * results["Clinical Data"]
        )

        st.metric("Combined model score", f"{combined * 100:.1f}%")
        st.write(f"Model output: **{result_label(combined)}**")
        st.caption(
            "This is an experimental weighted score, not a clinically "
            "validated probability. Sputum and cough are shown separately."
        )


# ---------------- DISCLAIMER ----------------
st.markdown("---")
st.warning(
    "Educational research project only. These outputs are not a medical "
    "diagnosis. Do not use this application to make treatment decisions. "
    "Consult a qualified healthcare professional for TB testing."
)