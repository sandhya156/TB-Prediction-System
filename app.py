
from pathlib import Path
import tempfile
import sys

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
}
.hero p {
    color: #58749b;
    font-size: 16px;
}
.section, .result-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #dce7f7;
    margin-bottom: 18px;
}
.result-card {
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


@st.cache_resource
def load_cough_predictor():
    folder = str(BASE_DIR / "cough_audio_module")

    if folder not in sys.path:
        sys.path.insert(0, folder)

    from cough_module import predict_cough
    return predict_cough


# ---------------- PREDICTION FUNCTIONS ----------------
def predict_xray(uploaded_file):
    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    score = float(
        load_xray_model().predict(arr, verbose=0).reshape(-1)[0]
    )

    return score


def predict_clinical(input_dict):
    model, scaler, features = load_clinical_model()

    # Use the exact feature order expected by the saved model.
    missing = [f for f in features if f not in input_dict]

    if missing:
        raise ValueError(
            "Missing clinical features: " + ", ".join(missing)
        )

    values = np.array(
        [[input_dict[f] for f in features]],
        dtype=float
    )

    scaled = scaler.transform(values)

    score = float(
        model.predict(scaled, verbose=0).reshape(-1)[0]
    )

    return score


def predict_sputum(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # Confirm this preprocessing matches the sputum model's training.
    score = float(
        load_sputum_model().predict(arr, verbose=0).reshape(-1)[0]
    )

    return score


def predict_cough(uploaded_file):
    predictor = load_cough_predictor()
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=True
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file.flush()

        score = float(predictor([temp_file.name]))

    return score


# ---------------- DISPLAY HELPERS ----------------
def result_label(score):
    return (
        "TB-positive class"
        if score >= 0.5
        else "Normal/negative class"
    )


def show_result(title, score):
    st.markdown(f"""
    <div class="result-card">
        <h3>{title}</h3>
        <div class="score">{score * 100:.1f}%</div>
        <p>{result_label(score)}</p>
        <p class="small-note">
            Experimental model output — not a confirmed diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)


def is_valid_score(score):
    return (
        score is not None
        and np.isfinite(score)
        and 0.0 <= score <= 1.0
    )


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🩺 TB Prediction System</h1>
    <p>
        Multimodal research dashboard for chest X-ray, sputum images,
        cough audio and clinical data.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUTS ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📥 Prediction Center")
st.write(
    "Provide one or more inputs, then click Analyze. "
    "Only successfully processed inputs will be considered."
)
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
        st.image(
            xray_file,
            caption="Uploaded chest X-ray",
            width=250
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🔬 2. Sputum Image")

    sputum_file = st.file_uploader(
        "Upload sputum smear image",
        type=["jpg", "jpeg", "png"],
        key="sputum_upload"
    )

    if sputum_file:
        st.image(
            sputum_file,
            caption="Uploaded sputum image",
            width=250
        )

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🎧 3. Cough Audio")
    st.write("Upload a WAV recording for the existing cough model.")

    cough_file = st.file_uploader(
        "Upload cough audio",
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

    if not use_clinical:
        st.session_state.pop("clinical_values", None)

    if use_clinical:
        with st.form("clinical_form"):
            st.write("Fill in the required details.")

            col1, col2 = st.columns(2)

            with col1:
                age_text = st.text_input(
                    "Age in years",
                    placeholder="Enter age"
                )

                gender = st.selectbox(
                    "Gender",
                    ["Select...", "Male", "Female"]
                )

                duration_text = st.text_input(
                    "Cough duration (weeks)",
                    placeholder="Enter duration"
                )

                fever = st.selectbox(
                    "Fever?",
                    ["Select...", "No", "Yes"]
                )

                weight_loss = st.selectbox(
                    "Weight loss?",
                    ["Select...", "No", "Yes"]
                )

                night_sweats = st.selectbox(
                    "Night sweats?",
                    ["Select...", "No", "Yes"]
                )

            with col2:
                chest_pain = st.selectbox(
                    "Chest pain?",
                    ["Select...", "No", "Yes"]
                )

                breathlessness = st.selectbox(
                    "Shortness of breath?",
                    ["Select...", "No", "Yes"]
                )

                hemoptysis = st.selectbox(
                    "Coughing blood?",
                    ["Select...", "No", "Yes"]
                )

                smoking = st.selectbox(
                    "Smoking status?",
                    ["Select...", "No", "Yes"]
                )

                bmi_text = st.text_input(
                    "BMI",
                    placeholder="Enter BMI"
                )

                contact_tb = st.selectbox(
                    "Contact with TB patient?",
                    ["Select...", "No", "Yes"]
                )

            submitted = st.form_submit_button(
                "Save Clinical Inputs"
            )

        if submitted:
            # Remove old values first, so invalid new inputs
            # cannot accidentally reuse a previous patient's data.
            st.session_state.pop("clinical_values", None)

            try:
                if not age_text.strip():
                    raise ValueError("Enter age.")

                if not duration_text.strip():
                    raise ValueError("Enter cough duration.")

                if not bmi_text.strip():
                    raise ValueError("Enter BMI.")

                age = float(age_text)
                duration = float(duration_text)
                bmi = float(bmi_text)

                if not 0 <= age <= 120:
                    raise ValueError("Age must be between 0 and 120.")

                if not 0 <= duration <= 52:
                    raise ValueError(
                        "Cough duration must be between 0 and 52 weeks."
                    )

                if not 10 <= bmi <= 45:
                    raise ValueError("BMI must be between 10 and 45.")

                choices = [
                    gender, fever, weight_loss, night_sweats,
                    chest_pain, breathlessness, hemoptysis,
                    smoking, contact_tb
                ]

                if any(value == "Select..." for value in choices):
                    raise ValueError(
                        "Select an option for every clinical question."
                    )

                clinical_values = {
                    "patient_age": age,
                    "gender": 1 if gender == "Male" else 0,
                    "cough_duration": duration,
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
                st.success("Clinical inputs saved.")

            except ValueError as e:
                st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- ANALYZE ----------------
st.markdown("---")

if st.button(
    "🔍 Analyze Available Inputs",
    use_container_width=True
):
    clinical_values = st.session_state.get("clinical_values")

    if not any([
        xray_file,
        sputum_file,
        cough_file,
        clinical_values
    ]):
        st.warning(
            "Upload at least one file or save valid clinical inputs."
        )

    else:
        results = {}
        errors = {}

        with st.spinner("Analyzing available inputs..."):

            if xray_file:
                try:
                    score = predict_xray(xray_file)

                    if is_valid_score(score):
                        results["Chest X-ray"] = score
                    else:
                        raise ValueError("Model returned an invalid score.")

                except Exception as e:
                    errors["Chest X-ray"] = str(e)

            if sputum_file:
                try:
                    score = predict_sputum(sputum_file)

                    if is_valid_score(score):
                        results["Sputum Image"] = score
                    else:
                        raise ValueError("Model returned an invalid score.")

                except Exception as e:
                    errors["Sputum Image"] = str(e)

            if cough_file:
                try:
                    score = predict_cough(cough_file)

                    if is_valid_score(score):
                        results["Cough Audio"] = score
                    else:
                        raise ValueError("Model returned an invalid score.")

                except Exception as e:
                    errors["Cough Audio"] = str(e)

            if clinical_values:
                try:
                    score = predict_clinical(clinical_values)

                    if is_valid_score(score):
                        results["Clinical Data"] = score
                    else:
                        raise ValueError("Model returned an invalid score.")

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
        cols = st.columns(2)

        for index, (name, score) in enumerate(results.items()):
            with cols[index % 2]:
                show_result(name, score)

    if errors:
        st.subheader("⚠️ Modules needing attention")

        for name, error in errors.items():
            st.error(f"{name}: {error}")

    # ---------------- COMBINED DEMO SCORE ----------------
    available_scores = {
        name: score
        for name, score in results.items()
        if is_valid_score(score)
    }

    st.markdown("---")
    st.subheader("🎯 Combined Score — Available Models")

    if len(available_scores) >= 2:
        # Equal-weight arithmetic average for software demonstration only.
        # It is NOT a clinically validated fusion model.
        combined = float(
            np.mean(list(available_scores.values()))
        )

        st.metric(
            "Experimental combined score",
            f"{combined * 100:.1f}%"
        )

        st.write("Successfully included models:")

        for model_name in available_scores:
            st.write(f"✓ {model_name}")

        st.write(
            f"Experimental model output: **{result_label(combined)}**"
        )

        st.caption(
            "This is an unvalidated equal-weight average of model outputs, "
            "not a validated TB probability or diagnosis."
        )

    elif len(available_scores) == 1:
        st.info(
            "Only one model produced a score. A combined score "
            "requires at least two successful model outputs."
        )

    else:
        st.info("No valid model scores are available for combination.")


# ---------------- DISCLAIMER ----------------
st.markdown("---")

st.warning(
    "Educational research project only. These model outputs are not "
    "a medical diagnosis. The combined score is experimental and "
    "not clinically validated. Consult a qualified healthcare "
    "professional for TB testing and medical decisions."
)
