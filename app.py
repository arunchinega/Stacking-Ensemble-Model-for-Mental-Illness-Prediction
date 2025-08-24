import streamlit as st
import pandas as pd
import joblib
import warnings

# --- MUST be the first Streamlit command ---
st.set_page_config(page_title="Mental Illness Prediction App", layout="wide")

warnings.filterwarnings('ignore')

# --- Load Model and Features ---
try:
    best_model = joblib.load('best_model.joblib')
    selected_features = joblib.load('rfe_features.joblib')
    st.success("Model and features loaded successfully.")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'best_model.joblib' and 'rfe_features.joblib' exist.")
    st.stop()

# --- Preprocessing Function ---
def preprocess_input(user_input, selected_features):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    aligned_df = input_encoded.reindex(columns=selected_features, fill_value=0)
    return aligned_df

# --- Streamlit UI ---
st.title("Mental Illness Prediction App")
st.markdown("### Powered by a Stacking Ensemble Model")

st.markdown("""
This application predicts the presence of mental illness based on patient characteristics.
Please fill in the following details to get a prediction.
""")

# --- Input Section ---
st.subheader("Patient Characteristics")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Demographics & Social")
    religious_pref = st.selectbox(
        "Religious Preference",
        ['NO', 'YES'],
        key="religious_pref"
    )
    race = st.selectbox(
        "Race",
        ['WHITE ONLY', 'NON-WHITE'],
        key="race"
    )
    criminal_justice_status = st.selectbox(
        "Criminal Justice Status",
        ['YES', 'NO'],
        key="criminal_justice"
    )

with col2:
    st.write("Health & Disability")
    critical_clinical_record = st.selectbox(
        "Critical Clinical Record (indicator)*",
        ['YES', 'NO'],
        key="ccr",
        help="Indicates whether a critical clinical condition record is present. "
             "This factor was identified as important for improving predictions, "
             "especially for minority classes."
    )
    no_chronic_med_condition = st.selectbox(
        "No Chronic Medical Condition",
        ['YES', 'NO'],
        key="chronic_condition"
    )
    intellectual_disability_missing = st.selectbox(
        "Intellectual Disability Info (available or missing)*",
        ['YES', 'NO'],
        key="intellectual_missing"
    )
    autism_spectrum_missing = st.selectbox(
        "Autism Spectrum Info (available or missing)*",
        ['YES', 'NO'],
        key="autism_missing"
    )

with col3:
    st.write("Missing & Combined Features")
    principal_diagnosis_missing = st.selectbox(
        "Principal Diagnosis Information (available or missing)*",
        ['YES', 'NO'],
        key="diagnosis_missing",
        help="Indicates whether the principal diagnosis record was missing. "
             "Patterns of missingness are highly predictive in this dataset."
    )
    combined_diagnosis_education = st.selectbox(
        "Combined Diagnosis & Education",
        ['YES', 'NO'],
        key="combined_education"
    )

# --- Input Mapping ---
input_map = {
    'Religious Preference_I BELONG TO A FORMAL RELIGIOUS GROUP': religious_pref,
    'Race_WHITE ONLY': 'YES' if race == 'WHITE ONLY' else 'NO',
    'Criminal Justice Status_YES': criminal_justice_status,
    'Serious Mental Illness_YES': critical_clinical_record,  # kept model feature name, mapped to user-friendly label
    'No Chronic Med Condition_YES': no_chronic_med_condition,
    'Intellectual Disability_Missing': intellectual_disability_missing,
    'Autism Spectrum_Missing': autism_spectrum_missing,
    'Principal Diagnosis Class_Missing': principal_diagnosis_missing,
    'Combined_Diagnosis_Education_MENTAL ILLNESS_NOT APPLICABLE': combined_diagnosis_education,
}

# --- Prepare User Input ---
user_input_dict = {feature: 0 for feature in selected_features}
for key, value in input_map.items():
    if key in user_input_dict and value == 'YES':
        user_input_dict[key] = 1

# --- Prediction ---
if st.button("Get Prediction"):
    preprocessed_input = preprocess_input(user_input_dict, selected_features)
    prediction = best_model.predict(preprocessed_input)
    label_map = {0: 'NO', 1: 'UNKNOWN', 2: 'YES'}
    predicted_label = label_map.get(prediction[0], "Prediction Error")

    st.subheader("Prediction Result")
    st.write(f"Prediction: {predicted_label}")

    if predicted_label == 'YES':
        st.warning("The model predicts the presence of mental illness.")
    elif predicted_label == 'NO':
        st.success("The model predicts the absence of mental illness.")
    else:
        st.info("The model was unable to make a clear prediction.")
