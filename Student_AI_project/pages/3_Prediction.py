import os
import pickle

import streamlit as st

from utils.preprocessing import prepare_single_input, score_to_grade, score_to_pass_fail
from database.database import save_prediction

st.set_page_config(page_title="Prediction", page_icon="🧮", layout="wide")
st.title("🧮 Predict Student Performance")

MODEL_PATH = "student_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("No trained model found. Run `python train_model.py` first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)
model = payload["model"]

st.write("Enter a student's details below to predict their final score using the "
         "trained Bagging ensemble model.")

with st.form("prediction_form"):
    name = st.text_input("Student name (optional)", value="")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("Attendance (%)", 40.0, 100.0, 80.0, 1.0)
        previous_score = st.slider("Previous exam score", 0.0, 100.0, 65.0, 1.0)
        sleep_hours = st.slider("Average sleep (hours)", 3.0, 10.0, 7.0, 0.5)
    with col2:
        parental_support = st.radio("Parental support", ["Low", "High"], index=1)
        extracurricular = st.radio("Extracurricular activities", ["No", "Yes"], index=1)
        internet_access = st.radio("Internet access at home", ["No", "Yes"], index=1)
        study_group = st.selectbox("Study group", ["A", "B", "C"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "parental_support": 1 if parental_support == "High" else 0,
        "extracurricular": 1 if extracurricular == "Yes" else 0,
        "internet_access": 1 if internet_access == "Yes" else 0,
        "study_group": study_group,
    }

    X = prepare_single_input(input_dict)
    predicted_score = float(model.predict(X)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))
    grade = score_to_grade(predicted_score)
    result = score_to_pass_fail(predicted_score)

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted final score", f"{predicted_score:.1f}")
    r2.metric("Grade", grade)
    r3.metric("Result", result)

    if result == "Pass":
        st.success(f"Predicted outcome: **{result}** (Grade {grade})")
    else:
        st.error(f"Predicted outcome: **{result}** (Grade {grade})")

    record = {**input_dict, "student_name": name or "Unnamed",
              "predicted_score": predicted_score, "grade": grade, "result": result}
    save_prediction(record)
    st.caption("This prediction has been saved to History.")
