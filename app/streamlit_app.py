"""Streamlit dashboard for the student placement predictor.

Run with:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from placement.config import MODEL_PATH, REPORTS_DIR  # noqa: E402
from placement.predict import PlacementPredictor  # noqa: E402

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_predictor() -> PlacementPredictor:
    return PlacementPredictor(MODEL_PATH)


st.title("🎓 Student Placement Predictor")
st.caption(
    "Estimates placement likelihood from academic, skill, and engagement data. "
    "Intended for counselling and preparation planning, not for admission or hiring decisions."
)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m placement.train` first.")
    st.stop()

predictor = load_predictor()

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profile & academics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EE", "ME", "CE"])
        extracurricular = st.radio("Extracurricular activities", ["Yes", "No"], horizontal=True)
        cgpa = st.slider("CGPA (10-point scale)", 4.5, 10.0, 7.5, 0.01)
        tenth = st.number_input("Class X percentage", 40.0, 100.0, 85.0, 0.1)
        twelfth = st.number_input("Class XII percentage", 40.0, 100.0, 80.0, 0.1)
        backlogs = st.number_input("Active backlogs", min_value=0, max_value=8, value=0, step=1)

    with col2:
        st.subheader("Experience")
        internships = st.number_input("Internships completed", 0, 3, 1, 1)
        projects = st.number_input("Projects completed", 0, 10, 2, 1)
        certifications = st.number_input("Certifications", 0, 10, 1, 1)
        hackathons = st.number_input("Hackathons attended", 0, 5, 0, 1)

    with col3:
        st.subheader("Skills & habits")
        coding = st.slider("Coding assessment score", 0.0, 100.0, 60.0, 0.5)
        aptitude = st.slider("Aptitude score", 0.0, 100.0, 60.0, 0.5)
        communication = st.slider("Communication score", 0.0, 100.0, 62.0, 0.5)
        study_hours = st.number_input("Self-study hours per day", 0.0, 12.0, 3.5, 0.5)
        attendance = st.slider("Attendance (%)", 30.0, 100.0, 80.0, 0.5)

    submitted = st.form_submit_button("Predict placement", type="primary")

if submitted:
    student = {
        "gender": gender,
        "branch": branch,
        "extracurricular": extracurricular,
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "internships": internships,
        "projects": projects,
        "certifications": certifications,
        "hackathons": hackathons,
        "coding_score": coding,
        "aptitude_score": aptitude,
        "communication_score": communication,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,
    }
    result = predictor.predict_one(student)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Placement probability", f"{result['placement_probability'] * 100:.1f}%")
    m2.metric("Predicted outcome", "Placed" if result["prediction"] else "Not placed")
    m3.metric("Confidence band", result["risk_level"])
    st.progress(float(result["placement_probability"]))

    st.subheader("Recommendations")
    if result["recommendations"]:
        for tip in result["recommendations"]:
            st.write(f"- {tip}")
    else:
        st.success("Strong profile across all tracked areas. Keep it up.")

with st.expander("Model details and feature importance"):
    st.write(f"**Model:** {predictor.metadata['model_name']}")
    st.write(f"**Trained at:** {predictor.metadata['trained_at']}")
    importance_path = REPORTS_DIR / "permutation_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        st.bar_chart(importance.set_index("feature")["importance_mean"])
    else:
        st.info("Feature importance report not found.")
