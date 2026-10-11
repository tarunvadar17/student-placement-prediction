"""Inference API: single-student prediction and improvement recommendations."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import ALL_FEATURES, MODEL_PATH
from .preprocessing import validate_schema


def risk_level(probability: float) -> str:
    """Map a placement probability to a human-readable confidence band."""
    if probability >= 0.75:
        return "Strong placement likelihood"
    if probability >= 0.50:
        return "Moderate placement likelihood"
    if probability >= 0.25:
        return "Low placement likelihood"
    return "Very low placement likelihood"


def get_recommendations(student: dict) -> list[str]:
    """Return rule-based, actionable suggestions based on the student's profile."""
    tips: list[str] = []
    if student.get("backlogs", 0) > 0:
        tips.append("Clear active backlogs; most recruiters screen them out early.")
    if student.get("cgpa", 10) < 7.0:
        tips.append("Raise your CGPA where possible, since many companies apply a cut-off.")
    if student.get("coding_score", 100) < 65:
        tips.append("Practise data structures and algorithms on a regular schedule.")
    if student.get("aptitude_score", 100) < 60:
        tips.append("Work through timed aptitude tests to improve speed and accuracy.")
    if student.get("communication_score", 100) < 60:
        tips.append("Take part in mock interviews and group discussions.")
    if student.get("internships", 0) == 0:
        tips.append("Pursue at least one internship to demonstrate practical experience.")
    if student.get("projects", 0) < 2:
        tips.append("Build two or more substantial projects and host them publicly.")
    if student.get("certifications", 0) < 2:
        tips.append("Earn a relevant industry certification to strengthen your profile.")
    if student.get("hackathons", 0) == 0:
        tips.append("Participate in hackathons to build teamwork and problem-solving evidence.")
    if student.get("attendance_percentage", 100) < 75:
        tips.append("Improve attendance, as many institutions require a minimum threshold.")
    return tips


class PlacementPredictor:
    """Loads a trained model bundle and serves predictions."""

    def __init__(self, model_path: Path | str = MODEL_PATH) -> None:
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle["metadata"]
        self.threshold = self.metadata.get("decision_threshold", 0.5)

    def predict_one(self, student: dict) -> dict:
        """Predict placement for a single student record."""
        df = pd.DataFrame([student])
        validate_schema(df, require_target=False)
        df = df[ALL_FEATURES]

        probability = float(self.pipeline.predict_proba(df)[0, 1])
        return {
            "placement_probability": round(probability, 4),
            "prediction": int(probability >= self.threshold),
            "risk_level": risk_level(probability),
            "recommendations": get_recommendations(student),
        }
