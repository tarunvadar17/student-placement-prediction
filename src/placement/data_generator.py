"""Synthetic student dataset generator.

Generates plausible relationships between student attributes and placement
outcomes so the full pipeline can run without institutional data. The
generated data is for demonstration only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

BRANCH_EFFECT = {"CSE": 0.8, "IT": 0.6, "ECE": 0.3, "EE": 0.1, "ME": -0.1, "CE": -0.2}
BRANCH_PROBS = [0.30, 0.15, 0.15, 0.10, 0.15, 0.15]


def generate_student_data(n_samples: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a DataFrame of synthetic student records with a binary `placed` target."""
    rng = np.random.default_rng(seed)
    n = n_samples

    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.55, 0.42, 0.03])
    branch = rng.choice(list(BRANCH_EFFECT), size=n, p=BRANCH_PROBS)
    extracurricular = rng.choice(["Yes", "No"], size=n, p=[0.40, 0.60])

    cgpa = np.clip(rng.normal(7.2, 0.9, n), 4.5, 10.0)
    tenth = np.clip(cgpa * 9.5 + rng.normal(0, 4, n), 45, 100)
    twelfth = np.clip(tenth + rng.normal(-2, 6, n), 40, 100)
    backlogs = np.clip(rng.poisson(0.4, n), 0, 8)
    internships = rng.choice([0, 1, 2, 3], size=n, p=[0.40, 0.35, 0.18, 0.07])
    projects = np.clip(rng.poisson(2.0, n), 0, 10)
    certifications = np.clip(rng.poisson(1.5, n), 0, 10)
    hackathons = np.clip(rng.poisson(0.5, n), 0, 5)
    coding = np.clip(rng.normal(60, 15, n), 0, 100)
    aptitude = np.clip(rng.normal(60, 14, n), 0, 100)
    communication = np.clip(rng.normal(62, 15, n), 0, 100)
    study_hours = np.clip(rng.normal(3.5, 1.5, n), 0, 12)
    attendance = np.clip(rng.normal(80, 10, n), 30, 100)

    branch_effect = np.array([BRANCH_EFFECT[b] for b in branch])
    extra_effect = (extracurricular == "Yes").astype(float)

    score = (
        1.1 * (cgpa - 7.2) / 0.9
        + 0.8 * (coding - 60) / 15
        + 0.6 * (aptitude - 60) / 14
        + 0.5 * (communication - 62) / 15
        + 0.6 * np.minimum(internships, 2)
        + 0.15 * np.minimum(projects, 5)
        + 0.12 * np.minimum(certifications, 5)
        + 0.20 * np.minimum(hackathons, 3)
        + 0.40 * extra_effect
        + 0.50 * branch_effect
        - 0.90 * backlogs
        + 0.02 * (attendance - 80)
        + 0.10 * (study_hours - 3.5)
        - 0.80
    )
    probability = 1 / (1 + np.exp(-(score + rng.normal(0, 0.6, n))))
    placed = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "gender": gender,
            "branch": branch,
            "extracurricular": extracurricular,
            "cgpa": np.round(cgpa, 2),
            "tenth_percentage": np.round(tenth, 1),
            "twelfth_percentage": np.round(twelfth, 1),
            "backlogs": backlogs,
            "internships": internships,
            "projects": projects,
            "certifications": certifications,
            "hackathons": hackathons,
            "coding_score": np.round(coding, 1),
            "aptitude_score": np.round(aptitude, 1),
            "communication_score": np.round(communication, 1),
            "study_hours_per_day": np.round(study_hours, 1),
            "attendance_percentage": np.round(attendance, 1),
            "placed": placed,
        }
    )

    # Realistic data quality issues: a small share of missing values.
    for col in ["study_hours_per_day", "attendance_percentage"]:
        df.loc[rng.random(n) < 0.02, col] = np.nan

    return df


def ensure_dataset(path: Path | str, n_samples: int = 5000) -> Path:
    """Create the synthetic dataset at `path` if it does not already exist."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_student_data(n_samples).to_csv(path, index=False)
    return path
