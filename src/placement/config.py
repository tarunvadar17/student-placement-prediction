"""Central configuration: paths, feature schema, and training constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student_placement.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "placement_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TARGET = "placed"
TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

NUMERIC_FEATURES = [
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "internships",
    "projects",
    "certifications",
    "hackathons",
    "coding_score",
    "aptitude_score",
    "communication_score",
    "study_hours_per_day",
    "attendance_percentage",
]

CATEGORICAL_FEATURES = ["gender", "branch", "extracurricular"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
