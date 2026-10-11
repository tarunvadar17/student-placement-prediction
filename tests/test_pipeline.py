"""Unit and integration tests for the placement pipeline."""
import pandas as pd
import pytest

from placement.config import ALL_FEATURES, TARGET
from placement.data_generator import generate_student_data
from placement.predict import PlacementPredictor, get_recommendations, risk_level
from placement.preprocessing import build_preprocessor, validate_schema
from placement.train import run_training


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_student_data(n_samples=800, seed=7)


def test_generator_produces_expected_schema(dataset):
    assert set(ALL_FEATURES + [TARGET]).issubset(dataset.columns)
    assert len(dataset) == 800
    assert set(dataset[TARGET].unique()) <= {0, 1}


def test_generator_is_reproducible():
    a = generate_student_data(n_samples=100, seed=3)
    b = generate_student_data(n_samples=100, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_validate_schema_rejects_missing_columns(dataset):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(dataset.drop(columns=["cgpa"]))


def test_preprocessor_handles_missing_values(dataset):
    X = dataset[ALL_FEATURES]
    transformed = build_preprocessor().fit_transform(X)
    assert not pd.isna(transformed).any()


def test_risk_bands():
    assert risk_level(0.9).startswith("Strong")
    assert risk_level(0.6).startswith("Moderate")
    assert risk_level(0.3).startswith("Low")
    assert risk_level(0.1).startswith("Very low")


def test_recommendations_flag_weak_areas():
    tips = get_recommendations({"backlogs": 2, "internships": 0, "coding_score": 40})
    assert any("backlogs" in t.lower() for t in tips)
    assert any("internship" in t.lower() for t in tips)


def test_training_and_prediction_end_to_end(dataset, tmp_path):
    summary = run_training(
        dataset,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        tune=False,
    )
    assert summary["test"]["roc_auc"] > 0.6
    assert (tmp_path / "models" / "placement_model.joblib").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()

    predictor = PlacementPredictor(tmp_path / "models" / "placement_model.joblib")
    sample = dataset[ALL_FEATURES].iloc[0].to_dict()
    result = predictor.predict_one(sample)
    assert 0.0 <= result["placement_probability"] <= 1.0
    assert result["prediction"] in {0, 1}
    assert isinstance(result["recommendations"], list)
