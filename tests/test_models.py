"""Unit tests for ML models, anomaly detector, calibration, and survival analyzer."""

import pytest
import numpy as np
import pandas as pd
from src.features.pipeline import SafetyFeaturePipeline
from src.models.baseline import BaselineModel
from src.models.ml_models import RandomForestSafetyModel, XGBoostSafetyModel
from src.models.anomaly import OperationalAnomalyDetector
from src.models.calibration import CalibratedSafetyModel
from src.models.survival import SafetySurvivalAnalyzer


@pytest.fixture
def sample_features_and_targets(sample_raw_telemetry):
    pipeline = SafetyFeaturePipeline()
    X, y, meta = pipeline.fit_transform(sample_raw_telemetry)
    return X, y, sample_raw_telemetry


def test_baseline_dummy_model(sample_features_and_targets):
    """Test Dummy Baseline prediction."""
    X, y, _ = sample_features_and_targets
    model = BaselineModel(model_type="dummy_prior")
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_random_forest_model(sample_features_and_targets):
    """Test Random Forest safety model fit and feature importances."""
    X, y, _ = sample_features_and_targets
    model = RandomForestSafetyModel(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    preds = model.predict(X, threshold=0.5)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert isinstance(model.feature_importances_, pd.Series)
    assert len(model.feature_importances_) == X.shape[1]


def test_xgboost_model(sample_features_and_targets):
    """Test XGBoost safety model fitting and threshold prediction."""
    X, y, _ = sample_features_and_targets
    model = XGBoostSafetyModel(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = model.predict(X, threshold=0.35)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_anomaly_detector(sample_features_and_targets):
    """Test Isolation Forest anomaly detection and continuous scoring."""
    X, y, _ = sample_features_and_targets
    detector = OperationalAnomalyDetector(contamination=0.1, n_estimators=20, random_state=42)
    detector.fit(X)

    flags = detector.predict_anomaly(X)
    scores = detector.score_samples(X)

    assert len(flags) == len(X)
    assert set(flags).issubset({0, 1})
    assert len(scores) == len(X)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_calibrated_safety_model(sample_features_and_targets):
    """Test probability calibration wrapper and Brier score evaluation."""
    X, y, _ = sample_features_and_targets
    base_model = RandomForestSafetyModel(n_estimators=10, max_depth=2, random_state=42)
    base_model.fit(X, y)

    cal_model = CalibratedSafetyModel(base_estimator=base_model, method="isotonic")
    cal_model.fit_calibration(X, y)

    cal_probs = cal_model.predict_proba(X)
    assert cal_probs.shape == (len(X), 2)
    assert np.allclose(cal_probs.sum(axis=1), 1.0)

    cal_diag = cal_model.evaluate_calibration(X, y)
    assert "brier_score" in cal_diag
    assert 0.0 <= cal_diag["brier_score"] <= 1.0


def test_survival_analyzer(sample_raw_telemetry):
    """Test Kaplan-Meier and Cox Proportional Hazards survival analyzer."""
    analyzer = SafetySurvivalAnalyzer()
    analyzer.fit(sample_raw_telemetry)

    table = analyzer.get_survival_table()
    assert "Day" in table.columns
    assert "Overall_Survival_Prob" in table.columns
    assert len(table) == 30
    # Survival probabilities should be monotonic non-increasing or between 0 and 1
    assert np.all((table["Overall_Survival_Prob"] >= 0.0) & (table["Overall_Survival_Prob"] <= 1.05))
