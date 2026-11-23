"""Unit tests for SHAP explainability and instance root-cause attribution."""

import numpy as np
import pandas as pd
import pytest
from src.features.pipeline import SafetyFeaturePipeline
from src.models.ml_models import RandomForestSafetyModel
from src.explainability.shap_explainer import SafetySHAPExplainer


def test_shap_explainer_initialization(sample_raw_telemetry):
    """Test that SHAP TreeExplainer initializes on trained tree model."""
    pipeline = SafetyFeaturePipeline()
    X, y, meta = pipeline.fit_transform(sample_raw_telemetry)

    model = RandomForestSafetyModel(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    explainer = SafetySHAPExplainer(model_wrapper=model, feature_names=list(X.columns))
    explainer.fit()

    assert explainer.is_fitted
    assert explainer.expected_value is not None


def test_shap_instance_explanation(sample_raw_telemetry):
    """Test local instance explanation breakdown for a single shift."""
    pipeline = SafetyFeaturePipeline()
    X, y, meta = pipeline.fit_transform(sample_raw_telemetry)

    model = RandomForestSafetyModel(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    explainer = SafetySHAPExplainer(model_wrapper=model, feature_names=list(X.columns))
    explainer.fit()

    explanation = explainer.explain_instance(X.iloc[0], top_n=3)
    assert "top_risk_drivers" in explanation
    assert "top_mitigating_factors" in explanation
    assert "base_expected_value" in explanation
    assert isinstance(explanation["top_risk_drivers"], list)
