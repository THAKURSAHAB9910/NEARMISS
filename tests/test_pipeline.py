"""Integration tests verifying end-to-end pipeline execution and artifact persistence."""

import os
from pathlib import Path
import pytest
import pandas as pd
from src.data.ingestion import load_safety_data, split_data_temporal
from src.features.pipeline import SafetyFeaturePipeline
from src.models.ml_models import RandomForestSafetyModel
from src.models.calibration import CalibratedSafetyModel
from src.evaluation.metrics import evaluate_binary_predictions


def test_full_pipeline_integration(tmp_path):
    """Test full cycle: load -> split -> feature pipeline -> train -> calibrate -> evaluate."""
    df = load_safety_data()
    train_df, val_df, test_df = split_data_temporal(df)

    # Use a small subset for fast integration testing
    train_sub = train_df.head(500)
    val_sub = val_df.head(200)
    test_sub = test_df.head(200)

    pipeline = SafetyFeaturePipeline()
    X_train, y_train, _ = pipeline.fit_transform(train_sub)
    X_val, y_val, _ = pipeline.transform(val_sub, return_targets=True)
    X_test, y_test, _ = pipeline.transform(test_sub, return_targets=True)

    # Train model
    model = RandomForestSafetyModel(n_estimators=15, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Calibrate
    calibrator = CalibratedSafetyModel(base_estimator=model, method="isotonic")
    calibrator.fit_calibration(X_val, y_val)

    # Evaluate
    probs = calibrator.predict_proba(X_test)[:, 1]
    preds = calibrator.predict(X_test, threshold=0.35)
    metrics = evaluate_binary_predictions(y_test, preds, probs)

    assert "recall" in metrics
    assert "precision" in metrics
    assert "pr_auc" in metrics
    assert "total_cost" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0

    # Save and reload pipeline
    pipe_path = tmp_path / "test_pipe.joblib"
    pipeline.save(pipe_path)
    loaded_pipe = SafetyFeaturePipeline.load(pipe_path)
    assert loaded_pipe.is_fitted
