"""Unit tests for feature engineering, temporal rolling windows, and risk indicators."""

import pytest
import numpy as np
import pandas as pd
from src.features.temporal import extract_calendar_features, compute_rolling_sensor_features
from src.features.near_miss import compute_near_miss_features
from src.features.risk_indicators import compute_operational_risk_indicators
from src.features.pipeline import SafetyFeaturePipeline


def test_calendar_features(sample_raw_telemetry):
    """Test calendar and cyclical feature extraction."""
    df_feats = extract_calendar_features(sample_raw_telemetry)
    assert "day_of_week" in df_feats.columns
    assert "sin_day_of_year" in df_feats.columns
    assert "cos_day_of_year" in df_feats.columns
    assert "is_night_shift" in df_feats.columns
    assert df_feats["is_night_shift"].iloc[1] == 1 # Night shift index


def test_rolling_sensor_features_no_leakage(sample_raw_telemetry):
    """Test that rolling window features are computed without future data leakage."""
    df_roll = compute_rolling_sensor_features(sample_raw_telemetry, windows=[7])
    assert "vibration_rms_mms_roll_mean_7d" in df_roll.columns

    # First day rolling mean should equal first day vibration
    first_vib = sample_raw_telemetry["vibration_rms_mms"].iloc[0]
    first_roll = df_roll["vibration_rms_mms_roll_mean_7d"].iloc[0]
    assert np.isclose(first_vib, first_roll)


def test_near_miss_features(sample_raw_telemetry):
    """Test near-miss rolling accumulations and momentum ratios."""
    df_nm = compute_near_miss_features(sample_raw_telemetry, windows=[7])
    assert "near_miss_sum_7d" in df_nm.columns
    assert "near_miss_ewm_4d" in df_nm.columns
    assert "is_near_miss_cluster_7d" in df_nm.columns

    # Day 0 had 0 near misses, day 7 has accumulated near misses
    assert df_nm["near_miss_sum_7d"].iloc[0] == 0
    assert df_nm["near_miss_sum_7d"].iloc[7] > 0


def test_operational_risk_indicators(sample_raw_telemetry):
    """Test ISO 10816 vibration guidelines and Swiss Cheese barrier breach counting."""
    df_risk = compute_operational_risk_indicators(
        sample_raw_telemetry,
        vibration_alert_threshold=2.8,
        vibration_danger_threshold=4.5,
    )
    assert "is_vibration_alert" in df_risk.columns
    assert "swiss_cheese_barrier_breaches" in df_risk.columns
    assert "maintenance_backlog_index" in df_risk.columns

    # Check vibration alert flag triggers correctly
    for idx, row in df_risk.iterrows():
        expected_flag = 1 if row["vibration_rms_mms"] >= 2.8 else 0
        assert row["is_vibration_alert"] == expected_flag


def test_pipeline_fit_transform(sample_raw_telemetry):
    """Test that SafetyFeaturePipeline fits and transforms cleanly."""
    pipeline = SafetyFeaturePipeline()
    X, y, meta = pipeline.fit_transform(sample_raw_telemetry)

    assert pipeline.is_fitted
    assert len(X) == len(sample_raw_telemetry)
    assert len(y) == len(sample_raw_telemetry)
    assert "record_id" not in X.columns
    assert "incident_next_7d" not in X.columns
    assert not X.isnull().values.any() # Zero missing values
