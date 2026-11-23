"""Unit tests for data ingestion, schema enforcement, and boundary checks."""

import pytest
import pandas as pd
from pydantic import ValidationError
from src.data.validation import validate_raw_telemetry, DataValidationError, SafetyObservation


def test_safety_observation_pydantic_valid(sample_raw_telemetry):
    """Test that a valid dictionary parses cleanly into SafetyObservation."""
    row = sample_raw_telemetry.iloc[0].to_dict()
    obs = SafetyObservation(**row)
    assert obs.record_id == "REC_TEST_000"
    assert obs.vibration_rms_mms == 1.5
    assert obs.shift_type == "Day"


def test_safety_observation_invalid_shift():
    """Test that invalid shift_type fails Pydantic validation."""
    invalid_data = {
        "record_id": "REC_ERR_01",
        "timestamp": "2024-01-01",
        "facility": "Refinery-North",
        "unit_id": "Refinery-North_PUMP_01",
        "equipment_type": "Centrifugal Pump",
        "shift_type": "Graveyard", # Invalid shift
        "ambient_temperature_c": 22.0,
        "process_temperature_c": 60.0,
        "vibration_rms_mms": 1.5,
        "vibration_kurtosis": 3.0,
        "operating_pressure_bar": 8.5,
        "rotational_speed_rpm": 1780.0,
        "duty_cycle_pct": 85.0,
        "acoustic_emission_db": 65.0,
        "days_since_last_pm": 10,
        "pm_overdue_days": 0,
        "open_maintenance_tickets": 0,
        "daily_near_miss_count": 0,
        "near_miss_category": "None",
        "shift_overtime_hours": 0.0,
        "operator_fatigue_index": 45.0,
        "safety_audit_score": 92.0,
    }
    with pytest.raises(ValidationError):
        SafetyObservation(**invalid_data)


def test_validate_raw_telemetry_missing_columns():
    """Test that dataframe missing required columns raises DataValidationError."""
    bad_df = pd.DataFrame({"record_id": ["REC_1"], "timestamp": ["2024-01-01"]})
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_raw_telemetry(bad_df)


def test_validate_raw_telemetry_clipping(sample_raw_telemetry):
    """Test that extreme outlier values are safely clipped to physical bounds."""
    df_extreme = sample_raw_telemetry.copy()
    df_extreme.loc[0, "vibration_rms_mms"] = 999.0 # Extreme unphysical outlier
    df_extreme.loc[0, "duty_cycle_pct"] = 150.0 # Above 100%

    cleaned = validate_raw_telemetry(df_extreme)
    assert cleaned.loc[0, "vibration_rms_mms"] == 50.0 # Clipped to max
    assert cleaned.loc[0, "duty_cycle_pct"] == 100.0 # Clipped to max
