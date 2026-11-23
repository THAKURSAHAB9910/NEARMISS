"""Pytest test configuration and shared fixtures for NEARMISS."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import yaml


@pytest.fixture(scope="session")
def sample_config():
    """Load project configuration dictionary."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_raw_telemetry():
    """Generate a small deterministic telemetry sample DataFrame for unit testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = []
    for i, d in enumerate(dates):
        data.append({
            "record_id": f"REC_TEST_{i:03d}",
            "timestamp": d.strftime("%Y-%m-%d"),
            "facility": "Refinery-North",
            "unit_id": "Refinery-North_PUMP_01",
            "equipment_type": "Centrifugal Pump",
            "shift_type": "Day" if i % 2 == 0 else "Night",
            "ambient_temperature_c": 22.0 + (i * 0.5),
            "process_temperature_c": 60.0 + (i * 1.2),
            "vibration_rms_mms": 1.5 + (i * 0.2),
            "vibration_kurtosis": 3.0 + (i * 0.1),
            "operating_pressure_bar": 8.5 + (i * 0.1),
            "rotational_speed_rpm": 1780.0,
            "duty_cycle_pct": 85.0,
            "acoustic_emission_db": 65.0 + i,
            "days_since_last_pm": 10 + i,
            "pm_overdue_days": max(0, i - 5),
            "open_maintenance_tickets": 1 if i > 3 else 0,
            "daily_near_miss_count": 2 if i > 6 else (1 if i > 3 else 0),
            "near_miss_category": "Mechanical_Guard_Trip" if i > 3 else "None",
            "shift_overtime_hours": 2.0 if i % 2 == 1 else 0.0,
            "operator_fatigue_index": 45.0 + (i * 3.0),
            "safety_audit_score": 92.0 - i,
            "incident_today": 1 if i == 8 else 0,
            "incident_next_7d": 1 if i < 8 and i >= 1 else 0,
            "days_to_incident": (8 - i) if i < 8 else 30,
            "survival_event": 1 if i < 8 and i >= 1 else 0,
            "data_origin": "test_fixture",
        })
    return pd.DataFrame(data)
