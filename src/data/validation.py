"""Data validation and sanity checking module for industrial safety telemetry.
Uses Pydantic for record validation and vectorized checks for batch dataframes.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data fails schema, integrity or boundary checks."""
    pass


class SafetyObservation(BaseModel):
    """Pydantic model representing a single validated telemetry observation."""
    record_id: str
    timestamp: str
    facility: str
    unit_id: str
    equipment_type: str
    shift_type: str
    ambient_temperature_c: float = Field(..., ge=-50.0, le=80.0)
    process_temperature_c: float = Field(..., ge=-20.0, le=250.0)
    vibration_rms_mms: float = Field(..., ge=0.0, le=50.0)
    vibration_kurtosis: float = Field(..., ge=1.0, le=30.0)
    operating_pressure_bar: float = Field(..., ge=0.0, le=200.0)
    rotational_speed_rpm: float = Field(..., ge=0.0, le=15000.0)
    duty_cycle_pct: float = Field(..., ge=0.0, le=100.0)
    acoustic_emission_db: float = Field(..., ge=20.0, le=140.0)
    days_since_last_pm: int = Field(..., ge=0)
    pm_overdue_days: int = Field(..., ge=0)
    open_maintenance_tickets: int = Field(..., ge=0, le=50)
    daily_near_miss_count: int = Field(..., ge=0, le=20)
    near_miss_category: str
    shift_overtime_hours: float = Field(..., ge=0.0, le=24.0)
    operator_fatigue_index: float = Field(..., ge=0.0, le=100.0)
    safety_audit_score: float = Field(..., ge=0.0, le=100.0)
    incident_today: Optional[int] = Field(default=0, ge=0, le=1)
    incident_next_7d: Optional[int] = Field(default=None, ge=0, le=1)
    days_to_incident: Optional[int] = Field(default=None, ge=1)
    data_origin: Optional[str] = "unspecified"

    @field_validator("shift_type")
    @classmethod
    def check_shift(cls, v: str) -> str:
        valid_shifts = {"Day", "Swing", "Night"}
        if v not in valid_shifts:
            raise ValueError(f"Invalid shift_type: '{v}'. Must be one of {valid_shifts}")
        return v


REQUIRED_COLUMNS = [
    "record_id",
    "timestamp",
    "facility",
    "unit_id",
    "equipment_type",
    "shift_type",
    "ambient_temperature_c",
    "process_temperature_c",
    "vibration_rms_mms",
    "vibration_kurtosis",
    "operating_pressure_bar",
    "rotational_speed_rpm",
    "duty_cycle_pct",
    "acoustic_emission_db",
    "days_since_last_pm",
    "pm_overdue_days",
    "open_maintenance_tickets",
    "daily_near_miss_count",
    "near_miss_category",
    "shift_overtime_hours",
    "operator_fatigue_index",
    "safety_audit_score",
]

NUMERIC_RANGES = {
    "ambient_temperature_c": (-50.0, 80.0),
    "process_temperature_c": (-20.0, 250.0),
    "vibration_rms_mms": (0.0, 50.0),
    "vibration_kurtosis": (1.0, 30.0),
    "operating_pressure_bar": (0.0, 200.0),
    "rotational_speed_rpm": (0.0, 15000.0),
    "duty_cycle_pct": (0.0, 100.0),
    "acoustic_emission_db": (20.0, 140.0),
    "days_since_last_pm": (0, 1000),
    "pm_overdue_days": (0, 1000),
    "open_maintenance_tickets": (0, 50),
    "daily_near_miss_count": (0, 20),
    "shift_overtime_hours": (0.0, 24.0),
    "operator_fatigue_index": (0.0, 100.0),
    "safety_audit_score": (0.0, 100.0),
}


def validate_raw_telemetry(df: pd.DataFrame, drop_invalid: bool = False) -> pd.DataFrame:
    """Validate a batch dataframe against required schema and physical boundary checks."""
    logger.info(f"Starting validation for dataframe with shape {df.shape}...")

    # 1. Missing columns check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Input dataframe missing required columns: {missing_cols}")

    df_clean = df.copy()

    # 2. Duplicate handling
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["unit_id", "timestamp"]).reset_index(drop=True)
    dups_removed = initial_len - len(df_clean)
    if dups_removed > 0:
        logger.warning(f"Removed {dups_removed} duplicate unit-timestamp records.")

    # 3. Timestamp parsing
    try:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    except Exception as e:
        raise DataValidationError(f"Failed to parse 'timestamp' column as datetime: {e}")

    # 4. Check for unexpected nulls in required columns
    # Handle categorical string 'None' read as NaN
    if "near_miss_category" in df_clean.columns:
        df_clean["near_miss_category"] = df_clean["near_miss_category"].fillna("None")

    null_counts = df_clean[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning(f"Columns with nulls detected: {cols_with_nulls.to_dict()}")
        if drop_invalid:
            df_clean = df_clean.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
            logger.info(f"Dropped invalid null rows; remaining rows: {len(df_clean)}")
        else:
            # Impute sensible medians for numeric, 'None' for categorical
            for col in cols_with_nulls.index:
                if col in NUMERIC_RANGES:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logger.info(f"Imputed column '{col}' with median {median_val}")
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")

    # 5. Range and boundary validation
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df_clean.columns:
            out_of_bounds = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            n_viol = out_of_bounds.sum()
            if n_viol > 0:
                logger.warning(f"Column '{col}' has {n_viol} records outside valid bounds [{min_val}, {max_val}].")
                # Clip to physical bounds to prevent numerical explosion
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

    logger.info(f"Validation successful. Clean dataframe ready with {len(df_clean)} records.")
    return df_clean
