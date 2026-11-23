"""Temporal feature engineering module.
Extracts calendar features, cyclical encodings, and rolling sensor aggregations without leakage.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar-based temporal features and shift encodings."""
    df_out = df.copy()
    ts = pd.to_datetime(df_out["timestamp"])

    df_out["day_of_week"] = ts.dt.dayofweek
    df_out["day_of_month"] = ts.dt.day
    df_out["month"] = ts.dt.month
    df_out["day_of_year"] = ts.dt.dayofyear
    df_out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical day-of-year encoding (seasonal ambient variations)
    df_out["sin_day_of_year"] = np.sin(2 * np.pi * df_out["day_of_year"] / 365.25)
    df_out["cos_day_of_year"] = np.cos(2 * np.pi * df_out["day_of_year"] / 365.25)

    # Cyclical day-of-week encoding
    df_out["sin_day_of_week"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["cos_day_of_week"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)

    # Shift binary flags
    df_out["is_night_shift"] = (df_out["shift_type"] == "Night").astype(int)
    df_out["is_swing_shift"] = (df_out["shift_type"] == "Swing").astype(int)

    return df_out


def compute_rolling_sensor_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """Compute rolling window means, standard deviations, and maximums per equipment unit.
    Calculated strictly over past records up to day t (no lookahead into t+1).
    """
    if windows is None:
        windows = [7, 14, 30]

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_out = df_out.sort_values(by=["unit_id", "timestamp"]).reset_index(drop=True)

    numeric_sensor_cols = [
        "vibration_rms_mms",
        "process_temperature_c",
        "duty_cycle_pct",
        "acoustic_emission_db",
    ]

    grouped = df_out.groupby("unit_id")

    for w in windows:
        for col in numeric_sensor_cols:
            # Rolling mean
            df_out[f"{col}_roll_mean_{w}d"] = grouped[col].transform(
                lambda s: s.rolling(window=w, min_periods=1).mean()
            )
            # Rolling std
            df_out[f"{col}_roll_std_{w}d"] = grouped[col].transform(
                lambda s: s.rolling(window=w, min_periods=1).std().fillna(0.0)
            )

        # Rolling max for vibration and temp
        df_out[f"vibration_rms_roll_max_{w}d"] = grouped["vibration_rms_mms"].transform(
            lambda s: s.rolling(window=w, min_periods=1).max()
        )
        df_out[f"process_temp_roll_max_{w}d"] = grouped["process_temperature_c"].transform(
            lambda s: s.rolling(window=w, min_periods=1).max()
        )

    return df_out
