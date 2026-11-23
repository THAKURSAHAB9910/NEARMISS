"""Near-miss frequency, clustering, and warning accumulation feature engineering.
Implements Heinrich's safety triangle ratios and exponential decay warning indices.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_near_miss_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """Compute near-miss frequency, momentum, and warning clustering per unit without leakage."""
    if windows is None:
        windows = [7, 14, 30]

    df_out = df.copy()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_out = df_out.sort_values(by=["unit_id", "timestamp"]).reset_index(drop=True)

    grouped = df_out.groupby("unit_id")

    for w in windows:
        # Sum of near misses in the past w days
        df_out[f"near_miss_sum_{w}d"] = grouped["daily_near_miss_count"].transform(
            lambda s: s.rolling(window=w, min_periods=1).sum()
        )
        # Rate of days with at least one near miss in the past w days
        df_out[f"near_miss_active_days_ratio_{w}d"] = grouped["daily_near_miss_count"].transform(
            lambda s: (s > 0).astype(float).rolling(window=w, min_periods=1).mean()
        )

    # Exponentially weighted near-miss intensity (halflife of 4 days)
    # High value means sudden burst of recent near-misses
    df_out["near_miss_ewm_4d"] = grouped["daily_near_miss_count"].transform(
        lambda s: s.ewm(halflife=4, min_periods=1).mean()
    )

    # Near-miss momentum: 7-day normalized rate vs 30-day normalized rate
    # If > 1.0, warning frequency is accelerating
    rate_7d = df_out.get("near_miss_sum_7d", df_out["daily_near_miss_count"]) / 7.0
    if "near_miss_sum_30d" in df_out.columns:
        rate_30d = (df_out["near_miss_sum_30d"] / 30.0) + 1e-5
    else:
        rate_30d = (rate_7d) + 1e-5
    df_out["near_miss_acceleration_ratio"] = (rate_7d / rate_30d).clip(upper=10.0)

    # Warning cluster indicator: >= 3 near misses in the last 7 days
    df_out["is_near_miss_cluster_7d"] = (df_out["near_miss_sum_7d"] >= 3).astype(int)

    # Cumulative past incidents per unit up to day t (strictly expanding past history)
    if "incident_today" in df_out.columns:
        df_out["cum_past_incidents"] = grouped["incident_today"].transform(
            lambda s: s.shift(1, fill_value=0).cumsum()
        )
        df_out["cum_past_near_misses"] = grouped["daily_near_miss_count"].transform(
            lambda s: s.shift(1, fill_value=0).cumsum()
        )
        # Heinrich ratio: near-misses per incident (stabilized with laplace smoothing)
        df_out["heinrich_ratio_unit"] = (
            (df_out["cum_past_near_misses"] + 10.0) / (df_out["cum_past_incidents"] + 1.0)
        )
    else:
        df_out["cum_past_incidents"] = 0
        df_out["cum_past_near_misses"] = 0
        df_out["heinrich_ratio_unit"] = 30.0

    return df_out
