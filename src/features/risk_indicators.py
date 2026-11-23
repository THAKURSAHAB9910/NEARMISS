"""Operational, mechanical, and human risk indicator engineering.
Implements ISO 10816 vibration guidelines, thermal gradients, and multi-barrier failure indices.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_operational_risk_indicators(
    df: pd.DataFrame,
    vibration_alert_threshold: float = 2.8,
    vibration_danger_threshold: float = 4.5
) -> pd.DataFrame:
    """Compute domain-specific risk indicators for mechanical, thermal, and human factors."""
    df_out = df.copy()

    # 1. ISO 10816 Mechanical Vibration Severity Indicators
    df_out["is_vibration_alert"] = (df_out["vibration_rms_mms"] >= vibration_alert_threshold).astype(int)
    df_out["is_vibration_danger"] = (df_out["vibration_rms_mms"] >= vibration_danger_threshold).astype(int)
    df_out["vibration_severity_ratio"] = df_out["vibration_rms_mms"] / vibration_alert_threshold
    # Peakedness above Gaussian kurtosis (normal=3.0)
    df_out["vibration_kurtosis_excess"] = np.maximum(0.0, df_out["vibration_kurtosis"] - 3.0)

    # 2. Thermal and Environmental Strain
    df_out["thermal_gradient"] = df_out["process_temperature_c"] - df_out["ambient_temperature_c"]
    # Ratio against absolute ambient temperature
    df_out["thermal_stress_ratio"] = df_out["process_temperature_c"] / (df_out["ambient_temperature_c"] + 273.15)

    # 3. Preventive Maintenance Backlog & Ticket Density
    df_out["is_pm_overdue"] = (df_out["pm_overdue_days"] > 0).astype(int)
    df_out["pm_overdue_normalized"] = df_out["pm_overdue_days"] / 14.0
    # Compounding maintenance risk: overdue days interacting with open tickets
    df_out["maintenance_backlog_index"] = (df_out["pm_overdue_days"] + 1) * (df_out["open_maintenance_tickets"] + 1)

    # 4. Human Factors, Overtime & Shift Fatigue
    df_out["operator_fatigue_normalized"] = df_out["operator_fatigue_index"] / 100.0
    df_out["high_fatigue_flag"] = (df_out["operator_fatigue_index"] >= 65.0).astype(int)
    df_out["overtime_severity"] = df_out["shift_overtime_hours"] / 8.0

    # Human-Machine Interaction stress: tired crew operating vibrating machine
    df_out["human_mechanical_stress"] = df_out["operator_fatigue_normalized"] * df_out["vibration_rms_mms"]

    # 5. Safety Audit Health
    df_out["safety_audit_deficit"] = np.maximum(0.0, 100.0 - df_out["safety_audit_score"])

    # 6. Swiss Cheese Multi-Barrier Failure Index
    # Alignment of multiple critical holes in safety defenses
    cluster_flag = df_out["is_near_miss_cluster_7d"] if "is_near_miss_cluster_7d" in df_out.columns else 0
    df_out["swiss_cheese_barrier_breaches"] = (
        df_out["is_vibration_alert"]
        + df_out["is_pm_overdue"]
        + df_out["high_fatigue_flag"]
        + cluster_flag
        + (df_out["safety_audit_score"] < 80.0).astype(int)
    )

    return df_out
