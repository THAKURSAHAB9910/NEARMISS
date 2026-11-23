"""Time-to-event and survival analysis for industrial safety.
Fits Kaplan-Meier survival curves and Cox Proportional Hazards regression using lifelines.
"""

import logging
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

logger = logging.getLogger(__name__)


class SafetySurvivalAnalyzer:
    """Survival analysis model estimating time-to-incident hazard curves."""

    def __init__(self):
        self.kmf_overall = KaplanMeierFitter()
        self.kmf_high_nm = KaplanMeierFitter()
        self.kmf_low_nm = KaplanMeierFitter()
        self.kmf_high_vib = KaplanMeierFitter()
        self.kmf_low_vib = KaplanMeierFitter()
        self.cph = CoxPHFitter(penalizer=0.05)
        self.cph_summary_: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "SafetySurvivalAnalyzer":
        """Fit Kaplan-Meier curves and Cox Proportional Hazards on historical observations."""
        logger.info(f"Fitting survival models on {len(df)} observations...")

        durations = df["days_to_incident"].values
        events = df["survival_event"].values

        # 1. Overall Kaplan-Meier curve
        self.kmf_overall.fit(durations, event_observed=events, label="Fleet Overall")

        # 2. Stratified by Near-Miss clustering (high vs low)
        high_nm_mask = (df["daily_near_miss_count"] >= 2) if "daily_near_miss_count" in df.columns else (events == 1)
        self.kmf_high_nm.fit(durations[high_nm_mask], event_observed=events[high_nm_mask], label="High Near-Miss Warning")
        self.kmf_low_nm.fit(durations[~high_nm_mask], event_observed=events[~high_nm_mask], label="Normal / Low Near-Miss")

        # 3. Stratified by ISO Vibration alert (>= 2.8 mm/s)
        high_vib_mask = (df["vibration_rms_mms"] >= 2.8) if "vibration_rms_mms" in df.columns else (durations < 7)
        self.kmf_high_vib.fit(durations[high_vib_mask], event_observed=events[high_vib_mask], label="Vibration Alert (>=2.8 mm/s)")
        self.kmf_low_vib.fit(durations[~high_vib_mask], event_observed=events[~high_vib_mask], label="Vibration Normal (<2.8 mm/s)")

        # 4. Cox Proportional Hazards Regression
        # Select key interpretable covariates
        cph_cols = [
            "days_to_incident",
            "survival_event",
            "vibration_rms_mms",
            "pm_overdue_days",
            "daily_near_miss_count",
            "operator_fatigue_index",
            "shift_overtime_hours",
            "safety_audit_score",
        ]
        available_cols = [c for c in cph_cols if c in df.columns]
        cph_df = df[available_cols].dropna().copy()

        # Sample for fast, robust Cox fitting if dataset is large
        if len(cph_df) > 5000:
            cph_sample = cph_df.sample(n=5000, random_state=42)
        else:
            cph_sample = cph_df

        try:
            self.cph.fit(
                cph_sample,
                duration_col="days_to_incident",
                event_col="survival_event"
            )
            self.cph_summary_ = self.cph.summary[["exp(coef)", "se(coef)", "p", "-log2(p)"]].copy()
            self.cph_summary_.columns = ["Hazard_Ratio", "Std_Error", "p_value", "neg_log2_p"]
            logger.info("Cox Proportional Hazards model fitted successfully.")
        except Exception as e:
            logger.warning(f"CoxPH fitting encountered non-convergence or singularity: {e}")
            self.cph_summary_ = pd.DataFrame()

        self.is_fitted = True
        return self

    def get_survival_table(self) -> pd.DataFrame:
        """Return tabulated survival probabilities over days 1 to 30."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get survival table.")

        timeline = np.arange(1, 31)
        table = pd.DataFrame({
            "Day": timeline,
            "Overall_Survival_Prob": self.kmf_overall.survival_function_at_times(timeline).values,
            "High_Near_Miss_Survival": self.kmf_high_nm.survival_function_at_times(timeline).values,
            "Low_Near_Miss_Survival": self.kmf_low_nm.survival_function_at_times(timeline).values,
            "Vibration_Alert_Survival": self.kmf_high_vib.survival_function_at_times(timeline).values,
            "Vibration_Normal_Survival": self.kmf_low_vib.survival_function_at_times(timeline).values,
        })
        return table

    def save(self, filepath: str) -> None:
        """Serialize survival analyzer."""
        joblib.dump(self, filepath)
        logger.info(f"Saved SafetySurvivalAnalyzer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SafetySurvivalAnalyzer":
        """Load serialized survival analyzer."""
        analyzer = joblib.load(filepath)
        logger.info(f"Loaded SafetySurvivalAnalyzer from {filepath}")
        return analyzer
