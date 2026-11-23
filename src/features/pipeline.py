"""Feature pipeline orchestration module.
Fits preprocessors on training splits and transforms test/inference data with zero data leakage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.temporal import extract_calendar_features, compute_rolling_sensor_features
from src.features.near_miss import compute_near_miss_features
from src.features.risk_indicators import compute_operational_risk_indicators

logger = logging.getLogger(__name__)


# Non-feature identifier and leakage columns that must be dropped from ML feature matrices
METADATA_AND_TARGET_COLS = [
    "record_id",
    "timestamp",
    "unit_id",
    "incident_today",
    "incident_next_7d",
    "days_to_incident",
    "survival_event",
    "data_origin",
]

CATEGORICAL_COLS = [
    "facility",
    "equipment_type",
    "shift_type",
    "near_miss_category",
]


class SafetyFeaturePipeline:
    """End-to-end feature engineering and pre-processing pipeline for industrial safety prediction."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.windows = self.cfg["features"]["rolling_windows_days"]
        self.vib_alert = self.cfg["features"]["iso_vibration_alert_threshold"]
        self.vib_danger = self.cfg["features"]["iso_vibration_danger_threshold"]

        self.feature_columns_: Optional[List[str]] = None
        self.category_levels_: Dict[str, List[str]] = {}
        self.is_fitted: bool = False

    def _engineer_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sequential domain feature transforms."""
        df_feats = extract_calendar_features(df)
        df_feats = compute_rolling_sensor_features(df_feats, windows=self.windows)
        df_feats = compute_near_miss_features(df_feats, windows=self.windows)
        df_feats = compute_operational_risk_indicators(
            df_feats,
            vibration_alert_threshold=self.vib_alert,
            vibration_danger_threshold=self.vib_danger,
        )
        return df_feats

    def fit(self, df_train: pd.DataFrame) -> "SafetyFeaturePipeline":
        """Fit categorical encodings and register feature schema strictly from training data."""
        logger.info(f"Fitting SafetyFeaturePipeline on {len(df_train)} training observations...")
        df_feats = self._engineer_features_df(df_train)

        # Record unique categories seen in train set
        for col in CATEGORICAL_COLS:
            if col in df_feats.columns:
                unique_cats = sorted(df_feats[col].astype(str).unique().tolist())
                self.category_levels_[col] = unique_cats

        # Build dummy columns to determine exact feature schema
        df_encoded = self._encode_categoricals(df_feats)
        feature_cols = [c for c in df_encoded.columns if c not in METADATA_AND_TARGET_COLS]
        self.feature_columns_ = sorted(feature_cols)
        self.is_fitted = True

        logger.info(f"Pipeline fitted successfully. Total engineered features: {len(self.feature_columns_)}")
        return self

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features adhering strictly to fitted category levels."""
        df_out = df.copy()
        for col, levels in self.category_levels_.items():
            if col in df_out.columns:
                for lvl in levels:
                    col_name = f"{col}__{lvl}"
                    df_out[col_name] = (df_out[col].astype(str) == lvl).astype(int)
                df_out = df_out.drop(columns=[col])
        return df_out

    def transform(
        self,
        df: pd.DataFrame,
        return_targets: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.Series, pd.DataFrame], pd.DataFrame]:
        """Transform input dataframe into model feature matrix X, target y, and metadata."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform.")

        df_feats = self._engineer_features_df(df)
        df_encoded = self._encode_categoricals(df_feats)

        # Align with fitted feature columns
        for col in self.feature_columns_:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        X = df_encoded[self.feature_columns_].copy()

        # Handle any possible remaining NaNs from early rolling windows
        X = X.fillna(0.0)

        if return_targets and "incident_next_7d" in df_encoded.columns:
            y = df_encoded["incident_next_7d"].astype(int)
            meta_cols = [c for c in ["record_id", "timestamp", "unit_id", "facility", "equipment_type", "days_to_incident", "survival_event"] if c in df.columns]
            meta = df[meta_cols].copy()
            return X, y, meta

        return X

    def fit_transform(
        self,
        df_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Fit on training data and return transformed feature matrix, target, and metadata."""
        return self.fit(df_train).transform(df_train, return_targets=True)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted feature pipeline artifact."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved SafetyFeaturePipeline artifact to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SafetyFeaturePipeline":
        """Load fitted feature pipeline artifact."""
        pipeline = joblib.load(filepath)
        logger.info(f"Loaded SafetyFeaturePipeline from {filepath} (fitted={pipeline.is_fitted})")
        return pipeline
