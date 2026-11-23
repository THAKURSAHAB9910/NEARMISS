"""Unsupervised anomaly detection module for unusual operational patterns.
Uses Isolation Forest to detect out-of-spec operational sensor combinations.
"""

import logging
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CORE_SENSOR_COLS = [
    "vibration_rms_mms",
    "vibration_kurtosis",
    "process_temperature_c",
    "ambient_temperature_c",
    "operating_pressure_bar",
    "rotational_speed_rpm",
    "duty_cycle_pct",
    "acoustic_emission_db",
]


class OperationalAnomalyDetector:
    """Unsupervised Isolation Forest detector for identifying irregular multi-sensor operating states."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.sensor_cols_: List[str] = CORE_SENSOR_COLS
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "OperationalAnomalyDetector":
        """Fit Isolation Forest on continuous physical sensor parameters."""
        logger.info(f"Fitting OperationalAnomalyDetector on {len(X)} observations...")
        available_sensors = [c for c in self.sensor_cols_ if c in X.columns]
        if not available_sensors:
            raise ValueError(f"None of the required sensor columns {self.sensor_cols_} found in features.")
        self.sensor_cols_ = available_sensors

        sensor_data = X[self.sensor_cols_].fillna(0.0)
        scaled_data = self.scaler.fit_transform(sensor_data)
        self.detector.fit(scaled_data)
        self.is_fitted = True
        return self

    def predict_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary anomaly indicator: 1 = Anomaly / Irregular State, 0 = Normal."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before predict.")
        sensor_data = X[self.sensor_cols_].fillna(0.0)
        scaled_data = self.scaler.transform(sensor_data)
        raw_pred = self.detector.predict(scaled_data) # -1 is anomaly, 1 is inlier
        return (raw_pred == -1).astype(int)

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Return continuous normalized anomaly score between 0.0 (safe/normal) and 1.0 (severe anomaly)."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before score_samples.")
        sensor_data = X[self.sensor_cols_].fillna(0.0)
        scaled_data = self.scaler.transform(sensor_data)
        decision = self.detector.decision_function(scaled_data)
        # Decision function: large positive is normal, negative is anomalous
        # Normalize to [0, 1] range where 1.0 is extreme anomaly
        scores = 1.0 / (1.0 + np.exp(3.0 * decision))
        return np.clip(scores, 0.0, 1.0)

    def save(self, filepath: str) -> None:
        """Serialize anomaly detector artifact."""
        joblib.dump(self, filepath)
        logger.info(f"Saved anomaly detector artifact to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "OperationalAnomalyDetector":
        """Load serialized anomaly detector artifact."""
        model = joblib.load(filepath)
        logger.info(f"Loaded anomaly detector from {filepath}")
        return model
