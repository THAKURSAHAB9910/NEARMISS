"""Probability calibration and threshold optimization module for decision support.
Calibrates tree ensemble probabilities using Isotonic Regression or Platt Scaling.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class CalibratedSafetyModel:
    """Calibrates an underlying estimator's predicted probabilities using a validation split."""

    def __init__(self, base_estimator, method: str = "isotonic"):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator = None
        self.optimal_threshold_: float = 0.5
        self.is_fitted: bool = False

    def fit_calibration(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cost_fn: float = 50000.0,
        cost_fp: float = 1500.0
    ) -> "CalibratedSafetyModel":
        """Calibrate probabilities on held-out validation data and optimize decision threshold."""
        logger.info(f"Calibrating probabilities using method='{self.method}' on {len(X_val)} validation samples...")

        # Obtain uncalibrated positive class probabilities from base estimator
        raw_val_probs = self.base_estimator.predict_proba(X_val)[:, 1]
        y_val_arr = y_val.values.astype(int)

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_val_probs, y_val_arr)
            cal_probs = self.calibrator.predict(raw_val_probs)
        elif self.method in ("sigmoid", "platt"):
            self.calibrator = LogisticRegression(solver="lbfgs")
            self.calibrator.fit(raw_val_probs.reshape(-1, 1), y_val_arr)
            cal_probs = self.calibrator.predict_proba(raw_val_probs.reshape(-1, 1))[:, 1]
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

        self.is_fitted = True

        # Optimize decision threshold by minimizing total operational safety cost
        thresholds = np.linspace(0.05, 0.95, 91)
        best_cost = float("inf")
        best_thresh = 0.5

        for th in thresholds:
            y_pred = (cal_probs >= th).astype(int)
            fn = np.sum((y_val_arr == 1) & (y_pred == 0))
            fp = np.sum((y_val_arr == 0) & (y_pred == 1))
            total_cost = (fn * cost_fn) + (fp * cost_fp)
            if total_cost < best_cost:
                best_cost = total_cost
                best_thresh = float(th)

        self.optimal_threshold_ = best_thresh
        logger.info(f"Optimal cost-sensitive threshold determined: {self.optimal_threshold_:.3f} (min cost: ${best_cost:,.0f})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities [P(safe), P(incident)]."""
        if not self.is_fitted or self.calibrator is None:
            raise RuntimeError("Model must be calibrated before predict_proba.")

        raw_probs = self.base_estimator.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            cal_p1 = self.calibrator.predict(raw_probs)
        else:
            cal_p1 = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        cal_p1 = np.clip(cal_p1, 0.0, 1.0)
        cal_p0 = 1.0 - cal_p1
        return np.column_stack([cal_p0, cal_p1])

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary class using optimal cost threshold or custom threshold."""
        th = self.optimal_threshold_ if threshold is None else threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= th).astype(int)

    def evaluate_calibration(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Union[float, np.ndarray]]:
        """Compute calibration diagnostics: Brier score and reliability curve coordinates."""
        probs = self.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, probs)
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")

        return {
            "brier_score": float(brier),
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "mean_predicted_prob": float(np.mean(probs)),
            "observed_positive_rate": float(np.mean(y_test)),
        }

    def save(self, filepath: str) -> None:
        """Serialize calibrated model artifact."""
        joblib.dump(self, filepath)
        logger.info(f"Saved CalibratedSafetyModel to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CalibratedSafetyModel":
        """Load serialized calibrated model artifact."""
        model = joblib.load(filepath)
        logger.info(f"Loaded CalibratedSafetyModel from {filepath}")
        return model
