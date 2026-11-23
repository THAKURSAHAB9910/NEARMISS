"""Tree ensemble models for industrial safety risk forecasting:
Random Forest and XGBoost with class-imbalance weighting.
"""

import logging
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class RandomForestSafetyModel:
    """Random Forest classifier tailored for multi-factor safety risk forecasting."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.model = RandomForestClassifier(**self.params)
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestSafetyModel":
        """Fit Random Forest on features X and binary safety incident targets y."""
        logger.info(f"Training RandomForestClassifier on {len(X)} rows with imbalance weighting...")
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary outcome using specified threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities [P(safe), P(incident)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return Gini feature importances as a pandas Series."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to access feature importances.")
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)

    def save(self, filepath: str) -> None:
        """Serialize model artifact."""
        joblib.dump(self, filepath)
        logger.info(f"Saved Random Forest model artifact to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "RandomForestSafetyModel":
        """Load serialized model artifact."""
        model = joblib.load(filepath)
        logger.info(f"Loaded Random Forest model from {filepath}")
        return model


class XGBoostSafetyModel:
    """Extreme Gradient Boosting classifier with cost-sensitive scale_pos_weight for safety prediction."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        eval_metric: str = "logloss",
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "eval_metric": eval_metric,
            "n_jobs": -1,
        }
        self.model: Optional[XGBClassifier] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostSafetyModel":
        """Fit XGBoost model with automatic scale_pos_weight calculation if not explicitly provided."""
        logger.info(f"Training XGBoost on {len(X_train)} rows...")
        self.feature_names_ = list(X_train.columns)

        # Calculate scale_pos_weight if needed: count(negative) / count(positive)
        if self.params["scale_pos_weight"] is None:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            calculated_ratio = float(n_neg) / float(max(1, n_pos))
            self.params["scale_pos_weight"] = round(calculated_ratio, 2)
            logger.info(f"Auto-calculated XGBoost scale_pos_weight: {self.params['scale_pos_weight']}")

        self.model = XGBClassifier(**self.params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary safety incident at specified probability threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities [P(safe), P(incident)]."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return gain-based feature importances as a pandas Series."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted to access feature importances.")
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)

    def save(self, filepath: str) -> None:
        """Serialize model artifact."""
        joblib.dump(self, filepath)
        logger.info(f"Saved XGBoost model artifact to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "XGBoostSafetyModel":
        """Load serialized model artifact."""
        model = joblib.load(filepath)
        logger.info(f"Loaded XGBoost model from {filepath}")
        return model
