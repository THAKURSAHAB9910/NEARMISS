"""Baseline models for comparative benchmarking in safety prediction.
Implements dummy heuristic prior rate and regularized logistic regression.
"""

import logging
from typing import Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class BaselineModel:
    """Benchmark baseline model providing comparison against advanced tree ensembles."""

    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state

        if model_type == "dummy_prior":
            self.model = DummyClassifier(strategy="prior", random_state=random_state)
        elif model_type == "dummy_uniform":
            self.model = DummyClassifier(strategy="uniform", random_state=random_state)
        elif model_type == "logistic_regression":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ))
            ])
        else:
            raise ValueError(f"Unknown baseline model_type: {model_type}")

        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """Fit the baseline model on training features and binary targets."""
        logger.info(f"Fitting BaselineModel ({self.model_type}) on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability distribution [P(0), P(1)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def save(self, filepath: str) -> None:
        """Serialize baseline model artifact."""
        joblib.dump(self, filepath)
        logger.info(f"Saved baseline model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BaselineModel":
        """Load serialized baseline model artifact."""
        model = joblib.load(filepath)
        logger.info(f"Loaded baseline model from {filepath}")
        return model
