"""SHAP explainability engine for tree ensemble safety risk forecasting.
Provides global feature attribution and local instance root-cause explanations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class SafetySHAPExplainer:
    """Wrapper around SHAP TreeExplainer for industrial safety root-cause attribution."""

    def __init__(self, model_wrapper, feature_names: Optional[List[str]] = None):
        self.model_wrapper = model_wrapper
        # Extract underlying tree model
        if hasattr(model_wrapper, "model") and model_wrapper.model is not None:
            self.model = model_wrapper.model
        else:
            self.model = model_wrapper

        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self.expected_value: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, background_data: Optional[pd.DataFrame] = None) -> "SafetySHAPExplainer":
        """Initialize SHAP TreeExplainer with model and optional background sample."""
        logger.info("Initializing SHAP TreeExplainer...")
        try:
            if background_data is not None:
                bg_sample = background_data.sample(min(100, len(background_data)), random_state=42)
                self.explainer = shap.TreeExplainer(self.model, data=bg_sample)
            else:
                self.explainer = shap.TreeExplainer(self.model)

            # Store expected value (prior base value)
            ev = self.explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                # For binary classification, take positive class index 1 if 2 classes, else scalar
                self.expected_value = float(ev[1] if len(ev) > 1 else ev[0])
            else:
                self.expected_value = float(ev)

            self.is_fitted = True
            logger.info(f"SHAP TreeExplainer initialized. Base expected value: {self.expected_value:.4f}")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP TreeExplainer: {e}")
            raise e

        return self

    def compute_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute SHAP values for a feature matrix X."""
        if not self.is_fitted or self.explainer is None:
            self.fit()

        feat_names = list(X.columns) if self.feature_names is None else self.feature_names
        shap_values = self.explainer.shap_values(X, check_additivity=False)

        # For binary classification in some tree models, shap_values might be a list [class_0, class_1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_pos = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_vals_pos = shap_values[:, :, 1]
        else:
            shap_vals_pos = np.asarray(shap_values)

        df_shap = pd.DataFrame(shap_vals_pos, columns=feat_names, index=X.index)
        return shap_vals_pos, df_shap

    def explain_instance(
        self,
        row_features: pd.Series,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """Explain an individual equipment/shift observation."""
        if not self.is_fitted or self.explainer is None:
            self.fit()

        df_row = pd.DataFrame([row_features])
        feat_names = list(df_row.columns) if self.feature_names is None else self.feature_names

        shap_vals, df_shap = self.compute_shap_values(df_row)
        shap_row = df_shap.iloc[0]

        # Top drivers elevating risk (positive SHAP)
        risk_elevators = shap_row[shap_row > 0].sort_values(ascending=False).head(top_n)
        top_risks = [
            {
                "feature": feat,
                "value": float(row_features.get(feat, 0.0)),
                "shap_impact": float(val),
                "direction": "INCREASES_RISK",
            }
            for feat, val in risk_elevators.items()
        ]

        # Top factors mitigating risk (negative SHAP)
        risk_mitigators = shap_row[shap_row < 0].sort_values().head(top_n)
        top_mitigations = [
            {
                "feature": feat,
                "value": float(row_features.get(feat, 0.0)),
                "shap_impact": float(val),
                "direction": "DECREASES_RISK",
            }
            for feat, val in risk_mitigators.items()
        ]

        return {
            "base_expected_value": self.expected_value,
            "total_shap_sum": float(shap_row.sum()),
            "top_risk_drivers": top_risks,
            "top_mitigating_factors": top_mitigations,
            "feature_contributions": shap_row.to_dict(),
        }

    def generate_global_summary_plots(
        self,
        X_sample: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, str]:
        """Generate and save SHAP summary and bar plots to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shap_vals_pos, df_shap = self.compute_shap_values(X_sample)

        # 1. Feature Importance Bar Plot (Mean |SHAP|)
        bar_plot_path = output_dir / "shap_summary_bar.png"
        plt.figure(figsize=(10, 6))
        mean_abs = np.abs(shap_vals_pos).mean(axis=0)
        top_indices = np.argsort(mean_abs)[-15:]

        y_pos = np.arange(len(top_indices))
        plt.barh(y_pos, mean_abs[top_indices], color="#d9534f", edgecolor="black", alpha=0.85)
        plt.yticks(y_pos, [X_sample.columns[i] for i in top_indices], fontsize=9)
        plt.xlabel("Mean |SHAP Value| (Impact on Safety Incident Probability)")
        plt.title("NEARMISS: Top 15 Global Safety Risk Drivers (SHAP Feature Importance)")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(bar_plot_path, dpi=200)
        plt.close()

        # 2. SHAP Beeswarm Summary Plot
        beeswarm_path = output_dir / "shap_beeswarm.png"
        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_vals_pos,
            X_sample,
            max_display=15,
            show=False,
            plot_size=(11, 7)
        )
        plt.title("SHAP Beeswarm: How Operational Signals Shift Incident Risk", fontsize=12)
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=200)
        plt.close()

        logger.info(f"Saved SHAP plots to {bar_plot_path} and {beeswarm_path}")
        return {
            "shap_bar_path": str(bar_plot_path),
            "shap_beeswarm_path": str(beeswarm_path),
        }
