"""Industrial safety metrics computation module.
Computes Precision, Recall, F1, PR-AUC, ROC-AUC, Brier score, confusion matrix,
and financial/safety operational cost analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    cost_fn: float = 50000.0,
    cost_fp: float = 1500.0,
) -> Dict[str, Any]:
    """Compute comprehensive performance and operational cost metrics for a binary safety classifier."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Operational safety cost calculation
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    unmitigated_cost = np.sum(y_true == 1) * cost_fn
    cost_reduction_pct = ((unmitigated_cost - total_cost) / unmitigated_cost * 100.0) if unmitigated_cost > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "balanced_accuracy": float(bal_acc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_cost": float(total_cost),
        "cost_reduction_pct": float(cost_reduction_pct),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }

    # Probability-dependent metrics
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = 0.5

            try:
                metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            except Exception:
                metrics["pr_auc"] = float(np.mean(y_true))

        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    return metrics


def compute_comprehensive_metrics(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = 0.35,
    cost_fn: float = 50000.0,
    cost_fp: float = 1500.0,
) -> pd.DataFrame:
    """Compare multiple models across all safety metrics on a common test dataset."""
    results = []
    y_test_arr = y_test.values.astype(int)

    for name, model in models_dict.items():
        # Get probabilities
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = None
        except Exception:
            probs = None

        # Get binary predictions
        if hasattr(model, "predict"):
            if "threshold" in model.predict.__code__.co_varnames:
                preds = model.predict(X_test, threshold=decision_threshold)
            else:
                preds = model.predict(X_test)
        elif probs is not None:
            preds = (probs >= decision_threshold).astype(int)
        else:
            preds = np.zeros(len(y_test_arr), dtype=int)

        m = evaluate_binary_predictions(
            y_true=y_test_arr,
            y_pred=preds,
            y_prob=probs,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        m["model_name"] = name
        results.append(m)

    df_results = pd.DataFrame(results)
    # Reorder columns with model_name first
    cols = ["model_name", "pr_auc", "recall", "precision", "f1_score", "roc_auc", "brier_score", "false_negatives", "false_positives", "total_cost"]
    existing_cols = [c for c in cols if c in df_results.columns] + [c for c in df_results.columns if c not in cols]
    return df_results[existing_cols]
