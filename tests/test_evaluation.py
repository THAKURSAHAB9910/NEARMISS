"""Unit tests for evaluation metrics, confusion matrix, and cost functions."""

import numpy as np
import pytest
from src.evaluation.metrics import evaluate_binary_predictions


def test_evaluate_binary_predictions_perfect():
    """Test metrics for perfect prediction."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.9, 0.95])

    m = evaluate_binary_predictions(y_true, y_pred, y_prob, cost_fn=50000, cost_fp=1500)
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["false_negatives"] == 0
    assert m["false_positives"] == 0
    assert m["total_cost"] == 0.0
    assert m["cost_reduction_pct"] == 100.0


def test_evaluate_binary_predictions_all_negative():
    """Test metrics when naive classifier predicts all negative."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1])

    m = evaluate_binary_predictions(y_true, y_pred, y_prob, cost_fn=50000, cost_fp=1500)
    assert m["recall"] == 0.0
    assert m["false_negatives"] == 2
    assert m["total_cost"] == 100000.0 # 2 * 50,000
    assert m["cost_reduction_pct"] == 0.0
