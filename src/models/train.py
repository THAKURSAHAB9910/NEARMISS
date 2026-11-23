"""Master training orchestrator for NEARMISS.
Executes temporal feature engineering, trains all models, performs probability calibration,
fits anomaly detectors and survival models, and persists all artifacts.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml

from src.data.ingestion import load_safety_data, split_data_temporal
from src.features.pipeline import SafetyFeaturePipeline
from src.models.baseline import BaselineModel
from src.models.ml_models import RandomForestSafetyModel, XGBoostSafetyModel
from src.models.anomaly import OperationalAnomalyDetector
from src.models.calibration import CalibratedSafetyModel
from src.models.survival import SafetySurvivalAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute end-to-end training pipeline and return trained artifacts."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and temporally split data
    logger.info("=== STEP 1: Ingestion and Leak-Free Temporal Splitting ===")
    df = load_safety_data(config_path=config_path)
    train_df, val_df, test_df = split_data_temporal(df, config_path=config_path)

    # 2. Feature Engineering Pipeline
    logger.info("=== STEP 2: Feature Engineering & Preprocessing ===")
    pipeline = SafetyFeaturePipeline(config_path=config_path)
    X_train, y_train, meta_train = pipeline.fit_transform(train_df)
    X_val, y_val, meta_val = pipeline.transform(val_df, return_targets=True)
    X_test, y_test, meta_test = pipeline.transform(test_df, return_targets=True)

    pipeline_path = models_dir / "feature_pipeline.joblib"
    pipeline.save(pipeline_path)

    # 3. Train Baseline Models
    logger.info("=== STEP 3: Training Baseline Models ===")
    baseline_dummy = BaselineModel(model_type="dummy_prior", random_state=cfg["project"]["random_seed"])
    baseline_dummy.fit(X_train, y_train)
    baseline_dummy.save(str(models_dir / "baseline_dummy.joblib"))

    baseline_logreg = BaselineModel(model_type="logistic_regression", random_state=cfg["project"]["random_seed"])
    baseline_logreg.fit(X_train, y_train)
    baseline_logreg.save(str(models_dir / "baseline_logreg.joblib"))

    # 4. Train Random Forest
    logger.info("=== STEP 4: Training Random Forest Ensemble ===")
    rf_cfg = cfg["models"]["random_forest"]
    rf_model = RandomForestSafetyModel(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight=rf_cfg["class_weight"],
        random_state=rf_cfg["random_state"],
    )
    rf_model.fit(X_train, y_train)
    rf_model.save(str(models_dir / "random_forest.joblib"))

    # 5. Train XGBoost
    logger.info("=== STEP 5: Training XGBoost Ensemble ===")
    xgb_cfg = cfg["models"]["xgboost"]
    xgb_model = XGBoostSafetyModel(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=xgb_cfg.get("scale_pos_weight"),
        random_state=xgb_cfg["random_state"],
        eval_metric=xgb_cfg["eval_metric"],
    )
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_model.save(str(models_dir / "xgboost.joblib"))

    # 6. Train Operational Anomaly Detector (Isolation Forest)
    logger.info("=== STEP 6: Training Operational Anomaly Detector ===")
    anom_cfg = cfg["models"]["anomaly_detector"]
    anomaly_detector = OperationalAnomalyDetector(
        contamination=anom_cfg["contamination"],
        n_estimators=anom_cfg["n_estimators"],
        random_state=anom_cfg["random_state"],
    )
    anomaly_detector.fit(X_train)
    anomaly_detector.save(str(models_dir / "anomaly_detector.joblib"))

    # 7. Probability Calibration
    logger.info("=== STEP 7: Calibrating Probabilities for Decision Support ===")
    calibrator = CalibratedSafetyModel(
        base_estimator=xgb_model,
        method=cfg["models"]["calibration"]["method"],
    )
    calibrator.fit_calibration(
        X_val,
        y_val,
        cost_fn=cfg["evaluation"]["cost_false_negative"],
        cost_fp=cfg["evaluation"]["cost_false_positive"],
    )
    calibrator.save(str(models_dir / "calibrated_xgb.joblib"))

    # 8. Survival Analysis Modeling
    logger.info("=== STEP 8: Fitting Time-to-Event Survival Models ===")
    survival_analyzer = SafetySurvivalAnalyzer()
    survival_analyzer.fit(train_df)
    survival_analyzer.save(str(models_dir / "survival_analyzer.joblib"))

    # Save processed splits for fast re-use in evaluation and testing
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(processed_dir / "X_train.parquet")
    y_train.to_frame("target").to_parquet(processed_dir / "y_train.parquet")
    X_val.to_parquet(processed_dir / "X_val.parquet")
    y_val.to_frame("target").to_parquet(processed_dir / "y_val.parquet")
    X_test.to_parquet(processed_dir / "X_test.parquet")
    y_test.to_frame("target").to_parquet(processed_dir / "y_test.parquet")
    meta_test.to_parquet(processed_dir / "meta_test.parquet")

    logger.info("=== ALL MODEL ARTIFACTS AND PROCESSED DATA SUCCESSFULLY PERSISTED ===")
    return {
        "pipeline": pipeline,
        "baseline_dummy": baseline_dummy,
        "baseline_logreg": baseline_logreg,
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "anomaly_detector": anomaly_detector,
        "calibrator": calibrator,
        "survival_analyzer": survival_analyzer,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


if __name__ == "__main__":
    run_training_pipeline()
