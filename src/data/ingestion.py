"""Data ingestion module for NEARMISS.
Handles loading telemetry from files or database connections and enforces validation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import yaml

from src.data.validation import validate_raw_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_safety_data(
    file_path: Optional[str] = None,
    config_path: str = "config.yaml",
    validate: bool = True
) -> pd.DataFrame:
    """Load industrial telemetry data, validating against physical constraints."""
    if file_path is None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        synthetic_dir = Path(cfg["paths"]["synthetic_data_dir"])
        target_csv = synthetic_dir / "industrial_safety_telemetry.csv"
        if not target_csv.exists():
            from src.data.generator import generate_industrial_telemetry
            logger.info(f"Target data not found at {target_csv}. Generating benchmark telemetry...")
            df = generate_industrial_telemetry(config_path)
            return df
        file_path = str(target_csv)

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

    logger.info(f"Loading safety telemetry from {path}...")
    df = pd.read_csv(path)

    if validate:
        df = validate_raw_telemetry(df)

    return df


def split_data_temporal(
    df: pd.DataFrame,
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data temporally into Train, Validation, and Test sets to prevent data leakage."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_end = pd.to_datetime(cfg["splitting"]["train_end_date"])
    val_end = pd.to_datetime(cfg["splitting"]["val_end_date"])
    test_end = pd.to_datetime(cfg["splitting"]["test_end_date"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    train_mask = df["timestamp"] <= train_end
    val_mask = (df["timestamp"] > train_end) & (df["timestamp"] <= val_end)
    test_mask = (df["timestamp"] > val_end) & (df["timestamp"] <= test_end)

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    logger.info(
        f"Temporal split complete without leakage:\n"
        f"  Train: {len(train_df)} rows ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})\n"
        f"  Val:   {len(val_df)} rows ({val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()})\n"
        f"  Test:  {len(test_df)} rows ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})"
    )

    return train_df, val_df, test_df
