"""Utility script to download and verify publicly accessible benchmark datasets.
Includes the UCI AI4I 2020 Predictive Maintenance dataset and documents public safety registries.
"""

import io
import logging
import zipfile
from pathlib import Path
import urllib.request
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UCI_AI4I_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def download_uci_ai4i(output_dir: Path) -> Path:
    """Download and extract the UCI AI4I 2020 Predictive Maintenance dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_target = output_dir / "ai4i2020.csv"

    if csv_target.exists():
        logger.info(f"UCI AI4I dataset already exists at {csv_target}")
        return csv_target

    logger.info(f"Downloading UCI AI4I dataset from {UCI_AI4I_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(UCI_AI4I_URL, headers=headers)

    with urllib.request.urlopen(req, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for filename in z.namelist():
            if filename.endswith(".csv"):
                content = z.read(filename)
                with open(csv_target, "wb") as f:
                    f.write(content)
                logger.info(f"Extracted {filename} -> {csv_target}")
            elif filename.endswith(".txt") or filename.endswith(".json"):
                meta_file = output_dir / Path(filename).name
                with open(meta_file, "wb") as f:
                    f.write(z.read(filename))

    if not csv_target.exists():
        raise FileNotFoundError(f"Failed to extract ai4i2020.csv from archive.")

    df = pd.read_csv(csv_target)
    logger.info(f"Successfully loaded UCI AI4I dataset with shape {df.shape} and columns: {list(df.columns)}")
    return csv_target


if __name__ == "__main__":
    ext_dir = Path("data/external/ai4i_2020")
    download_uci_ai4i(ext_dir)
