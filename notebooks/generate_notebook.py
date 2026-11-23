"""Generates the reproducible exploratory data analysis notebook."""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NEARMISS — Exploratory Data Analysis & Safety Precursor Discovery\n",
            "This notebook investigates industrial telemetry, leading indicators, near-miss clusters, and equipment degradation patterns before incidents occur."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "PROJECT_ROOT = Path('..' if Path.cwd().name == 'notebooks' else '.').resolve()\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from src.data.ingestion import load_safety_data\n",
            "from src.data.validation import validate_raw_telemetry\n",
            "\n",
            "df = load_safety_data(config_path=str(PROJECT_ROOT / 'config.yaml'))\n",
            "print('Telemetry shape:', df.shape)\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Class Distribution and Target Incident Window\n",
            "Inspect class imbalance and incident occurrences in the 7-day lookahead window."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Target incident_next_7d value counts:')\n",
            "print(df['incident_next_7d'].value_counts(normalize=True))\n",
            "print('\\nTotal incidents logged:', df['incident_today'].sum())\n",
            "print('Total near-misses logged:', df['daily_near_miss_count'].sum())\n",
            "ratio = df['daily_near_miss_count'].sum() / max(1, df['incident_today'].sum())\n",
            "print(f'Empirical Heinrich Ratio: {ratio:.1f} near-misses per incident')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. ISO 10816 Vibration Severity Analysis\n",
            "Evaluate mechanical vibration velocity RMS against ISO 10816 thresholds (Alert: 2.8 mm/s, Danger: 4.5 mm/s)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(9, 4))\n",
            "sns.histplot(df['vibration_rms_mms'], kde=True, bins=40, color='#337ab7')\n",
            "plt.axvline(2.8, color='orange', linestyle='--', label='ISO Alert Threshold (2.8 mm/s)')\n",
            "plt.axvline(4.5, color='red', linestyle='--', label='ISO Danger Threshold (4.5 mm/s)')\n",
            "plt.title('Vibration Velocity RMS Distribution vs ISO 10816 Guidelines')\n",
            "plt.xlabel('Vibration Velocity RMS (mm/s)')\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Maintenance Backlog vs Safety Risk\n",
            "Inspect relationship between overdue maintenance days and incident occurrence."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(8, 4))\n",
            "sns.boxplot(x='incident_next_7d', y='pm_overdue_days', data=df, palette='Set2')\n",
            "plt.title('Preventive Maintenance Backlog (Overdue Days) by 7-Day Incident Outcome')\n",
            "plt.xlabel('Incident in Next 7 Days (0 = No, 1 = Yes)')\n",
            "plt.ylabel('Days PM Overdue')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.14.0"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/01_exploratory_data_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Saved notebook to", out_path)
