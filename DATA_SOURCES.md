# DATA_SOURCES.md — NEARMISS Data Catalog & Provenance

## 1. Overview
The **NEARMISS** system predicts imminent occupational and operational safety risks across industrial facilities. Identifying leading indicators (near-miss clusters, deferred preventive maintenance, sensor anomalies, and human fatigue) requires longitudinal, time-series telemetry linked to safety incident registries.

This document details:
1. Publicly accessible industrial safety and equipment datasets evaluated.
2. Verified URLs, licences, schemas, and limitations.
3. Rationale for combining external benchmarks with an explicitly labelled, physics-grounded synthetic telemetry generator.
4. Data separation and transparency guarantees.

---

## 2. Public Industrial Datasets Investigated

### 2.1 UCI AI4I 2020 Predictive Maintenance Dataset
- **Provider**: UCI Machine Learning Repository / Stephan Matzka (Hochschule für Technik und Wirtschaft Berlin)
- **URL**: [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Direct Download URL**: `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Sample Size**: 10,000 rows × 14 columns
- **Key Columns**:
  - `UDI`: Unique identifier (1 to 10,000)
  - `Product ID`: Variant identifier (L/M/H quality variants)
  - `Air temperature [K]`: Generated via random walk around 300 K
  - `Process temperature [K]`: Air temperature + 10 K plus operational variance
  - `Rotational speed [rpm]`: Calculated from power balance
  - `Torque [Nm]`: Torque distributed around 40 Nm
  - `Tool wear [min]`: Cumulative minutes of machining wear
  - `Machine failure`: Binary target (1 = Failure, 0 = Normal, 3.39% positive rate)
  - Failure modes: `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failure)
- **Storage Location**: `data/external/ai4i_2020/ai4i2020.csv`
- **Utility**: Validates machine failure physics, torque/speed power balances, and severe class imbalance (~3.4%).
- **Limitations**:
  - Cross-sectional snapshots per milling pass; lacks continuous plant-level timestamps.
  - Does not contain safety hazard near-miss reporting or human factor variables (e.g., operator fatigue, shift rotation).

---

### 2.2 OSHA Severe Injury Reports (SIR)
- **Provider**: U.S. Department of Labor, Occupational Safety and Health Administration (OSHA)
- **URL**: [https://www.osha.gov/severe-injury-reports](https://www.osha.gov/severe-injury-reports)
- **Licence**: U.S. Government Work (Public Domain)
- **Scope**: ~80,000+ severe occupational injury events since 2015 (in-patient hospitalizations, amputations, eye loss).
- **Key Columns**: Event date, Employer name, City/State, NAICS code, Inspection number, Narrative description, Nature of injury, Source of injury.
- **Limitations**:
  - Pure incident registry (lagging indicators only).
  - No continuous baseline operational telemetry (no records of safe working shifts or equipment operating hours).
  - Near-miss events are not federally reportable to OSHA and therefore absent from public records.

---

### 2.3 MSHA Accidents, Injuries, and Illnesses Database
- **Provider**: U.S. Department of Labor, Mine Safety and Health Administration (MSHA)
- **URL**: [https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp](https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp)
- **Licence**: U.S. Government Work (Public Domain)
- **Scope**: Decades of mining accidents, fatalities, non-fatal injuries, and mining operator logs.
- **Limitations**:
  - Event-based reporting; lacks high-frequency machinery sensor traces and shift-level near-miss logs across consistent time windows.

---

### 2.4 NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset
- **Provider**: NASA Ames Prognostics Center of Excellence (PCoE)
- **URL**: [https://data.nasa.gov](https://data.nasa.gov) / NASA Open Data
- **Licence**: NASA Open Source Agreement / Public Domain
- **Scope**: Run-to-failure simulations of turbofan engines under varying flight conditions.
- **Limitations**:
  - Pure aerospace asset degradation; does not capture occupational safety, human factors, or near-miss hazard logs.

---

## 3. The Industrial Safety Data Gap & Rationale for Hybrid Modelling

In industrial enterprises, continuous sensor streams, near-miss incident logs, and maintenance work orders are protected by strict liability and confidentiality agreements. Public safety portals (OSHA/MSHA) only publish severe injuries *post-facto*, while equipment degradation benchmarks (AI4I, C-MAPSS) focus strictly on mechanical breakdown without human-machine safety context.

To evaluate a **Predictive Industrial Safety Intelligence System** that estimates the probability of an incident over a future 7-day operational window, we implement:

1. **External Real Public Reference**:
   - `data/external/ai4i_2020/ai4i2020.csv` is preserved in its original form and used as an external machine failure benchmark.
2. **Domain-Grounded Synthetic Telemetry Benchmark**:
   - Explicitly created in `data/synthetic/industrial_safety_telemetry.csv`.
   - Built via `src/data/generator.py` using established industrial safety principles:
     - **Heinrich's Triangle / Frank Bird's Safety Pyramid**: For every 1 severe incident, there are ~10 minor incidents and ~300 near-misses.
     - **Reason's Swiss Cheese Model**: Catastrophes happen when mechanical defects, deferred maintenance, environmental extremes, and human fatigue align simultaneously.
     - **ISO 10816 Mechanical Vibration Severity Guidelines**: Rotating machinery vibration RMS thresholds (Normal < 2.8 mm/s, Alert 2.8-4.5 mm/s, Danger > 4.5 mm/s).
     - **Circadian & Shift Work Fatigue Modeling**: Night shifts and extended overtime elevate cognitive error and near-miss frequency.

---

## 4. Dataset Transparency & Separation Policy

- **No Real-World Data Fabrication**: External benchmark files in `data/external/` are downloaded directly from authenticated sources and verified against their source hashes.
- **Explicit Labelling**: All generated records reside strictly under `data/synthetic/` and declare `data_origin: "domain_grounded_synthetic_benchmark"` in metadata.
- **Reproducibility**: The synthetic dataset is generated using fixed pseudo-random seeds (`SEED=42`), allowing exact re-creation.
- **No Private Data**: No proprietary credentials, private worker names, or confidential operational data are used.
