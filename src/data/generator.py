"""Domain-grounded industrial safety telemetry generator.

Explicitly generates synthetic benchmark data grounded in:
1. Heinrich's Safety Triangle (300 near-misses : 29 minor : 1 major incident)
2. Reason's Swiss Cheese Model (incident = alignment of mechanical, maintenance, and human factors)
3. ISO 10816 Mechanical Vibration Severity Guidelines
4. Circadian & Shift Fatigue Models
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


EQUIPMENT_PROFILES = {
    "Centrifugal Pump": {
        "base_rpm": 1780.0,
        "rpm_std": 35.0,
        "base_vibration": 1.4, # mm/s RMS
        "vibration_drift_rate": 0.02, # mm/s increase per week unmaintained
        "base_pressure": 8.5, # bar
        "pressure_std": 0.4,
        "base_temp_offset": 22.0,
        "base_acoustic": 62.0, # dB
    },
    "Gas Compressor": {
        "base_rpm": 2950.0,
        "rpm_std": 50.0,
        "base_vibration": 2.1,
        "vibration_drift_rate": 0.035,
        "base_pressure": 28.0,
        "pressure_std": 1.5,
        "base_temp_offset": 45.0,
        "base_acoustic": 78.0,
    },
    "CNC Machining Cell": {
        "base_rpm": 4200.0,
        "rpm_std": 120.0,
        "base_vibration": 1.1,
        "vibration_drift_rate": 0.015,
        "base_pressure": 4.5,
        "pressure_std": 0.2,
        "base_temp_offset": 15.0,
        "base_acoustic": 68.0,
    },
    "Overhead Crane": {
        "base_rpm": 650.0,
        "rpm_std": 80.0,
        "base_vibration": 0.9,
        "vibration_drift_rate": 0.012,
        "base_pressure": 2.0,
        "pressure_std": 0.1,
        "base_temp_offset": 10.0,
        "base_acoustic": 58.0,
    },
    "Heat Exchanger": {
        "base_rpm": 0.0, # static fluid asset
        "rpm_std": 0.0,
        "base_vibration": 0.4,
        "vibration_drift_rate": 0.005,
        "base_pressure": 14.0,
        "pressure_std": 0.8,
        "base_temp_offset": 55.0,
        "base_acoustic": 52.0,
    },
    "Conveyor Transfer Line": {
        "base_rpm": 450.0,
        "rpm_std": 15.0,
        "base_vibration": 1.6,
        "vibration_drift_rate": 0.025,
        "base_pressure": 3.0,
        "pressure_std": 0.2,
        "base_temp_offset": 12.0,
        "base_acoustic": 72.0,
    },
}

NEAR_MISS_CATEGORIES = [
    "None",
    "Mechanical_Guard_Trip",
    "Pressure_Excursion",
    "Thermal_Spike",
    "Slip_Trip_Fall_Hazard",
    "Chemical_Vapor_Weep",
    "Electrical_Arcing_Alert",
]


def generate_industrial_telemetry(config_path: str = "config.yaml") -> pd.DataFrame:
    """Generate longitudinal industrial safety and sensor telemetry across facilities."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    gen_cfg = cfg["generator"]
    facilities = gen_cfg["facilities"]
    equipment_types = gen_cfg["equipment_types"]
    start_date = pd.to_datetime(gen_cfg["start_date"])
    end_date = pd.to_datetime(gen_cfg["end_date"])
    lookahead_days = gen_cfg["lookahead_window_days"]

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)
    logger.info(f"Generating telemetry for {len(facilities)} facilities across {n_days} days...")

    records = []

    # Shift distribution
    shifts = ["Day", "Swing", "Night"]
    shift_probs = [0.45, 0.35, 0.20]

    for fac_idx, facility in enumerate(facilities):
        for eq_idx, eq_type in enumerate(equipment_types):
            unit_id = f"{facility}_{eq_type.replace(' ', '_').upper()}_{eq_idx+1:02d}"
            profile = EQUIPMENT_PROFILES[eq_type]

            # Maintenance state tracking
            days_since_pm = rng.integers(5, 60)
            pm_schedule_interval = rng.choice([60, 90, 120])
            open_tickets = rng.integers(0, 2)
            audit_score = rng.uniform(85, 98)

            # Equipment condition variables that drift
            degradation = rng.uniform(0.0, 0.2)

            for d_idx, current_date in enumerate(date_range):
                # Day of year seasonal factor
                day_of_year = current_date.dayofyear
                ambient_temp = 20.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + rng.normal(0, 2.5)

                # Maintenance aging and random PM service
                days_since_pm += 1
                pm_overdue_days = max(0, days_since_pm - pm_schedule_interval)

                # Periodic maintenance reset or repair
                if pm_overdue_days > 0 and rng.random() < 0.08:
                    days_since_pm = 0
                    open_tickets = max(0, open_tickets - rng.integers(1, 3))
                    degradation = max(0.0, degradation - 0.4)
                    audit_score = min(100.0, audit_score + rng.uniform(2, 6))
                elif rng.random() < 0.015: # Unplanned maintenance action
                    days_since_pm = 0
                    degradation = max(0.0, degradation - 0.3)

                # Progressive degradation drift
                degradation += profile["vibration_drift_rate"] * (1.0 + 0.5 * (pm_overdue_days > 0)) * (rng.uniform(0.7, 1.3))
                degradation = np.clip(degradation, 0.0, 4.0)

                # Shift assignments
                shift = rng.choice(shifts, p=shift_probs)
                is_night = 1 if shift == "Night" else 0
                is_weekend = 1 if current_date.weekday() >= 5 else 0

                # Duty cycle & overtime
                base_duty = 82.0 if is_night else 88.0
                duty_cycle_pct = np.clip(base_duty + rng.normal(0, 6.0) + (10.0 if pm_overdue_days > 20 else 0), 20.0, 100.0)
                
                overtime_hours = np.clip(rng.exponential(scale=1.5 if is_night or is_weekend else 0.8), 0.0, 14.0)

                # Operator fatigue index (0-100)
                fatigue_index = np.clip(
                    30.0
                    + (25.0 if is_night else 0.0)
                    + (overtime_hours * 3.8)
                    + (rng.normal(0, 5.0)),
                    10.0,
                    100.0
                )

                # Open maintenance tickets stochastic accumulation
                if rng.random() < 0.05 and pm_overdue_days > 0:
                    open_tickets = min(6, open_tickets + 1)

                # Physical sensor parameters
                # Vibration velocity RMS (mm/s)
                vibration_rms = profile["base_vibration"] + (degradation * 0.8) + rng.normal(0, 0.15)
                vibration_rms = max(0.2, vibration_rms)

                # Vibration kurtosis (peakedness, bearing impact detection)
                vibration_kurtosis = 3.0 + (degradation * 1.5) + rng.normal(0, 0.3)
                vibration_kurtosis = max(2.2, vibration_kurtosis)

                # Process temperature
                thermal_strain = (duty_cycle_pct / 100.0) * profile["base_temp_offset"] * (1.0 + degradation * 0.25)
                process_temp = ambient_temp + thermal_strain + rng.normal(0, 1.8)

                # Pressure
                pressure = profile["base_pressure"] * (1.0 + (duty_cycle_pct - 80.0) * 0.003) + rng.normal(0, profile["pressure_std"])
                pressure = max(0.5, pressure)

                # Rotational speed
                rpm = profile["base_rpm"] + rng.normal(0, profile["rpm_std"])
                if profile["base_rpm"] == 0:
                    rpm = 0.0

                # Acoustic emissions
                acoustic_db = profile["base_acoustic"] + (degradation * 4.5) + rng.normal(0, 1.5)

                # Safety audit score drift
                audit_score = np.clip(audit_score - (0.05 if open_tickets > 2 else 0.01) + rng.normal(0, 0.2), 65.0, 100.0)

                # Near-miss Poisson intensity (Heinrich leading indicator)
                # Hazard increases when vibration > alert, PM overdue, high fatigue, or open tickets
                base_nm_rate = 0.12 # average ~0.12 near-misses/day
                nm_multiplier = 1.0
                if vibration_rms > 2.8: # ISO alert
                    nm_multiplier *= 2.0
                if vibration_rms > 4.5: # ISO danger
                    nm_multiplier *= 3.5
                if pm_overdue_days > 14:
                    nm_multiplier *= 2.2
                if fatigue_index > 65:
                    nm_multiplier *= 1.8
                if open_tickets >= 3:
                    nm_multiplier *= 1.6
                if audit_score < 80:
                    nm_multiplier *= 1.5

                nm_lambda = base_nm_rate * nm_multiplier
                daily_near_misses = rng.poisson(lam=min(3.5, nm_lambda))

                if daily_near_misses > 0:
                    hazard_pool = [
                        "Mechanical_Guard_Trip",
                        "Pressure_Excursion",
                        "Thermal_Spike",
                        "Slip_Trip_Fall_Hazard",
                        "Chemical_Vapor_Weep",
                        "Electrical_Arcing_Alert",
                    ]
                    # Select category correlated with physical stress
                    if vibration_rms > 3.0:
                        nm_cat = "Mechanical_Guard_Trip"
                    elif process_temp > 65:
                        nm_cat = "Thermal_Spike"
                    elif pressure > profile["base_pressure"] * 1.2:
                        nm_cat = "Pressure_Excursion"
                    elif fatigue_index > 70:
                        nm_cat = "Slip_Trip_Fall_Hazard"
                    else:
                        nm_cat = rng.choice(hazard_pool)
                else:
                    nm_cat = "None"

                # Swiss Cheese Incident Hazard formulation for this day
                # True latent failure hazard:
                # Need multiple barriers to fail simultaneously
                barrier_1_mechanical = 1.0 if vibration_rms > 3.2 else 0.0
                barrier_2_maintenance = 1.0 if pm_overdue_days > 10 else 0.0
                barrier_3_human = 1.0 if fatigue_index > 60 else 0.0
                barrier_4_nearmiss = 1.0 if daily_near_misses >= 2 else (0.4 if daily_near_misses == 1 else 0.0)

                latent_score = (
                    0.35 * barrier_1_mechanical
                    + 0.30 * barrier_2_maintenance
                    + 0.20 * barrier_3_human
                    + 0.25 * barrier_4_nearmiss
                    + (0.15 if vibration_rms > 4.5 else 0.0)
                    + (0.15 if open_tickets >= 3 else 0.0)
                )

                # Incident probability today
                daily_incident_prob = 0.002 + 0.16 * np.power(min(1.0, latent_score / 1.1), 2.5)
                incident_today = 1 if rng.random() < daily_incident_prob else 0

                records.append({
                    "record_id": f"REC_{facility[:3].upper()}_{unit_id[-7:]}_{current_date.strftime('%Y%m%d')}",
                    "timestamp": current_date.strftime("%Y-%m-%d"),
                    "facility": facility,
                    "unit_id": unit_id,
                    "equipment_type": eq_type,
                    "shift_type": shift,
                    "ambient_temperature_c": round(ambient_temp, 2),
                    "process_temperature_c": round(process_temp, 2),
                    "vibration_rms_mms": round(vibration_rms, 3),
                    "vibration_kurtosis": round(vibration_kurtosis, 3),
                    "operating_pressure_bar": round(pressure, 2),
                    "rotational_speed_rpm": round(rpm, 1),
                    "duty_cycle_pct": round(duty_cycle_pct, 1),
                    "acoustic_emission_db": round(acoustic_db, 1),
                    "days_since_last_pm": int(days_since_pm),
                    "pm_overdue_days": int(pm_overdue_days),
                    "open_maintenance_tickets": int(open_tickets),
                    "daily_near_miss_count": int(daily_near_misses),
                    "near_miss_category": nm_cat,
                    "shift_overtime_hours": round(overtime_hours, 2),
                    "operator_fatigue_index": round(fatigue_index, 1),
                    "safety_audit_score": round(audit_score, 1),
                    "incident_today": int(incident_today),
                    "data_origin": "domain_grounded_synthetic_benchmark"
                })

    df = pd.DataFrame(records)
    logger.info(f"Total raw observations generated: {len(df)}")

    # Calculate strictly forward-looking target variables without temporal leakage!
    # Target: Will an incident occur in the NEXT 7 DAYS [t+1, t+7] for this unit?
    logger.info("Computing strictly forward-looking lookahead target windows [t+1, t+7]...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["unit_id", "timestamp"]).reset_index(drop=True)

    incident_next_7d = []
    days_to_incident = []

    for unit_id, group in df.groupby("unit_id", sort=False):
        group_incidents = group["incident_today"].values
        n_obs = len(group_incidents)

        for i in range(n_obs):
            # Future slice strictly after today: i+1 to i+1+lookahead_days
            future_slice = group_incidents[i+1 : min(n_obs, i + 1 + lookahead_days)]
            if len(future_slice) == 0:
                # Boundary condition at the very end of time series
                incident_next_7d.append(0)
                days_to_incident.append(30) # Censored
            else:
                has_inc = int(np.sum(future_slice) > 0)
                incident_next_7d.append(has_inc)
                if has_inc:
                    # Days until the first incident
                    first_idx = np.where(future_slice == 1)[0][0] + 1
                    days_to_incident.append(first_idx)
                else:
                    days_to_incident.append(30) # Censored at max window

    df["incident_next_7d"] = incident_next_7d
    df["days_to_incident"] = days_to_incident
    # Event indicator for survival analysis: 1 if event observed within window, 0 if censored
    df["survival_event"] = df["incident_next_7d"]

    pos_rate = df["incident_next_7d"].mean() * 100
    logger.info(f"Lookahead target computed. Positive rate (incident in next 7d): {pos_rate:.2f}%")

    # Save to data/synthetic/
    synthetic_dir = Path(cfg["paths"]["synthetic_data_dir"])
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    out_path = synthetic_dir / "industrial_safety_telemetry.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved benchmark telemetry to {out_path} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    generate_industrial_telemetry()
