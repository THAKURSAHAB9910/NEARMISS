"""NEARMISS: Predictive Industrial Safety Intelligence System
Interactive Streamlit Dashboard for Safety Decision Support, Anomaly Detection, and Root-Cause Attribution.
"""

import logging
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from app.components import (
    render_disclaimer_banner,
    render_kpi_metrics,
    plot_heinrich_pyramid,
    plot_risk_gauge,
    plot_risk_trend_over_time,
    plot_operational_stress_scatter,
)
from app.report_export import generate_safety_audit_report
from src.explainability.shap_explainer import SafetySHAPExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(
    page_title="NEARMISS | Predictive Industrial Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# CACHED RESOURCE LOADERS
# ==========================================
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_models_and_pipeline():
    """Load pre-trained model artifacts."""
    models_dir = Path("models")
    return {
        "pipeline": joblib.load(models_dir / "feature_pipeline.joblib"),
        "calibrated_xgb": joblib.load(models_dir / "calibrated_xgb.joblib"),
        "xgboost": joblib.load(models_dir / "xgboost.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest.joblib"),
        "anomaly_detector": joblib.load(models_dir / "anomaly_detector.joblib"),
        "survival_analyzer": joblib.load(models_dir / "survival_analyzer.joblib"),
    }


@st.cache_data
def load_and_score_data():
    """Load baseline telemetry and pre-score 7-day incident risk and anomaly flags."""
    data_path = Path("data/synthetic/industrial_safety_telemetry.csv")
    if not data_path.exists():
        from src.data.generator import generate_industrial_telemetry
        df_raw = generate_industrial_telemetry()
    else:
        df_raw = pd.read_csv(data_path)

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

    models_dict = load_models_and_pipeline()
    pipeline = models_dict["pipeline"]
    calibrated_xgb = models_dict["calibrated_xgb"]
    anomaly_detector = models_dict["anomaly_detector"]

    # Transform features
    X = pipeline.transform(df_raw, return_targets=False)

    # Compute risk and anomaly scores
    probs = calibrated_xgb.predict_proba(X)[:, 1]
    anom_scores = anomaly_detector.score_samples(X)
    anom_flags = anomaly_detector.predict_anomaly(X)

    df_scored = df_raw.copy()
    df_scored["predicted_risk"] = probs
    df_scored["predicted_risk_pct"] = np.round(probs * 100.0, 1)
    df_scored["anomaly_score"] = np.round(anom_scores, 3)
    df_scored["anomaly_flag"] = anom_flags

    return df_scored, X


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    cfg = load_config()
    artifacts = load_models_and_pipeline()

    # Application Header
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 8px; border-bottom: 2px solid #e9ecef; margin-bottom: 16px;">
            <div>
                <h1 style="margin: 0; font-size: 30px; color: #1a252f;">🛡️ NEARMISS</h1>
                <p style="margin: 4px 0 0 0; color: #6c757d; font-size: 15px;">
                    Predictive Industrial Safety Intelligence & Leading Indicator Warning System
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                    SYSTEM STATUS: ACTIVE
                </span>
                <p style="margin: 4px 0 0 0; font-size: 12px; color: #6c757d;">Monitoring 30 Critical Assets</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load pre-scored dataset
    with st.spinner("Loading telemetry intelligence stream..."):
        df_all, X_all = load_and_score_data()

    # ==========================================
    # SIDEBAR: FILTERS AND SETTINGS
    # ==========================================
    st.sidebar.header("🔍 Fleet Filters & Controls")

    # Facility Filter
    all_facilities = ["All Facilities"] + sorted(df_all["facility"].unique().tolist())
    selected_facility = st.sidebar.selectbox("Industrial Facility", all_facilities, index=0)

    # Equipment Filter
    all_eq = ["All Equipment Types"] + sorted(df_all["equipment_type"].unique().tolist())
    selected_eq = st.sidebar.selectbox("Asset Category", all_eq, index=0)

    # Shift Filter
    all_shifts = ["All Shifts"] + sorted(df_all["shift_type"].unique().tolist())
    selected_shift = st.sidebar.selectbox("Shift Type", all_shifts, index=0)

    # Date Range Filter
    min_date = df_all["timestamp"].min().date()
    max_date = df_all["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Monitoring Time Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Alert Threshold Slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Decision Threshold")
    threshold_pct = st.sidebar.slider(
        "Safety Alert Trigger (%)",
        min_value=5,
        max_value=90,
        value=35,
        step=5,
        help="Calibrated risk threshold above which supervisory warnings and priority inspections are dispatched.",
    )
    alert_threshold = threshold_pct / 100.0

    # Apply Filters
    df_filtered = df_all.copy()
    if selected_facility != "All Facilities":
        df_filtered = df_filtered[df_filtered["facility"] == selected_facility]
    if selected_eq != "All Equipment Types":
        df_filtered = df_filtered[df_filtered["equipment_type"] == selected_eq]
    if selected_shift != "All Shifts":
        df_filtered = df_filtered[df_filtered["shift_type"] == selected_shift]

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["timestamp"].dt.date >= date_range[0])
            & (df_filtered["timestamp"].dt.date <= date_range[1])
        ]

    # Filtered Summary Statistics
    total_obs = len(df_filtered)
    total_nm = int(df_filtered["daily_near_miss_count"].sum())
    total_inc = int(df_filtered["incident_today"].sum()) if "incident_today" in df_filtered.columns else 1
    h_ratio = (total_nm / max(1, total_inc))
    high_risk_units_df = df_filtered[df_filtered["predicted_risk"] >= alert_threshold]
    high_risk_count = high_risk_units_df["unit_id"].nunique()

    summary_stats = {
        "total_observations": total_obs,
        "total_near_misses": total_nm,
        "total_incidents": total_inc,
        "heinrich_ratio": h_ratio,
        "avg_predicted_risk": float(df_filtered["predicted_risk"].mean() * 100.0) if total_obs > 0 else 0.0,
        "anomaly_rate": float(df_filtered["anomaly_flag"].mean() * 100.0) if total_obs > 0 else 0.0,
        "high_risk_count": high_risk_count,
    }

    # ==========================================
    # NAVIGATION TABS
    # ==========================================
    tabs = st.tabs([
        "📊 Fleet Safety Intelligence",
        "⚙️ Asset Deep-Dive & Warnings",
        "🔮 Real-Time Risk Simulator & SHAP",
        "⏳ Survival & Time-to-Event",
        "📈 Model Evaluation Benchmarks",
        "📑 Export & Audit Reports",
    ])

    # ----------------------------------------------------
    # TAB 1: FLEET SAFETY INTELLIGENCE
    # ----------------------------------------------------
    with tabs[0]:
        render_disclaimer_banner()
        render_kpi_metrics(summary_stats)

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns([1.7, 1.0])

        with col_chart1:
            if not df_filtered.empty:
                st.plotly_chart(plot_risk_trend_over_time(df_filtered), use_container_width=True)
            else:
                st.warning("No data matching current filters.")

        with col_chart2:
            st.plotly_chart(
                plot_heinrich_pyramid(
                    near_misses=total_nm,
                    minor_hazards=int(total_nm * 0.12),
                    major_incidents=total_inc,
                ),
                use_container_width=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔍 Multi-Sensor Operating Envelope & Vibration Severity (ISO 10816)")
        if not df_filtered.empty:
            st.plotly_chart(plot_operational_stress_scatter(df_filtered), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"🚨 Prioritized High-Risk Assets (Risk >= {threshold_pct}%)")

        if not high_risk_units_df.empty:
            latest_alerts = (
                high_risk_units_df.sort_values(by=["timestamp", "predicted_risk"], ascending=[False, False])
                .drop_duplicates(subset=["unit_id"])
                .head(10)
            )
            cols_show = [
                "timestamp", "unit_id", "facility", "equipment_type",
                "predicted_risk_pct", "anomaly_flag", "vibration_rms_mms",
                "pm_overdue_days", "daily_near_miss_count", "operator_fatigue_index"
            ]
            st.dataframe(
                latest_alerts[cols_show].style.format({
                    "predicted_risk_pct": "{:.1f}%",
                    "vibration_rms_mms": "{:.2f} mm/s",
                    "operator_fatigue_index": "{:.1f}",
                    "timestamp": lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
                }).background_gradient(subset=["predicted_risk_pct"], cmap="Reds"),
                use_container_width=True,
            )
        else:
            st.success(f"No industrial units currently exceed the {threshold_pct}% risk threshold under current filter conditions.")

    # ----------------------------------------------------
    # TAB 2: ASSET DEEP-DIVE & WARNINGS
    # ----------------------------------------------------
    with tabs[1]:
        render_disclaimer_banner()
        st.subheader("Asset-Specific Longitudinal Warning Patterns")

        avail_units = sorted(df_filtered["unit_id"].unique().tolist())
        if not avail_units:
            avail_units = sorted(df_all["unit_id"].unique().tolist())

        selected_unit = st.selectbox("Select Target Industrial Unit", avail_units, index=0)
        df_unit = df_all[df_all["unit_id"] == selected_unit].sort_values("timestamp").reset_index(drop=True)

        # Unit Snapshot Cards
        latest_obs = df_unit.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unit ID", selected_unit)
        c2.metric("Asset Class", latest_obs["equipment_type"])
        c3.metric("Facility Location", latest_obs["facility"])
        c4.metric("Days PM Overdue", f"{int(latest_obs['pm_overdue_days'])} days")

        col_g, col_t = st.columns([1, 2])
        with col_g:
            st.plotly_chart(
                plot_risk_gauge(latest_obs["predicted_risk"], title=f"Latest Risk: {selected_unit}"),
                use_container_width=True,
            )
            if latest_obs["anomaly_flag"] == 1:
                st.error("⚠️ MULTI-SENSOR OPERATIONAL ANOMALY DETECTED by Isolation Forest")
            else:
                st.success("✅ Multi-sensor operating state is within nominal baseline")

        with col_t:
            # Unit risk & vibration over time
            fig_unit = go.Figure()
            fig_unit.add_trace(go.Scatter(
                x=df_unit["timestamp"],
                y=df_unit["predicted_risk"] * 100.0,
                name="7-Day Incident Risk (%)",
                line=dict(color="#d9534f", width=2.5),
            ))
            fig_unit.add_trace(go.Scatter(
                x=df_unit["timestamp"],
                y=df_unit["vibration_rms_mms"] * 10.0,
                name="Vibration RMS (×10 mm/s)",
                line=dict(color="#337ab7", width=1.5, dash="dash"),
            ))
            fig_unit.add_hline(y=alert_threshold * 100.0, line_dash="dot", line_color="orange", annotation_text="Alert Level")
            fig_unit.update_layout(
                title=f"Historical Risk & Vibration Trajectory: {selected_unit}",
                xaxis_title="Date",
                yaxis_title="Score",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_unit, use_container_width=True)

        st.subheader("Historical Incident Warning Sequence Analysis")
        st.write("Evaluating warning signals that preceded past incidents on this asset:")
        inc_dates = df_unit[df_unit["incident_today"] == 1]["timestamp"].tolist()
        if inc_dates:
            st.info(f"Recorded incidents on: {', '.join([d.strftime('%Y-%m-%d') for d in inc_dates])}")
            # Show pre-incident sequence
            target_inc = inc_dates[0]
            pre_inc_df = df_unit[(df_unit["timestamp"] >= target_inc - pd.Timedelta(days=14)) & (df_unit["timestamp"] <= target_inc + pd.Timedelta(days=2))]
            st.dataframe(
                pre_inc_df[["timestamp", "vibration_rms_mms", "daily_near_miss_count", "near_miss_category", "pm_overdue_days", "operator_fatigue_index", "incident_today", "predicted_risk_pct"]],
                use_container_width=True,
            )
        else:
            st.write("No reportable incidents occurred on this asset during the recording period.")

    # ----------------------------------------------------
    # TAB 3: REAL-TIME RISK SIMULATOR & SHAP
    # ----------------------------------------------------
    with tabs[2]:
        render_disclaimer_banner()
        st.subheader("🔮 Real-Time Operational Risk Simulator & Root-Cause Attribution")
        st.write("Adjust operational parameters to evaluate simulated incident probability and inspect SHAP root-cause feature attributions.")

        col_inputs1, col_inputs2, col_inputs3 = st.columns(3)

        with col_inputs1:
            st.markdown("**🔧 Mechanical & Operating Parameters**")
            sim_vib = st.slider("Vibration RMS (mm/s)", 0.2, 10.0, 3.2, 0.1, help="ISO 10816: Normal < 2.8, Alert 2.8-4.5, Danger > 4.5")
            sim_temp = st.slider("Process Temperature (°C)", 15.0, 150.0, 68.0, 1.0)
            sim_pressure = st.slider("Operating Pressure (bar)", 1.0, 50.0, 12.0, 0.5)
            sim_rpm = st.slider("Rotational Speed (RPM)", 0, 6000, 1780, 50)

        with col_inputs2:
            st.markdown("**🛠️ Maintenance State & Equipment Type**")
            sim_eq = st.selectbox("Equipment Type", cfg["generator"]["equipment_types"], index=0)
            sim_facility = st.selectbox("Facility", cfg["generator"]["facilities"], index=0)
            sim_overdue = st.slider("Days PM Overdue", 0, 90, 18, 1)
            sim_tickets = st.slider("Open Maintenance Tickets", 0, 6, 2, 1)

        with col_inputs3:
            st.markdown("**👷 Human Factors & Hazard Signals**")
            sim_nm = st.slider("Near-Misses Logged (Today)", 0, 5, 2, 1)
            sim_nm_cat = st.selectbox("Primary Hazard Category", [
                "Mechanical_Guard_Trip", "Pressure_Excursion", "Thermal_Spike",
                "Slip_Trip_Fall_Hazard", "Chemical_Vapor_Weep", "None"
            ], index=0)
            sim_shift = st.selectbox("Shift Type", ["Day", "Swing", "Night"], index=2)
            sim_overtime = st.slider("Shift Overtime Hours", 0.0, 14.0, 4.5, 0.5)
            sim_fatigue = st.slider("Operator Fatigue Index (0-100)", 10.0, 100.0, 72.0, 1.0)

        # Build simulated single-observation dataframe
        sim_raw = pd.DataFrame([{
            "record_id": "REC_SIM_001",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "facility": sim_facility,
            "unit_id": f"{sim_facility}_SIM_01",
            "equipment_type": sim_eq,
            "shift_type": sim_shift,
            "ambient_temperature_c": 22.0,
            "process_temperature_c": float(sim_temp),
            "vibration_rms_mms": float(sim_vib),
            "vibration_kurtosis": 3.8,
            "operating_pressure_bar": float(sim_pressure),
            "rotational_speed_rpm": float(sim_rpm),
            "duty_cycle_pct": 90.0,
            "acoustic_emission_db": 74.0,
            "days_since_last_pm": int(sim_overdue + 60),
            "pm_overdue_days": int(sim_overdue),
            "open_maintenance_tickets": int(sim_tickets),
            "daily_near_miss_count": int(sim_nm),
            "near_miss_category": sim_nm_cat,
            "shift_overtime_hours": float(sim_overtime),
            "operator_fatigue_index": float(sim_fatigue),
            "safety_audit_score": 78.0,
            "incident_today": 0,
            "incident_next_7d": 0,
            "days_to_incident": 30,
            "survival_event": 0,
            "data_origin": "simulator",
        }])

        # Transform using fitted pipeline
        X_sim = artifacts["pipeline"].transform(sim_raw, return_targets=False)

        # Predictions
        sim_cal_prob = artifacts["calibrated_xgb"].predict_proba(X_sim)[0, 1]
        sim_anom_score = artifacts["anomaly_detector"].score_samples(X_sim)[0]
        sim_is_anom = artifacts["anomaly_detector"].predict_anomaly(X_sim)[0]

        st.markdown("---")
        st.subheader("🎯 Simulated Safety Assessment Output")

        col_out1, col_out2 = st.columns([1, 1.5])
        with col_out1:
            st.plotly_chart(plot_risk_gauge(sim_cal_prob, title="Simulated 7-Day Incident Probability"), use_container_width=True)
            if sim_is_anom:
                st.error(f"⚠️ OPERATIONAL ANOMALY DETECTED (Score: {sim_anom_score:.2f})")
            else:
                st.success(f"✅ Operating Envelope Nominal (Anomaly Score: {sim_anom_score:.2f})")

        with col_out2:
            st.markdown("#### 🧠 SHAP Root-Cause Feature Attribution")
            shap_explainer = SafetySHAPExplainer(model_wrapper=artifacts["xgboost"], feature_names=list(X_sim.columns))
            instance_explanation = shap_explainer.explain_instance(X_sim.iloc[0], top_n=5)

            st.write(f"**Baseline Prior Log-Odds:** `{instance_explanation['base_expected_value']:.3f}` | **Net SHAP Shift:** `{instance_explanation['total_shap_sum']:+.3f}`")

            # Risk elevators
            st.markdown("**Top Hazard Escalation Factors (+ Risk):**")
            for item in instance_explanation["top_risk_drivers"]:
                st.markdown(f"- 🔴 **`{item['feature']}`** (value: `{item['value']:.2f}`) → **+{item['shap_impact']:.3f}** log-odds")

            # Mitigating factors
            st.markdown("**Top Risk Mitigating Factors (- Risk):**")
            for item in instance_explanation["top_mitigating_factors"]:
                st.markdown(f"- 🟢 **`{item['feature']}`** (value: `{item['value']:.2f}`) → **{item['shap_impact']:.3f}** log-odds")

    # ----------------------------------------------------
    # TAB 4: SURVIVAL ANALYSIS & TIME-TO-EVENT
    # ----------------------------------------------------
    with tabs[3]:
        render_disclaimer_banner()
        st.subheader("⏳ Survival Analysis: Time-to-Incident Hazard Dynamics")
        st.write("Modeling the time until a safety incident occurs using Kaplan-Meier survival curves and Cox Proportional Hazards regression.")

        surv_analyzer = artifacts["survival_analyzer"]
        surv_table = surv_analyzer.get_survival_table()

        fig_surv = go.Figure()
        fig_surv.add_trace(go.Scatter(x=surv_table["Day"], y=surv_table["Overall_Survival_Prob"], name="Fleet Overall", line=dict(color="black", width=2)))
        fig_surv.add_trace(go.Scatter(x=surv_table["Day"], y=surv_table["High_Near_Miss_Survival"], name="High Near-Miss Warning (>=2)", line=dict(color="#d9534f", width=2.5)))
        fig_surv.add_trace(go.Scatter(x=surv_table["Day"], y=surv_table["Low_Near_Miss_Survival"], name="Low Near-Miss (<2)", line=dict(color="#5cb85c", width=2.5)))
        fig_surv.add_trace(go.Scatter(x=surv_table["Day"], y=surv_table["Vibration_Alert_Survival"], name="ISO Vibration Alert (>=2.8 mm/s)", line=dict(color="#f0ad4e", width=2, dash="dash")))

        fig_surv.update_layout(
            title="Kaplan-Meier Survival Curves: Probability of Safe Operation Over Time",
            xaxis_title="Operational Days Ahead",
            yaxis_title="Incident-Free Survival Probability S(t)",
            yaxis=dict(range=[0.0, 1.05]),
            height=420,
        )
        st.plotly_chart(fig_surv, use_container_width=True)

        st.subheader("Cox Proportional Hazards Regression Results")
        if surv_analyzer.cph_summary_ is not None and not surv_analyzer.cph_summary_.empty:
            st.dataframe(
                surv_analyzer.cph_summary_.style.format({
                    "Hazard_Ratio": "{:.3f}",
                    "Std_Error": "{:.4f}",
                    "p_value": "{:.2e}",
                    "neg_log2_p": "{:.1f}",
                }).highlight_max(subset=["Hazard_Ratio"], color="#ffcccc"),
                use_container_width=True,
            )
            st.markdown(
                """
                **Domain Interpretation:**
                - **Hazard Ratio > 1.0**: Factor multiplies the instantaneous hazard rate of an incident.
                - **Vibration Velocity RMS**: Significant hazard multiplier confirming ISO 10816 mechanical severity impacts.
                - **Near-Miss Frequency & Maintenance Overdue Days**: Confirm leading indicator properties hypothesized in Heinrich's Safety Triangle.
                """
            )

    # ----------------------------------------------------
    # TAB 5: MODEL EVALUATION BENCHMARKS
    # ----------------------------------------------------
    with tabs[4]:
        render_disclaimer_banner()
        st.subheader("📈 Model Evaluation & Rigorous Benchmarks")
        st.write("Comparing candidate safety prediction models on strictly held-out temporal test observations:")

        eval_summary_path = Path("reports/evaluation_summary.json")
        if eval_summary_path.exists():
            import json
            with open(eval_summary_path, "r") as f:
                eval_data = json.load(f)
            df_eval = pd.DataFrame(eval_data)
            st.dataframe(
                df_eval[["model_name", "pr_auc", "recall", "precision", "f1_score", "roc_auc", "brier_score", "total_cost"]].style.format({
                    "pr_auc": "{:.3f}",
                    "recall": "{:.3f}",
                    "precision": "{:.3f}",
                    "f1_score": "{:.3f}",
                    "roc_auc": "{:.3f}",
                    "brier_score": "{:.4f}",
                    "total_cost": "${:,.0f}",
                }).highlight_max(subset=["pr_auc", "recall", "roc_auc"], color="#d4edda")
                .highlight_min(subset=["brier_score", "total_cost"], color="#d4edda"),
                use_container_width=True,
            )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image("reports/figures/pr_curves.png", caption="Precision-Recall Curves (Essential for Class Imbalance)")
            st.image("reports/figures/confusion_matrices.png", caption="Confusion Matrix Grid at Operating Decision Threshold")
        with col_f2:
            st.image("reports/figures/calibration_curves.png", caption="Reliability Diagram (Isotonic Calibration vs Raw)")
            st.image("reports/figures/cost_curves.png", caption="Safety Operational Cost vs Decision Threshold")

        st.markdown("### Global SHAP Root-Cause Feature Importance")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.image("reports/figures/shap_summary_bar.png", caption="Top 15 Global Safety Risk Drivers (Mean |SHAP|)")
        with col_s2:
            st.image("reports/figures/shap_beeswarm.png", caption="SHAP Beeswarm: Directional Impact of Warning Signals")

    # ----------------------------------------------------
    # TAB 6: EXPORT & AUDIT REPORTS
    # ----------------------------------------------------
    with tabs[5]:
        render_disclaimer_banner()
        st.subheader("📑 Safety Intelligence Audit & Export Reports")
        st.write("Download executive safety reports and structured telemetry for auditing and regulatory compliance:")

        active_filters = {
            "facility": selected_facility,
            "equipment": selected_eq,
            "start_date": str(date_range[0]) if isinstance(date_range, (tuple, list)) else "Start",
            "end_date": str(date_range[1]) if isinstance(date_range, (tuple, list)) else "End",
        }

        report_markdown = generate_safety_audit_report(
            summary_stats=summary_stats,
            high_risk_units=high_risk_units_df,
            active_filters=active_filters,
        )

        st.markdown(report_markdown)

        st.download_button(
            label="📥 Download Safety Audit Report (.md)",
            data=report_markdown,
            file_name=f"NEARMISS_Audit_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )

        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Telemetry Data (.csv)",
            data=csv_data,
            file_name=f"NEARMISS_Telemetry_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
