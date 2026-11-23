"""Reusable UI components, Plotly visualizations, and layout blocks for NEARMISS Streamlit dashboard.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_disclaimer_banner():
    """Render mandatory safety decision-support disclaimer banner."""
    st.markdown(
        """
        <div style="background-color: #fff3cd; border-left: 6px solid #ffeeba; border-color: #ffe87c; padding: 12px 18px; margin-bottom: 20px; border-radius: 4px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 22px; margin-right: 12px;">⚠️</span>
                <span style="color: #856404; font-size: 14px; line-height: 1.4;">
                    <strong>DECISION SUPPORT NOTICE:</strong> The predicted safety risk score is an operational leading-indicator estimate intended to prioritize preventive maintenance, safety walk-throughs, and hazard mitigations. <em>It does not guarantee that an incident will occur, nor does a low score guarantee complete immunity.</em> Always adhere to OSHA and facility safety protocols.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_metrics(stats: Dict[str, float]):
    """Render top-level executive KPI cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Shifts Analyzed",
            value=f"{int(stats.get('total_observations', 0)):,}",
            help="Total shift-level telemetry observations matching filters.",
        )
    with col2:
        st.metric(
            label="Total Near-Misses",
            value=f"{int(stats.get('total_near_misses', 0)):,}",
            delta=f"{stats.get('near_miss_trend', 0):+.1f}% vs prev",
            delta_color="inverse",
            help="Reported pre-incident warning signals and unsafe conditions.",
        )
    with col3:
        st.metric(
            label="Historical Incidents",
            value=f"{int(stats.get('total_incidents', 0)):,}",
            help="Actual historical incidents recorded across selected units.",
        )
    with col4:
        st.metric(
            label="Heinrich Safety Ratio",
            value=f"{stats.get('heinrich_ratio', 0.0):.1f} : 1",
            help="Observed ratio of near-misses to incidents (Heinrich's triangle benchmark is 30:1 to 300:1).",
        )
    with col5:
        high_risk_count = int(stats.get("high_risk_count", 0))
        st.metric(
            label="High-Risk Units (>=35%)",
            value=f"{high_risk_count}",
            delta="Action Req" if high_risk_count > 0 else "Normal",
            delta_color="inverse",
            help="Number of active equipment assets exceeding the critical risk threshold.",
        )


def plot_heinrich_pyramid(near_misses: int, minor_hazards: int, major_incidents: int) -> go.Figure:
    """Render an interactive Heinrich Safety Pyramid diagram using Plotly."""
    fig = go.Figure(go.Funnel(
        y=["Major Incidents / Lost Time", "Minor Injuries / Damage", "Near-Misses & Unsafe Conditions"],
        x=[max(1, major_incidents), max(2, minor_hazards), max(10, near_misses)],
        textinfo="value+percent initial",
        marker={
            "color": ["#d9534f", "#f0ad4e", "#5bc0de"],
            "line": {"width": [2, 2, 2], "color": ["#721c24", "#856404", "#17a2b8"]},
        },
        connector={"line": {"color": "#6c757d", "dash": "dot"}},
    ))

    fig.update_layout(
        title="Heinrich's Safety Pyramid (Empirical Warning Base)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


def plot_risk_gauge(probability: float, title: str = "7-Day Incident Probability") -> go.Figure:
    """Render an industrial risk gauge meter."""
    pct = probability * 100.0

    if pct < 20.0:
        bar_color = "#28a745"
        status_text = "LOW RISK (NORMAL)"
    elif pct < 35.0:
        bar_color = "#ffc107"
        status_text = "GUARDED RISK"
    elif pct < 60.0:
        bar_color = "#fd7e14"
        status_text = "ELEVATED RISK"
    else:
        bar_color = "#dc3545"
        status_text = "CRITICAL RISK (IMMEDIATE ACTION)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 36, "color": bar_color}},
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{bar_color}'><b>{status_text}</b></span>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 20], "color": "rgba(40, 167, 69, 0.2)"},
                {"range": [20, 35], "color": "rgba(255, 193, 7, 0.2)"},
                {"range": [35, 60], "color": "rgba(253, 126, 20, 0.2)"},
                {"range": [60, 100], "color": "rgba(220, 53, 69, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 35.0,
            },
        },
    ))

    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_risk_trend_over_time(df: pd.DataFrame) -> go.Figure:
    """Plot daily average predicted risk and near-miss frequency over time."""
    df_agg = df.groupby("timestamp").agg(
        avg_risk=("predicted_risk", "mean"),
        total_near_misses=("daily_near_miss_count", "sum"),
        total_incidents=("incident_today", "sum") if "incident_today" in df.columns else ("predicted_risk", "count"),
    ).reset_index()

    fig = go.Figure()

    # Risk trend line
    fig.add_trace(go.Scatter(
        x=df_agg["timestamp"],
        y=df_agg["avg_risk"] * 100.0,
        name="Mean Predicted Risk (%)",
        line=dict(color="#d9534f", width=2.5),
        yaxis="y1",
    ))

    # Alert threshold line
    fig.add_hline(
        y=35.0,
        line_dash="dot",
        line_color="orange",
        annotation_text="Critical Alert Threshold (35%)",
        annotation_position="bottom right",
    )

    # Near-miss volume bars on secondary y-axis
    fig.add_trace(go.Bar(
        x=df_agg["timestamp"],
        y=df_agg["total_near_misses"],
        name="Daily Near-Misses",
        marker_color="rgba(23, 162, 184, 0.4)",
        yaxis="y2",
    ))

    fig.update_layout(
        title="Fleet Safety Risk & Near-Miss Accumulation Over Time",
        xaxis=dict(title="Operational Date"),
        yaxis=dict(title="Predicted Incident Risk (%)", range=[0, 100]),
        yaxis2=dict(
            title="Near-Miss Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def plot_operational_stress_scatter(df: pd.DataFrame) -> go.Figure:
    """Plot multi-dimensional operational stress: Vibration RMS vs Process Temp vs Risk."""
    sample_df = df.sample(min(1200, len(df)), random_state=42) if len(df) > 1200 else df

    fig = px.scatter(
        sample_df,
        x="vibration_rms_mms",
        y="process_temperature_c",
        color="predicted_risk",
        size="pm_overdue_days",
        hover_data=["unit_id", "facility", "equipment_type", "daily_near_miss_count"],
        color_continuous_scale="Reds",
        labels={
            "vibration_rms_mms": "Vibration Velocity RMS (mm/s)",
            "process_temperature_c": "Process Temperature (°C)",
            "predicted_risk": "Risk Score",
            "pm_overdue_days": "Days PM Overdue",
        },
        title="Multi-Dimensional Operating Envelope: Vibration vs Temperature vs Risk",
    )

    # Add ISO vibration guideline lines
    fig.add_vline(x=2.8, line_dash="dash", line_color="orange", annotation_text="ISO Alert (2.8 mm/s)")
    fig.add_vline(x=4.5, line_dash="dash", line_color="red", annotation_text="ISO Danger (4.5 mm/s)")

    fig.update_layout(height=420, margin=dict(l=30, r=30, t=50, b=30))
    return fig
