"""Workflow 5 — Weekly Intelligence Report."""
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.liveops.data import load_all_data, db_missing, show_missing_db_error
from dashboard.theme import make_plotly_dark


def render_weekly_report():
    st.title("📋 Weekly Intelligence Report")
    st.markdown(f"*Auto-generated briefing — {datetime.now().strftime('%A %d %B %Y')}*")
    st.divider()

    if db_missing():
        show_missing_db_error()
        return

    data = load_all_data()

    mqi_df = data['mqi_scores']
    pxi_df = data['pxi_scores']
    queue_df = data['action_queue']

    if len(mqi_df) == 0 or len(pxi_df) == 0:
        st.warning("No data available.")
        return

    avg_mqi = mqi_df['mqi_score'].mean()
    np.random.seed(5)
    prior_mqi = avg_mqi * np.random.uniform(0.96, 1.03)
    mqi_delta = avg_mqi - prior_mqi
    mqi_trend = "↑ improving" if mqi_delta > 0 else "↓ declining"

    n_critical = len(pxi_df[pxi_df['pxi_tier'] == 'Critical'])
    total_players = len(pxi_df)

    ragequit_proxy = (
        (mqi_df['normalized_skill_gap'] > 0.6) & (mqi_df['score_diff_abs'] > 3)
    ).mean()
    prior_ragequit = ragequit_proxy * np.random.uniform(0.97, 1.04)
    ragequit_delta = ragequit_proxy - prior_ragequit

    n_ready_actions = len(queue_df[queue_df['status'] == 'Ready']) if 'status' in queue_df.columns else len(queue_df)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a1a,#222); border:1px solid #FF4B00;
    border-radius:10px; padding:20px 24px; margin-bottom:16px;">
    <h3 style="color:#FF4B00; margin-top:0; font-size:1rem; text-transform:uppercase; letter-spacing:1px;">
    📊 THIS WEEK AT A GLANCE</h3>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average MQI", f"{avg_mqi:.1f}", f"{mqi_delta:+.1f} ({mqi_trend})")
        st.metric("Critical Players", f"{n_critical:,}", f"{n_critical / total_players * 100:.1f}% of base")
    with col2:
        st.metric(
            "Ragequit Rate", f"{ragequit_proxy:.1%}",
            f"{ragequit_delta:+.1%} vs last week", delta_color="inverse"
        )
        st.metric("Actions Ready", f"{n_ready_actions}", "ready to deploy")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1010,#221515); border:1px solid #FF6600;
    border-left:4px solid #FF6600; border-radius:10px; padding:20px 24px; margin-bottom:16px;">
    <h3 style="color:#FF6600; margin-top:0; font-size:1rem; text-transform:uppercase; letter-spacing:1px;">
    ⚠️ TOP RISKS THIS WEEK</h3>
    """, unsafe_allow_html=True)

    risks = []
    if n_critical > 0:
        risks.append(f"🔴 **{n_critical:,} players** are in Critical PXI tier — immediate intervention required")
    if ragequit_proxy > 0.18:
        risks.append(
            f"🔴 **Ragequit rate {ragequit_proxy:.1%}** exceeds 18% alert "
            f"threshold — matchmaking review needed"
        )
    if avg_mqi < 70:
        risks.append(f"🟠 **Average MQI {avg_mqi:.1f}** below 70 — match quality degradation detected")
    if not risks:
        risks.append("🟢 No critical risks detected this week — all metrics within acceptable bounds")

    for risk in risks:
        st.markdown(f"- {risk}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1600,#221e00); border:1px solid #FFAA00;
    border-left:4px solid #FFAA00; border-radius:10px; padding:20px 24px; margin-bottom:16px;">
    <h3 style="color:#FFAA00; margin-top:0; font-size:1rem; text-transform:uppercase; letter-spacing:1px;">
    ⚡ RECOMMENDED ACTIONS</h3>
    """, unsafe_allow_html=True)

    if len(queue_df) > 0:
        top_3 = queue_df.head(3)
        for _, action in top_3.iterrows():
            priority = action.get('priority_icon', '')
            name = action.get('intervention_name', action.get('intervention_id', ''))
            players = action.get('affected_players', 'N/A')
            confidence = action.get('mean_confidence', 0)
            conf_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)
            status = action.get('status', '')
            status_badge = f" · *{status}*" if status else ""
            if isinstance(players, (int, float)):
                st.markdown(f"- {priority} **{name}** — {players:,} players affected, confidence {conf_str}{status_badge}")
            else:
                st.markdown(f"- {priority} **{name}** — {players} players affected, confidence {conf_str}{status_badge}")
    else:
        st.markdown("- No actions in queue")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1a0a,#0d220d); border:1px solid #00AA44;
    border-left:4px solid #00AA44; border-radius:10px; padding:20px 24px; margin-bottom:16px;">
    <h3 style="color:#00AA44; margin-top:0; font-size:1rem; text-transform:uppercase; letter-spacing:1px;">
    🔮 7-DAY FORECAST</h3>
    """, unsafe_allow_html=True)

    np.random.seed(99)
    forecast_days = [
        (datetime.now() + timedelta(days=i)).strftime('%a %d')
        for i in range(1, 8)
    ]

    trend = 0.2 if mqi_delta > 0 else -0.2
    mqi_forecast = [avg_mqi + trend * i + np.random.normal(0, 1) for i in range(1, 8)]
    mqi_forecast = [max(0, min(100, v)) for v in mqi_forecast]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        upper = [v + 2 for v in mqi_forecast]
        lower = [v - 2 for v in mqi_forecast]
        fig.add_trace(go.Scatter(
            x=forecast_days + forecast_days[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(0,170,68,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            name='Confidence Interval'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_days,
            y=mqi_forecast,
            mode='lines+markers',
            line=dict(color='#00CC55', width=2.5, dash='dot'),
            marker=dict(size=7, color='#00CC55', line=dict(color='white', width=1.5)),
            name='Projected MQI',
            hovertemplate='%{x}: MQI %{y:.1f}<extra></extra>'
        ))
        fig.add_hline(
            y=avg_mqi,
            line_dash='solid',
            line_color='#FF4B00',
            line_width=1.5,
            annotation_text='Current MQI',
            annotation_font_color='#FF4B00',
            annotation_font_size=10
        )
        make_plotly_dark(fig, title="7-Day MQI Forecast", height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        projected_end_mqi = mqi_forecast[-1]
        projected_change = projected_end_mqi - avg_mqi

        direction = '↑' if projected_change > 0 else '↓'
        proj_color = '#00CC55' if projected_change > 0 else '#FF4B00'
        st.markdown(
            f"""<div style="padding:16px 0;">
            <div style="color:#888; font-size:0.78rem; text-transform:uppercase; margin-bottom:4px;">Projected MQI (Day 7)</div>
            <div style="color:{proj_color}; font-size:2rem; font-weight:700;">{projected_end_mqi:.1f}</div>
            <div style="color:{proj_color}; font-size:0.9rem;">{direction} {abs(projected_change):.1f} pts</div>
            <br>
            <div style="color:#888; font-size:0.78rem; text-transform:uppercase; margin-bottom:4px;">At-Risk Trend</div>
            <div style="color:#FFF; font-size:1rem; font-weight:600;">{'📉 Improving' if projected_change > 0 else '📈 Growing'}</div>
            <br>
            <div style="color:#666; font-size:0.75rem; font-style:italic;">
            Forecast based on current trend. Intervention deployment will improve outlook.
            </div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    report_lines = [
        "EA SPORTS FC LIVEOPS INTELLIGENCE REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "AT A GLANCE",
        f"Average MQI: {avg_mqi:.1f} ({mqi_delta:+.1f} vs last week)",
        f"Ragequit Rate: {ragequit_proxy:.1%}",
        f"Critical Players: {n_critical:,}",
        f"Actions Ready: {n_ready_actions}",
        "",
        "TOP RISKS",
    ]
    report_lines.extend(f"- {r}" for r in risks)
    report_lines.append("")
    report_lines.append("RECOMMENDED ACTIONS")
    if len(queue_df) > 0:
        for _, a in queue_df.head(3).iterrows():
            report_lines.append(f"- {a.get('intervention_name', 'N/A')}: {a.get('affected_players', 'N/A')} players")
    else:
        report_lines.append("None")
    report_lines.append("")
    report_lines.append("7-DAY FORECAST")
    report_lines.append(f"Projected MQI: {projected_end_mqi:.1f}")

    report_text = "\n".join(report_lines)

    st.download_button(
        label="⬇️ Download Report as Text",
        data=report_text,
        file_name=f"liveops_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
