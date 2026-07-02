"""Workflow 1 — Game Health Check."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.liveops.data import load_all_data, db_missing, show_missing_db_error
from dashboard.liveops.helpers import simulate_weekly_trend
from dashboard.theme import get_theme_colors, make_plotly_dark, kpi_card


def render_game_health():
    st.title("🏠 Game Health Check")
    st.markdown("*Weekly player experience snapshot — Monday morning briefing*")
    st.divider()

    if db_missing():
        show_missing_db_error()
        return

    data = load_all_data()
    theme = get_theme_colors()
    tier_colors = theme['tier_colors']

    mqi_df = data['mqi_scores']
    pxi_df = data['pxi_scores']

    if len(mqi_df) == 0 or len(pxi_df) == 0:
        st.warning("No data available.")
        return

    avg_mqi = mqi_df['mqi_score'].mean()
    ragequit_rate = (
        (mqi_df['normalized_skill_gap'] > 0.6) & (mqi_df['score_diff_abs'] > 3)
    ).mean()
    n_critical = len(pxi_df[pxi_df['pxi_tier'] == 'Critical'])
    pct_at_risk = len(pxi_df[pxi_df['pxi_tier'].isin(['At Risk', 'Critical'])]) / len(pxi_df) * 100

    np.random.seed(1)
    prior_mqi = avg_mqi * np.random.uniform(0.95, 1.02)
    prior_ragequit = ragequit_rate * np.random.uniform(0.97, 1.04)
    prior_critical = n_critical * np.random.uniform(0.93, 1.05)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average MQI",
            f"{avg_mqi:.1f}",
            delta=f"{avg_mqi - prior_mqi:+.1f} vs last week",
            delta_color="normal"
        )
    with col2:
        st.metric(
            "Ragequit Rate",
            f"{ragequit_rate:.1%}",
            delta=f"{ragequit_rate - prior_ragequit:+.1%} vs last week",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "Critical Players",
            f"{n_critical:,}",
            delta=f"{n_critical - int(prior_critical):+d} vs last week",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            "At-Risk Population",
            f"{pct_at_risk:.1f}%",
            delta="of active player base"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if avg_mqi < 75 or ragequit_rate > 0.20 or pct_at_risk > 15:
        st.markdown("""
        <div class="alert-critical">
        ⚠️ <strong>LIVEOPS ALERT</strong> —
        One or more health metrics require attention.
        Review the Action Queue for recommended interventions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-success">
        ✅ <strong>All systems healthy</strong> —
        No critical alerts this week. Player experience metrics within normal range.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        weeks = [f"W{i}" for i in range(1, 9)]
        mqi_trend = simulate_weekly_trend(avg_mqi, n_weeks=8, volatility=0.03, seed=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks, y=mqi_trend,
            mode='lines',
            line=dict(color=theme['primary'], width=0),
            fill='tozeroy',
            fillcolor='rgba(255,75,0,0.08)',
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=weeks, y=mqi_trend,
            mode='lines+markers',
            line=dict(color=theme['primary'], width=2.5),
            marker=dict(size=7, color=theme['primary'], line=dict(color='white', width=1.5)),
            name='Avg MQI',
            hovertemplate='%{y:.1f}<extra>Week %{x}</extra>'
        ))
        fig.add_hline(
            y=65, line_dash="dash", line_color=theme['warning'],
            annotation_text="Good threshold (65)",
            annotation_font_color=theme['warning'],
            annotation_font_size=11
        )
        make_plotly_dark(fig, title="Match Quality Index — 8-Week Trend", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ragequit_trend = simulate_weekly_trend(ragequit_rate, n_weeks=8, volatility=0.05, seed=20)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks, y=[v * 100 for v in ragequit_trend],
            mode='lines',
            line=dict(color='#FF7700', width=0),
            fill='tozeroy',
            fillcolor='rgba(255,119,0,0.08)',
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=weeks, y=[v * 100 for v in ragequit_trend],
            mode='lines+markers',
            line=dict(color='#FF7700', width=2.5),
            marker=dict(size=7, color='#FF7700', line=dict(color='white', width=1.5)),
            name='Ragequit Rate %',
            hovertemplate='%{y:.1f}%<extra>Week %{x}</extra>'
        ))
        fig.add_hline(
            y=20, line_dash="dash", line_color=theme['danger'],
            annotation_text="Alert threshold (20%)",
            annotation_font_color=theme['danger'],
            annotation_font_size=11
        )
        make_plotly_dark(fig, title="Ragequit Rate — 8-Week Trend (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Match Quality Tier Distribution")

    tier_counts = mqi_df['mqi_tier'].value_counts()
    tier_order = ['Poor', 'Below Average', 'Average', 'Good', 'Elite']
    tier_counts = tier_counts.reindex([t for t in tier_order if t in tier_counts.index])

    fig = go.Figure(go.Bar(
        x=tier_counts.index,
        y=tier_counts.values,
        marker=dict(
            color=[tier_colors.get(t, '#888888') for t in tier_counts.index],
            opacity=0.9,
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        text=[f"{v:,}<br>({v / len(mqi_df) * 100:.1f}%)" for v in tier_counts.values],
        textposition='outside',
        textfont=dict(color='#CCCCCC', size=11),
        hovertemplate='%{x}: %{y:,} matches<extra></extra>'
    ))
    make_plotly_dark(fig, title="Matches by Quality Tier — This Week", height=350)
    fig.update_layout(
        xaxis_title="Match Quality Tier",
        yaxis_title="Number of Matches",
        bargap=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Player Health Summary")

    pxi_tier_counts = pxi_df['pxi_tier'].value_counts()
    pxi_order = ['Critical', 'At Risk', 'Stable', 'Healthy']

    cols = st.columns(4)
    for i, tier in enumerate(pxi_order):
        count = pxi_tier_counts.get(tier, 0)
        pct = count / len(pxi_df) * 100
        color = tier_colors.get(tier, '#888888')
        with cols[i]:
            st.markdown(kpi_card(f"{count:,}", tier, color=color, subtitle=f"{pct:.1f}% of base"), unsafe_allow_html=True)
