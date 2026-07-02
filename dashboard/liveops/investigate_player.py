"""Workflow 2 — Investigate Player."""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from dashboard.liveops.data import load_all_data, db_missing, show_missing_db_error
from dashboard.theme import get_theme_colors

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def render_investigate_player():
    st.title("🔍 Investigate Player")
    st.markdown(
        "*Search any player to view their complete experience profile, "
        "risk assessment, and recommended interventions*"
    )
    st.divider()

    if db_missing():
        show_missing_db_error()
        return

    data = load_all_data()
    tier_colors = get_theme_colors()['tier_colors']

    pxi_df = data['pxi_scores']
    quit_df = data['quit_predictions']
    recs_df = data['recommendations']

    col1, col2 = st.columns([3, 1])
    with col1:
        player_input = st.text_input(
            "Enter Player ID",
            value=st.session_state.get('player_id', ''),
            placeholder="e.g. P1234_H or P5678_A",
            help="Player IDs follow format P{match_id}_{H/A}"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random At-Risk Player"):
            at_risk = pxi_df[pxi_df['pxi_tier'].isin(['At Risk', 'Critical'])]
            if len(at_risk) > 0:
                random_player = at_risk.sample(1)['player_id'].iloc[0]
                st.session_state['player_id'] = random_player
                st.rerun()

    if not player_input:
        return

    player_pxi = pxi_df[pxi_df['player_id'] == player_input]

    if len(player_pxi) == 0:
        st.error(f"Player **'{player_input}'** not found. Check the player ID format.")
        st.info("💡 Sample IDs: P1_H, P100_A, P500_H, P1000_A")
        return

    player_pxi = player_pxi.iloc[0]

    pxi_score = player_pxi.get('pxi_score', 0)
    pxi_tier = player_pxi.get('pxi_tier', 'Unknown')
    tier_color = tier_colors.get(pxi_tier, '#888888')

    st.markdown(
        f"""<div class="player-header" style="border-color: {tier_color};">
        <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
            <div>
                <div style="color:{tier_color}; font-size:1.5rem; font-weight:700; margin-bottom:4px;">
                    {player_input}
                </div>
                <span style="background:{tier_color}22; color:{tier_color}; border:1px solid {tier_color};
                border-radius:20px; padding:3px 12px; font-size:0.85rem; font-weight:600;">
                {pxi_tier}
                </span>
                <span style="color:#AAAAAA; margin-left:14px; font-size:0.9rem;">
                PXI Score: <strong style="color:#FFF;">{pxi_score:.1f}</strong> / 100
                </span>
            </div>
        </div>
        </div>""",
        unsafe_allow_html=True
    )

    st.subheader("Player Experience Profile")

    col1, col2, col3, col4 = st.columns(4)

    components = {
        'Match Quality': player_pxi.get('pxi_avg_mqi', 0),
        'Session Consistency': player_pxi.get('pxi_session_consistency', 0),
        'Engagement Trend': player_pxi.get('pxi_engagement_trend', 0),
        'Streak Factor': player_pxi.get('pxi_streak_factor', 0)
    }

    for col, (label, value) in zip([col1, col2, col3, col4], components.items()):
        color = '#00CC55' if value >= 65 else '#FFBB00' if value >= 40 else '#FF2222'
        pct = min(100, max(0, value))
        with col:
            st.markdown(
                f"""<div class="stat-card" style="border:1px solid {color}22;">
                <div style="color:{color}; font-size:1.8rem; font-weight:700;">{value:.0f}</div>
                <div style="color:#AAAAAA; font-size:0.75rem; text-transform:uppercase;
                letter-spacing:0.5px; margin-top:6px; margin-bottom:8px;">{label}</div>
                <div class="progress-container">
                    <div style="background:{color}; width:{pct}%; height:100%; border-radius:4px;
                    transition:width 0.5s ease;"></div>
                </div>
                </div>""",
                unsafe_allow_html=True
            )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pxi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "PXI Health Score", 'font': {'color': '#CCCCCC', 'size': 13}},
        number={'font': {'color': '#FFFFFF', 'size': 36}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': '#AAAAAA',
                'tickfont': {'color': '#AAAAAA', 'size': 10}
            },
            'bar': {'color': tier_color, 'thickness': 0.25},
            'bgcolor': '#1a1a1a',
            'bordercolor': '#333333',
            'borderwidth': 1,
            'steps': [
                {'range': [0, 35], 'color': '#2a0a0a'},
                {'range': [35, 55], 'color': '#2a1a0a'},
                {'range': [55, 75], 'color': '#0a1a2a'},
                {'range': [75, 100], 'color': '#0a2a0a'},
            ],
            'threshold': {
                'line': {'color': tier_color, 'width': 4},
                'thickness': 0.8,
                'value': pxi_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='#111111',
        font=dict(color='#FFFFFF', family='Inter, sans-serif'),
        height=280,
        margin=dict(l=30, r=30, t=50, b=20)
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Player Stats**")
        stats = {
            "Losing Streak": player_pxi.get('losing_streak', 0),
            "Winning Streak": player_pxi.get('winning_streak', 0),
            "Matches This Week": player_pxi.get('matches_this_week', 0),
            "Days Since Last Match": player_pxi.get('recency_days', 0),
            "Historical Ragequit Rate": f"{player_pxi.get('ragequit_rate_historical', 0):.0%}",
            "Win Rate (Last 10)": f"{player_pxi.get('win_rate_last10', 0):.0%}",
            "MQI Score": f"{player_pxi.get('mqi_score', 0):.1f}",
            "MQI Tier": player_pxi.get('mqi_tier', 'N/A')
        }

        for k, v in stats.items():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(
                    f"<span style='color:#888888; font-size:0.85rem;'>{k}</span>",
                    unsafe_allow_html=True
                )
            with col_b:
                val_color = '#FFFFFF'
                if k == "Losing Streak" and isinstance(v, (int, float)) and v >= 3:
                    val_color = '#FF4444'
                elif k == "Winning Streak" and isinstance(v, (int, float)) and v >= 3:
                    val_color = '#00CC55'
                st.markdown(
                    f"<span style='color:{val_color}; font-size:0.85rem; font-weight:600;'>{v}</span>",
                    unsafe_allow_html=True
                )

    st.divider()

    st.subheader("⚠️ Quit Risk Assessment")

    player_quit = quit_df[quit_df['player_id'] == player_input]

    if len(player_quit) > 0:
        player_quit = player_quit.iloc[0]
        quit_prob = float(player_quit.get('quit_probability', 0.0))

        quit_color = '#FF2222' if quit_prob >= 0.60 else '#FF8800' if quit_prob >= 0.40 else '#00CC55'
        risk_label = (
            '🔴 HIGH RISK' if quit_prob >= 0.60
            else '🟠 MODERATE RISK' if quit_prob >= 0.40
            else '🟢 LOW RISK'
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"""<div class="stat-card" style="border:2px solid {quit_color};
                box-shadow: 0 4px 20px {quit_color}33;">
                <div style="color:{quit_color}; font-size:2.5rem; font-weight:700;">{quit_prob:.0%}</div>
                <div style="color:#AAAAAA; font-size:0.8rem; margin:8px 0 4px;">Quit Probability</div>
                <div style="color:{quit_color}; font-size:0.85rem; font-weight:600;">{risk_label}</div>
                <div class="progress-container" style="margin-top:14px;">
                    <div style="background:{quit_color}; width:{quit_prob*100:.0f}%; height:100%;
                    border-radius:4px;"></div>
                </div>
                </div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown("**Top Quit Drivers (SHAP)**")
            for i in range(1, 4):
                feat = player_quit.get(f'driver_{i}_feature', '')
                shap_val = float(player_quit.get(f'driver_{i}_shap', 0.0))
                direction = player_quit.get(f'driver_{i}_direction', '')
                if feat:
                    bar_pct = min(100, abs(shap_val) * 200)
                    bar_color = '#FF4B00' if 'increases' in str(direction) else '#00CC55'
                    direction_icon = '📈' if 'increases' in str(direction) else '📉'
                    feat_display = str(feat).replace('_', ' ').title()
                    st.markdown(
                        f"""**{i}. {feat_display}**
<div class="progress-container">
<div style="background:{bar_color}; width:{bar_pct}%; height:100%; border-radius:4px;"></div>
</div>
<span style="color:#AAAAAA; font-size:0.8rem;">{direction_icon} SHAP: {shap_val:+.3f} — {direction}</span>""",
                        unsafe_allow_html=True
                    )
                    st.markdown("")
    else:
        st.info("No quit risk prediction available for this player.")

    st.divider()

    st.subheader("💡 Recommended Interventions")

    player_recs = recs_df[recs_df['player_id'] == player_input].sort_values(
        'confidence', ascending=False
    )

    if len(player_recs) == 0:
        st.info("No interventions met the confidence threshold for this player.")
    else:
        for _, rec in player_recs.iterrows():
            priority = rec.get('priority', '')
            conf = rec.get('confidence', 0)
            name = rec.get('intervention_name', '')
            desc = rec.get('description', '')
            reason = rec.get('reason', '')
            lift = rec.get('expected_d7_lift', '')

            with st.expander(f"{priority} — {name}  (Confidence: {float(conf):.0%})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"📋 **Description:** {desc}")
                    st.markdown(f"🎯 **Why:** {reason}")
                with c2:
                    st.markdown(f"📈 **Expected D7 Lift:** {lift}")
                    st.markdown(f"🔬 **Experiment Metric:** {rec.get('experiment_metric', '')}")
                    st.markdown(f"🛡️ **Guardrail:** {rec.get('guardrail_metric', '')}")

    if len(player_recs) > 0:
        st.divider()
        if st.button("📋 Generate Experiment Brief for Top Recommendation"):
            try:
                from intervention_engine import InterventionEngine, ALL_INTERVENTIONS
                engine = InterventionEngine()
                top_rec_id = player_recs.iloc[0]['intervention_id']
                top_intervention = next(
                    (i for i in ALL_INTERVENTIONS if i.id == top_rec_id),
                    ALL_INTERVENTIONS[0]
                )
                at_risk_count = len(recs_df[recs_df['intervention_id'] == top_rec_id])
                brief = engine.generate_experiment_brief(
                    top_intervention,
                    target_cohort_size=at_risk_count,
                    current_d7_rate=0.31
                )
                st.code(brief, language=None)
            except Exception as e:
                st.error(f"Could not generate brief: {e}")
