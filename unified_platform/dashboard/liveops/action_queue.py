"""Workflow 4 — LiveOps Action Queue."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.liveops.data import load_all_data, db_missing, show_missing_db_error
from dashboard.theme import make_plotly_dark

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def render_action_queue():
    st.title("⚡ LiveOps Action Queue")
    st.markdown("*Prioritised interventions for this week — ranked by confidence and player impact*")
    st.divider()

    if db_missing():
        show_missing_db_error()
        return

    data = load_all_data()

    queue_df = data['action_queue']
    recs_df = data['recommendations']
    pxi_df = data['pxi_scores']

    if len(pxi_df) == 0:
        st.warning("No data available.")
        return

    total_at_risk = len(pxi_df[pxi_df['pxi_tier'].isin(['At Risk', 'Critical'])])
    n_ready = len(queue_df[queue_df['status'] == 'Ready']) if 'status' in queue_df.columns else 0

    if 'priority_icon' in queue_df.columns:
        n_high = queue_df['priority_icon'].str.contains('HIGH', na=False).sum()
    else:
        n_high = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Players Needing Action", f"{total_at_risk:,}")
    with col2:
        st.metric("Actions Ready to Deploy", f"{n_ready}")
    with col3:
        st.metric("High Priority Actions", f"{n_high}")
    with col4:
        coverage = len(recs_df['player_id'].unique()) / len(pxi_df) * 100 if len(recs_df) > 0 else 0
        st.metric("Recommendation Coverage", f"{coverage:.1f}%")

    st.divider()

    st.subheader("📋 This Week's Action Queue")

    if len(queue_df) == 0:
        st.warning("No action queue data available.")
    else:
        available_cols = queue_df.columns.tolist()
        display_cols = [
            c for c in [
                'priority_icon', 'intervention_name',
                'affected_players', 'pct_of_at_risk',
                'mean_confidence', 'expected_d7_lift', 'status'
            ] if c in available_cols
        ]

        if not display_cols:
            st.dataframe(queue_df, use_container_width=True)
        else:
            display_df = queue_df[display_cols].copy()

            rename_map = {
                'priority_icon': 'Priority',
                'intervention_name': 'Action',
                'affected_players': 'Players Affected',
                'pct_of_at_risk': '% of At-Risk',
                'mean_confidence': 'Confidence',
                'expected_d7_lift': 'Expected D7 Lift',
                'status': 'Status'
            }
            display_df = display_df.rename(
                columns={k: v for k, v in rename_map.items() if k in display_df.columns}
            )

            if 'Confidence' in display_df.columns:
                display_df['Confidence'] = display_df['Confidence'].apply(
                    lambda x: f"{x:.0%}" if isinstance(x, float) else x
                )

            if '% of At-Risk' in display_df.columns:
                display_df['% of At-Risk'] = display_df['% of At-Risk'].apply(
                    lambda x: f"{float(x):.1%}" if pd.notna(x) else x
                )

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📋 Experiment Briefs")
    st.caption("Auto-generated experiment designs for the top 2 actions")

    try:
        from intervention_engine import InterventionEngine, ALL_INTERVENTIONS
        engine = InterventionEngine()

        top_2 = queue_df.head(2)

        for _, action in top_2.iterrows():
            int_id = action.get('intervention_id', '')
            intervention = next(
                (i for i in ALL_INTERVENTIONS if i.id == int_id),
                ALL_INTERVENTIONS[0]
            )

            cohort = int(action.get('affected_players', 500))
            brief = engine.generate_experiment_brief(
                intervention,
                target_cohort_size=cohort,
                current_d7_rate=0.31
            )

            name = action.get('intervention_name', int_id)
            title = f"Experiment Brief — {name}"

            with st.container(border=True):
                st.text(title)
                st.code(brief, language=None)

    except Exception as e:
        st.warning(f"Experiment brief generation unavailable: {e}")

    st.divider()
    st.subheader("Intervention Confidence Scores")

    if len(queue_df) > 0:
        conf_col = 'mean_confidence'
        name_col = 'intervention_name'

        if conf_col in queue_df.columns and name_col in queue_df.columns:
            sorted_queue = queue_df.sort_values(conf_col, ascending=True).copy()
            sorted_queue['conf_label'] = sorted_queue[conf_col].apply(lambda x: f"{x:.0%}")

            bar_colors = []
            for pi in sorted_queue.get('priority_icon', pd.Series([''] * len(sorted_queue))):
                pi = str(pi)
                if 'HIGH' in pi:
                    bar_colors.append('#FF2222')
                elif 'MEDIUM' in pi:
                    bar_colors.append('#FF8800')
                else:
                    bar_colors.append('#FFBB00')

            fig = go.Figure(go.Bar(
                x=sorted_queue[conf_col],
                y=sorted_queue[name_col],
                orientation='h',
                marker=dict(
                    color=bar_colors,
                    opacity=0.9,
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                ),
                text=sorted_queue['conf_label'],
                textposition='outside',
                textfont=dict(color='#CCCCCC', size=11),
                hovertemplate='%{y}: %{x:.0%} confidence<extra></extra>'
            ))
            make_plotly_dark(fig, title="Intervention Confidence Scores — Readiness to Deploy", height=320)
            fig.update_layout(
                xaxis_title="Mean Confidence Score",
                xaxis_tickformat='.0%',
                yaxis_title="",
                showlegend=False,
                margin=dict(l=180, r=60, t=55, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
