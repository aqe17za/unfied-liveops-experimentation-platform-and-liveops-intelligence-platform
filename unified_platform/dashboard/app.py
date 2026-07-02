"""Unified EA SPORTS FC dashboard — LiveOps Intelligence + Experimentation Engine.

Single Streamlit entry point (port 8505) with a top-level sidebar toggle
routing between the two platforms. Each page module loads its own data lazily
(only when selected) and shows a clear error state if its upstream pipeline
hasn't been run yet.
"""
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.theme import inject_global_css, get_theme_colors, load_config  # noqa: E402

st.set_page_config(
    page_title="EA SPORTS FC | Unified Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_global_css()

LIVEOPS_PAGES = [
    "🏠  Game Health Check",
    "🔍  Investigate Player",
    "📊  Match Quality Analysis",
    "⚡  LiveOps Action Queue",
    "📋  Weekly Intelligence Report",
]

EXPERIMENTATION_PAGES = ["Registry", "Experiment", "Segments", "Decision", "Audit"]


def render_liveops_page(page: str):
    from dashboard.liveops.game_health import render_game_health
    from dashboard.liveops.investigate_player import render_investigate_player
    from dashboard.liveops.match_quality import render_match_quality
    from dashboard.liveops.action_queue import render_action_queue
    from dashboard.liveops.weekly_report import render_weekly_report

    if "Game Health Check" in page:
        render_game_health()
    elif "Investigate Player" in page:
        render_investigate_player()
    elif "Match Quality Analysis" in page:
        render_match_quality()
    elif "Action Queue" in page:
        render_action_queue()
    elif "Intelligence Report" in page:
        render_weekly_report()


def render_experimentation_page(page: str, selected_experiment):
    from dashboard.experimentation.registry import render_registry
    from dashboard.experimentation.experiment import render_experiment
    from dashboard.experimentation.segments import render_segments
    from dashboard.experimentation.decision import render_decision
    from dashboard.experimentation.audit import render_audit

    if page == "Registry":
        render_registry()
    elif page == "Experiment":
        render_experiment(selected_experiment)
    elif page == "Segments":
        render_segments(selected_experiment)
    elif page == "Decision":
        render_decision(selected_experiment)
    elif page == "Audit":
        render_audit(selected_experiment)


def main():
    config = load_config()
    theme = get_theme_colors()

    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
            <div style="font-size:1.6rem; margin-bottom:4px;">⚽</div>
            <div style="color:{theme['primary']}; font-size:1.1rem; font-weight:700; letter-spacing:0.5px;">
            EA SPORTS FC</div>
            <div style="color:#666; font-size:0.75rem; margin-top:2px; letter-spacing:0.5px;">
            UNIFIED PLATFORM</div>
        </div>
        """, unsafe_allow_html=True)

        platform = st.radio(
            "PLATFORM",
            options=["LiveOps Intelligence", "Experimentation Engine"],
            label_visibility="collapsed",
        )

        st.divider()

        if platform == "LiveOps Intelligence":
            page = st.radio("NAVIGATION", options=LIVEOPS_PAGES, label_visibility="collapsed")
            selected_experiment = None
        else:
            from dashboard.experimentation.data_loader import (
                data_missing,
                get_all_experiment_ids,
                load_all,
            )

            selected_experiment = None
            if not data_missing():
                data = load_all()
                exec_df, _ = data["executive_summary"]
                experiment_ids = get_all_experiment_ids(exec_df)

                if "selected_experiment" not in st.session_state:
                    st.session_state["selected_experiment"] = experiment_ids[0] if experiment_ids else None
                if st.session_state["selected_experiment"] not in experiment_ids and experiment_ids:
                    st.session_state["selected_experiment"] = experiment_ids[0]

                if experiment_ids:
                    st.selectbox("Select Experiment", experiment_ids, key="selected_experiment")
                    st.divider()
                else:
                    st.warning("No experiments available.")

                selected_experiment = st.session_state.get("selected_experiment")

            page = st.radio("NAVIGATION", options=EXPERIMENTATION_PAGES, label_visibility="collapsed")

        st.divider()
        st.caption(f"📅 {datetime.now().strftime('%d %b %Y %H:%M')}")
        st.caption(f"v{config['project']['version']} | {config['project']['author']}")

    if platform == "LiveOps Intelligence":
        render_liveops_page(page)
    else:
        render_experimentation_page(page, selected_experiment)


if __name__ == "__main__":
    main()
