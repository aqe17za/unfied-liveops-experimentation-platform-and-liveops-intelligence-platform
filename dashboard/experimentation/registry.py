"""Experimentation page — Registry."""
import streamlit as st

from dashboard.experimentation.common import _page_title, _warn_if_error
from dashboard.experimentation.data_loader import (
    badge_for_decision,
    data_missing,
    get_all_decision_types,
    load_all,
    show_missing_data_error,
)


def render_registry():
    if data_missing():
        show_missing_data_error()
        return

    data = load_all()
    exec_df, exec_err = data["executive_summary"]
    dec_df, dec_err = data["decisions"]
    _warn_if_error("executive_summary.csv", exec_err)
    _warn_if_error("decisions.csv", dec_err)

    _page_title("Experiment Registry", "Portfolio-level view of decision outcomes and recommended next actions.")

    if exec_df.empty:
        st.error("No experiment data available. executive_summary.csv could not be loaded.")
        return

    decision_types = get_all_decision_types(dec_df) if not dec_df.empty else sorted(exec_df["decision"].dropna().unique())

    cols = st.columns(min(5, len(decision_types) + 1))
    cols[0].metric("Total experiments", len(exec_df))
    for index, dtype in enumerate(decision_types, start=1):
        cols[index % len(cols)].metric(dtype, int((exec_df["decision"] == dtype).sum()))

    st.markdown("<br/>", unsafe_allow_html=True)

    display_df = exec_df.copy()
    display_df.insert(1, "Signal", display_df["decision"].apply(badge_for_decision))
    display_df = display_df.rename(
        columns={
            "experiment_id": "Experiment",
            "decision": "Decision",
            "confidence": "Confidence",
            "primary_reason": "Primary Reason",
            "next_action": "Next Action",
        }
    )
    st.dataframe(display_df, hide_index=True, use_container_width=True)
