"""Experimentation page — Segments (heterogeneous treatment effects)."""
import matplotlib.pyplot as plt
import streamlit as st

from dashboard.experimentation.common import EMPTY_STATE, _colored_kv_grid, _page_title, _setup_chart, _warn_if_error
from dashboard.experimentation.data_loader import (
    data_missing,
    load_all,
    safe_get_row,
    safe_get_rows,
    show_missing_data_error,
    symbol_for_status,
)
from dashboard.theme import get_theme_colors


def render_segments(selected_experiment):
    if data_missing():
        show_missing_data_error()
        return

    if not selected_experiment:
        st.info(EMPTY_STATE)
        return

    data = load_all()
    theme = get_theme_colors()
    hte_summary_df, hte_summary_err = data["hte_summary"]
    hte_results_df, hte_results_err = data["heterogeneous_effects_results"]
    _warn_if_error("hte_summary.csv", hte_summary_err)
    _warn_if_error("heterogeneous_effects_results.csv", hte_results_err)

    _page_title(f"Segments - {selected_experiment}", "Heterogeneous treatment effects by player segment.")

    summary_row = safe_get_row(hte_summary_df, "experiment_id", selected_experiment)
    if summary_row is None:
        st.info("No segment analysis is available for this experiment.")
        return

    rollout_eligible = summary_row.get("rollout_eligible")
    audit_df, _ = data.get("decision_audit", (None, None))
    if audit_df is not None:
        audit_row = safe_get_row(audit_df, "experiment_id", selected_experiment)
        if audit_row is not None:
            if audit_row.get("decision") == "SHIP" and audit_row.get("guardrail_status") == "Clean":
                rollout_eligible = True

    _colored_kv_grid(
        [
            ("Best segment", summary_row.get("best_segment", "N/A"), "neutral"),
            ("Worst segment", summary_row.get("worst_segment", "N/A"), "neutral"),
            ("HTE type", summary_row.get("hte_type", "N/A"), "neutral"),
            ("Rollout eligible", symbol_for_status(rollout_eligible), "neutral"),
        ]
    )
    st.caption(f"Confidence: {summary_row.get('hte_confidence', 'N/A')}")

    segment_rows = safe_get_rows(hte_results_df, "experiment_id", selected_experiment)
    if segment_rows.empty:
        st.info("No per-segment detail available.")
        return

    dimensions = sorted(segment_rows["dimension"].dropna().unique().tolist())
    tabs = st.tabs(dimensions) if dimensions else []
    for tab, dimension in zip(tabs, dimensions):
        with tab:
            dim_rows = segment_rows[segment_rows["dimension"] == dimension].sort_values(
                "absolute_lift",
                ascending=True,
            )
            fig, ax = plt.subplots(figsize=(8, max(3, 0.48 * len(dim_rows))))
            colors = [theme['danger'] if value < 0 else theme['success'] for value in dim_rows["absolute_lift"]]
            ax.barh(dim_rows["segment_value"], dim_rows["absolute_lift"], color=colors, edgecolor="#ffffff", linewidth=0)
            ax.axvline(0, color="#64748b", linewidth=1.5, linestyle="--")
            ax.set_xlabel("Absolute lift")
            _setup_chart(ax, fig)

            st.markdown('<div style="background: #15171e; padding: 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-top: 16px;">', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig)

            display = dim_rows[
                [
                    "segment_value",
                    "n_control",
                    "n_treatment",
                    "absolute_lift",
                    "p_value",
                    "recommendation",
                    "reasoning",
                ]
            ].rename(
                columns={
                    "segment_value": "Segment",
                    "n_control": "n control",
                    "n_treatment": "n treatment",
                    "absolute_lift": "Lift",
                    "p_value": "p-value",
                    "recommendation": "Recommendation",
                    "reasoning": "Reasoning",
                }
            )
            st.markdown("<br/>", unsafe_allow_html=True)
            st.dataframe(display, hide_index=True, use_container_width=True)
