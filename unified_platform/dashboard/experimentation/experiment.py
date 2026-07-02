"""Experimentation page — Experiment detail."""
from html import escape

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard.experimentation.common import (
    EMPTY_STATE,
    _colored_kv_grid,
    _decision_pill,
    _format_float,
    _format_percent,
    _page_title,
    _panel,
    _setup_chart,
    _warn_if_error,
)
from dashboard.experimentation.data_loader import (
    data_missing,
    format_probability,
    format_pvalue,
    load_all,
    safe_get_row,
    show_missing_data_error,
    symbol_for_status,
)


def render_experiment(selected_experiment):
    if data_missing():
        show_missing_data_error()
        return

    if not selected_experiment:
        st.info(EMPTY_STATE)
        return

    data = load_all()
    val_df, val_err = data["validation_report"]
    stat_df, stat_err = data["statistical_results"]
    dec_df, dec_err = data["decisions"]
    exec_df, exec_err = data["executive_summary"]
    _warn_if_error("validation_report.csv", val_err)
    _warn_if_error("statistical_results.csv", stat_err)
    _warn_if_error("decisions.csv", dec_err)
    _warn_if_error("executive_summary.csv", exec_err)

    val_row = safe_get_row(val_df, "experiment_id", selected_experiment)
    stat_row = safe_get_row(stat_df, "experiment_id", selected_experiment)
    dec_row = safe_get_row(dec_df, "experiment_id", selected_experiment)
    exec_row = safe_get_row(exec_df, "experiment_id", selected_experiment)

    feature_name = selected_experiment
    if val_row is not None and pd.notna(val_row.get("feature_name")):
        feature_name = val_row["feature_name"]

    decision = dec_row.get("decision", "Unknown") if dec_row is not None else "Unknown"
    reason = exec_row.get("primary_reason", "No summary available.") if exec_row is not None else "No summary available."
    next_action = exec_row.get("next_action", "N/A") if exec_row is not None else "N/A"

    is_blocked = False
    if decision.upper() == "BLOCKED" or (val_row is not None and pd.notna(val_row.get("blocking_reason"))):
        is_blocked = True

    _page_title(f"{selected_experiment} - {feature_name}", "Experiment readiness, decision, and primary evidence.")

    if is_blocked:
        st.error(
            "### 🔴 Experiment Failed Validation\n\n"
            "The experiment failed the randomization quality check. Because the treatment and control groups were not properly balanced, any measured difference between them cannot be trusted."
        )

        st.markdown(
            "#### Why this matters\n"
            "- Statistical results would be misleading\n"
            "- Business decisions could be incorrect\n"
            "- The experiment should be rerun after fixing traffic allocation"
        )

        st.info(
            "#### Next Recommended Action\n"
            "Fix the traffic allocation issue and rerun the experiment before reviewing treatment performance."
        )
    else:
        _panel(
            f'<div style="margin-bottom: 20px;">{_decision_pill(decision)}</div>'
            f'<div style="font-weight:600; color:#f8fafc; font-size: 16px; margin-bottom: 12px;">{escape(str(reason))}</div>'
            f'<div style="color:#94a3b8; font-size: 15px;">Next action: <span style="color:#e2e8f0;">{escape(str(next_action))}</span></div>'
        )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div style="font-size: 20px; font-weight: 600; color: #f8fafc; margin-bottom: 16px; letter-spacing: -0.3px;">Validation</div>', unsafe_allow_html=True)
        if val_row is None:
            st.info("No validation data available.")
        else:
            def get_tone(val):
                if pd.isna(val) or val is None:
                    return "neutral"
                s = str(val).lower()
                if s in ("clean", "pass", "true", "ready", "yes"):
                    return "success"
                if s in ("violated", "fail", "false", "no"):
                    return "danger"
                return "warning"

            _colored_kv_grid(
                [
                    ("Experiment Health Score", _format_float(val_row.get("ers_score"), 0), get_tone(val_row.get("ers_label"))),
                    ("Deployment Readiness", val_row.get("ers_label", "N/A"), get_tone(val_row.get("ers_label"))),
                    ("Randomization Check", symbol_for_status(val_row.get("srm_passed")), get_tone(val_row.get("srm_passed"))),
                    ("Guardrail Evaluation", val_row.get("guardrail_status", "N/A"), get_tone(val_row.get("guardrail_status"))),
                ]
            )

            if is_blocked:
                st.error(
                    "#### Deployment Recommendation\n\n"
                    "This experiment should not be used for product decisions.\n\n"
                    "The randomization validation failed before statistical analysis could begin. Fix the traffic allocation issue and rerun the experiment.\n\n"
                    "**No statistical conclusions should be drawn from this run.**"
                )

    with col2:
        st.markdown('<div style="font-size: 20px; font-weight: 600; color: #f8fafc; margin-bottom: 16px; letter-spacing: -0.3px;">Statistics</div>', unsafe_allow_html=True)
        analyzed = stat_row is not None and stat_row.get("status") == "ANALYZED"
        if not analyzed or pd.isna(stat_row.get("p_value_uncorrected")):
            st.markdown(
                """
                <div style="background: #15171e; border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 24px; color: #94a3b8; font-size: 15px; line-height: 1.5;">
                No statistical analysis was performed because the experiment did not pass validation. Results are intentionally withheld to prevent incorrect business decisions.
                </div>
                """, unsafe_allow_html=True
            )
        else:
            _colored_kv_grid(
                [
                    ("Absolute lift", _format_float(stat_row.get("absolute_lift"), 4), "neutral"),
                    ("Relative lift", _format_percent(stat_row.get("relative_lift"), 1), "neutral"),
                    ("p-value", format_pvalue(stat_row.get("p_value_uncorrected")), "neutral"),
                    ("Bayesian P(B>A)", format_probability(stat_row.get("bayesian_prob_positive")), "neutral"),
                ]
            )

    if stat_row is not None and pd.notna(stat_row.get("control_mean")) and pd.notna(stat_row.get("treatment_mean")):
        st.markdown('<div style="font-size: 20px; font-weight: 600; color: #f8fafc; margin-top: 40px; margin-bottom: 16px; letter-spacing: -0.3px;">Control vs Treatment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        values = [stat_row["control_mean"], stat_row["treatment_mean"]]
        ax.bar(["Control", "Treatment"], values, color=["#334155", "#FF4B00"], width=0.4, edgecolor="#ffffff", linewidth=0)
        ax.set_ylabel("Primary metric")
        _setup_chart(ax, fig)
        for idx, val in enumerate(values):
            ax.text(idx, val + (max(values)*0.02), f"{val:.3f}", ha="center", va="bottom", color="#f8fafc", fontweight="bold")

        st.markdown('<div style="background: #15171e; padding: 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close(fig)
