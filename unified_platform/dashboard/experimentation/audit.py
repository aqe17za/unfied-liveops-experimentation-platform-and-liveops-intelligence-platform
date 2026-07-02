"""Experimentation page — Statistical audit."""
import pandas as pd
import streamlit as st

from dashboard.experimentation.common import (
    EMPTY_STATE,
    _colored_kv_grid,
    _format_float,
    _format_percent,
    _page_title,
    _rationale_textbox,
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


def _generate_bayesian_rationale(row):
    prob = row.get("bayesian_prob_positive")
    ci_lower = row.get("bayesian_ci_lower")
    ci_upper = row.get("bayesian_ci_upper")

    if pd.isna(prob) or pd.isna(ci_lower) or pd.isna(ci_upper):
        return "No Bayesian data available."

    prob_pct = prob * 100
    excludes_zero = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    text = f"Posterior probability that treatment outperforms control is {prob_pct:.1f}%. "

    if excludes_zero:
        text += "The 95% credible interval excludes zero, providing strong Bayesian evidence supporting "
        if ci_lower > 0:
            text += "treatment superiority."
        else:
            text += "control superiority."
    else:
        text += "The 95% credible interval includes zero, meaning the Bayesian evidence is not conclusive."

    return text


def render_audit(selected_experiment):
    if data_missing():
        show_missing_data_error()
        return

    if not selected_experiment:
        st.info(EMPTY_STATE)
        return

    data = load_all()
    val_df, val_err = data["validation_report"]
    stat_df, stat_err = data["statistical_results"]
    _warn_if_error("validation_report.csv", val_err)
    _warn_if_error("statistical_results.csv", stat_err)

    _page_title(f"Statistical Audit - {selected_experiment}", "Validation gates, frequentist stats, Bayesian evidence, and novelty checks.")

    val_row = safe_get_row(val_df, "experiment_id", selected_experiment)
    stat_row = safe_get_row(stat_df, "experiment_id", selected_experiment)
    analyzed = stat_row is not None and stat_row.get("status") == "ANALYZED"

    tab_validation, tab_freq, tab_bayes, tab_novelty = st.tabs(
        ["Validation", "Frequentist", "Bayesian", "Novelty"]
    )

    with tab_validation:
        st.markdown("<br/>", unsafe_allow_html=True)
        if val_row is None:
            st.info("No validation data available.")
        else:
            _colored_kv_grid(
                [
                    ("SRM", f"{symbol_for_status(val_row.get('srm_passed'))} p={format_pvalue(val_row.get('srm_p_value'))}", "neutral"),
                    ("ERS", f"{_format_float(val_row.get('ers_score'), 0)} / {val_row.get('ers_label', 'N/A')}", "neutral"),
                    ("Guardrails", val_row.get("guardrail_status", "N/A"), "neutral"),
                    ("Blocker", val_row.get("blocking_reason", "None") or "None", "neutral"),
                ]
            )

    with tab_freq:
        st.markdown("<br/>", unsafe_allow_html=True)
        if not analyzed:
            st.info("Not analyzed because the experiment was blocked by validation.")
        else:
            _colored_kv_grid(
                [
                    ("Absolute lift", _format_float(stat_row.get("absolute_lift"), 4), "neutral"),
                    ("Relative lift", _format_percent(stat_row.get("relative_lift"), 2), "neutral"),
                    ("Cohen's d", _format_float(stat_row.get("effect_size_cohens_d"), 3), "neutral"),
                    ("BH p-value", format_pvalue(stat_row.get("p_bh")), "neutral"),
                ]
            )
            st.caption(
                f"95% CI: [{_format_float(stat_row.get('ci_lower'), 4)}, "
                f"{_format_float(stat_row.get('ci_upper'), 4)}] | "
                f"Bonferroni: {format_pvalue(stat_row.get('p_bonferroni'))}"
            )

    with tab_bayes:
        st.markdown("<br/>", unsafe_allow_html=True)
        if not analyzed:
            st.info("Not analyzed because the experiment was blocked by validation.")
        else:
            _colored_kv_grid(
                [
                    ("P(treatment > control)", format_probability(stat_row.get("bayesian_prob_positive")), "neutral"),
                    ("Credible lower", _format_float(stat_row.get("bayesian_ci_lower"), 4), "neutral"),
                    ("Credible upper", _format_float(stat_row.get("bayesian_ci_upper"), 4), "neutral"),
                    ("Model", stat_row.get("bayesian_model_final", "N/A"), "neutral"),
                ]
            )
            st.caption(f"Prior: {stat_row.get('bayesian_prior', 'N/A')}")
            _rationale_textbox("Bayesian Rationale", _generate_bayesian_rationale(stat_row))

    with tab_novelty:
        st.markdown("<br/>", unsafe_allow_html=True)
        if not analyzed or pd.isna(stat_row.get("novelty_decay_ratio")):
            st.info("No novelty analysis is available for this experiment.")
        else:
            novelty_status = "Detected" if stat_row.get("novelty_detected", False) else "Not detected"
            _colored_kv_grid(
                [
                    ("Novelty", novelty_status, "neutral"),
                    ("Early lift", _format_float(stat_row.get("novelty_early_lift"), 4), "neutral"),
                    ("Late lift", _format_float(stat_row.get("novelty_late_lift"), 4), "neutral"),
                    ("Decay ratio", _format_float(stat_row.get("novelty_decay_ratio"), 2), "neutral"),
                ]
            )
