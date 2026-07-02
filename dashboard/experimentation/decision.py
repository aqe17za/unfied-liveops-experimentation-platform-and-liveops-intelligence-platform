"""Experimentation page — Decision trail."""
from html import escape

import streamlit as st

from dashboard.experimentation.common import (
    EMPTY_STATE,
    _colored_kv_grid,
    _decision_pill,
    _panel,
    _page_title,
    _rationale_textbox,
    _status_pill,
    _warn_if_error,
)
from dashboard.experimentation.data_loader import (
    data_missing,
    format_dcs,
    load_all,
    safe_get_row,
    show_missing_data_error,
    symbol_for_status,
)


def _generate_rationale(data, row, exp_id):
    decision = row.get("decision", "")
    guardrail = row.get("guardrail_status", "")
    novelty = row.get("novelty_detected", False)
    bh_warning = row.get("bh_warning", False)

    if decision == "KILL":
        if novelty:
            return "The observed improvement appears driven primarily by novelty and is unlikely to persist. Do not ship."
        if guardrail == "Violated":
            return "Although the primary metric improved, a guardrail metric regressed beyond acceptable thresholds. Do not ship."
        return "The experiment did not produce a positive effect or violated safety metrics. Do not ship."

    if decision == "SEGMENT ROLLOUT":
        best_segment = "relevant"
        hte_df, _ = data.get("hte_summary", (None, None))
        if hte_df is not None:
            hte_row = safe_get_row(hte_df, "experiment_id", exp_id)
            if hte_row is not None:
                best_segment = hte_row.get("best_segment", best_segment)
        return f"The overall effect is weak, but the pre-registered heterogeneous treatment effect is significant. Recommend rollout only to the {best_segment} player segment."

    if decision == "SHIP":
        text = "The experiment demonstrates statistically significant positive lift with no guardrail violations. "
        if bh_warning:
            text += "Evidence remains strong after multiple-testing correction. "
        text += "Recommend full rollout."
        return text

    if decision == "HUMAN REVIEW":
        return "The results are inconclusive or have conflicting signals. Escalate for manual review."

    return "No clear rationale available."


def render_decision(selected_experiment):
    if data_missing():
        show_missing_data_error()
        return

    if not selected_experiment:
        st.info(EMPTY_STATE)
        return

    data = load_all()
    audit_df, audit_err = data["decision_audit"]
    dec_df, dec_err = data["decisions"]
    _warn_if_error("decision_audit.csv", audit_err)
    _warn_if_error("decisions.csv", dec_err)

    _page_title(f"Decision Trail - {selected_experiment}", "Decision output, supporting logic, and signal checks.")

    audit_row = safe_get_row(audit_df, "experiment_id", selected_experiment)
    dec_row = safe_get_row(dec_df, "experiment_id", selected_experiment)

    if audit_row is None:
        st.info("No full audit trail is available for this experiment.")
        if dec_row is not None:
            _panel(f'<div style="color: #f1f5f9;"><b>Reasoning:</b> {escape(str(dec_row.get("reasoning", "N/A")))}</div>')
        return

    decision = audit_row.get("decision", "Unknown")
    _panel(
        f'<div style="margin-bottom: 20px;">{_decision_pill(decision)}</div>'
        f'<div style="color: #f8fafc; font-size: 15px; line-height: 1.6;"><b>Reasoning:</b> {escape(str(audit_row.get("reasoning", "N/A")))}</div>'
    )

    _colored_kv_grid(
        [
            ("Evidence strength", audit_row.get("evidence_strength", "N/A"), "neutral"),
            ("DCS", format_dcs(audit_row.get("dcs_final")), "neutral"),
            ("Guardrails", symbol_for_status(audit_row.get("guardrail_status")), "neutral"),
            ("Novelty", "Detected" if audit_row.get("novelty_detected", False) else "Not detected", "neutral"),
        ]
    )

    st.markdown('<div style="font-size: 20px; font-weight: 600; color: #f8fafc; margin-top: 32px; margin-bottom: 16px; letter-spacing: -0.3px;">Decision Path</div>', unsafe_allow_html=True)
    path = str(audit_row.get("decision_path_display", "N/A"))
    steps = [step.strip() for step in path.replace("=>", "->").split("->") if step.strip()]

    path_html = []
    for i, step in enumerate(steps):
        path_html.append(f'<span style="display: inline-block; padding: 8px 16px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; color: #cbd5e1; font-weight: 500; font-size: 14px;">{escape(step)}</span>')
        if i < len(steps) - 1:
            path_html.append('<span style="color: #64748b; margin: 0 8px;">➔</span>')

    st.markdown(f'<div style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 40px;">{"".join(path_html)}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 20px; font-weight: 600; color: #f8fafc; margin-bottom: 16px; letter-spacing: -0.3px;">Signal Checks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="display: flex; gap: 12px; margin-bottom: 32px;">' +
        " ".join(
            [
                _status_pill(audit_row.get("guardrail_status"), f"Guardrails: {audit_row.get('guardrail_status', 'N/A')}"),
                _status_pill(not audit_row.get("novelty_detected", False), "No novelty issue"),
                _status_pill(not audit_row.get("signal_disagreement", False), "Signals aligned"),
            ]
        ) + '</div>',
        unsafe_allow_html=True,
    )

    _rationale_textbox("Decision Rationale", _generate_rationale(data, audit_row, selected_experiment))
