"""Module 5 — Decision Engine.

Pure decision synthesis: computes no statistics of its own. Consumes
Module 2 (validation_report.csv, including guardrail_status), Module 3
(statistical_results.csv, frequentist + Bayesian), and Module 4
(hte_summary.csv, Designed/Incidental classification) outputs and turns
them into a decision (SHIP / KILL / EXTEND / SEGMENT ROLLOUT / HUMAN REVIEW)
with a structured, auditable decision path.

guardrail_status note: validation_report.csv did not originally carry a
guardrail_status column (Module 2 only ever computed SRM + ERS). Without
it, every experiment defaults to 'Unknown', and Unknown is a hard gate to
HUMAN REVIEW here — which would have routed all 5 analyzed experiments to
HUMAN REVIEW regardless of their actual evidence. So Module 2 was enriched
(ExperimentValidator.check_guardrails) using the one guardrail with a
numeric threshold documented consistently across every experiment card —
"Ragequit > baseline + 5%" — before this module was written.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CONFIG_PATH, 'r') as _f:
    _DECISION_CONFIG = yaml.safe_load(_f)['experimentation']['decision']

DEFAULT_ALPHA = _DECISION_CONFIG['alpha']
DEFAULT_MDE = _DECISION_CONFIG['mde']
DEFAULT_EFFECT_THRESHOLD = _DECISION_CONFIG['effect_threshold']
DCS_WEIGHTS = _DECISION_CONFIG['dcs_weights']


class DecisionEngine:
    """Pure decision synthesis. Consumes Module 2-4 outputs."""

    def __init__(self, alpha=DEFAULT_ALPHA, mde=DEFAULT_MDE, effect_threshold=DEFAULT_EFFECT_THRESHOLD):
        self.alpha = alpha
        self.mde = mde
        self.effect_threshold = effect_threshold  # [0-100] scale, not 0-1

    # ------------------------------------------------------------------
    # Component scores [0-100]
    # ------------------------------------------------------------------

    def score_frequentist_signal(self, p_value):
        """Tiered p-value scoring (conservative thresholds).

        p < 0.001: 100. p < 0.01: 90. p < 0.05: 75.
        0.05 <= p < 0.20: 30. p >= 0.20: 0 (genuinely no statistical support,
        distinct from the 30 given to the 0.05-0.20 "weak evidence" band).
        """
        if p_value < 0.001:
            return 100
        elif p_value < 0.01:
            return 90
        elif p_value < 0.05:
            return 75
        elif p_value < 0.20:
            return 30
        else:
            return 0

    def score_bayesian_signal(self, bayesian_prob, absolute_lift):
        """100 x (P(B>A) - 0.5) / 0.5, capped at 50 if |lift| < 0.5xMDE
        (statistical confidence in direction shouldn't read as strong
        evidence when the practical lift is negligible)."""
        if bayesian_prob <= 0.5:
            return 0
        base = 100 * (bayesian_prob - 0.5) / 0.5
        if abs(absolute_lift) < 0.5 * self.mde:
            return min(base, 50)
        return min(base, 100)

    def score_effect_size(self, absolute_lift):
        """(|lift| / MDE) x 100, capped at 100."""
        if self.mde == 0:
            return 50
        return min(abs(absolute_lift) / self.mde * 100, 100)

    def score_validation(self, ers_score):
        """ERS directly (already 0-100)."""
        return ers_score

    def score_hte_confidence(self, hte_confidence):
        """High: 100, Medium: 75, Low: 50, None: 50."""
        return {"High": 100, "Medium": 75, "Low": 50, "None": 50}.get(hte_confidence, 50)

    # ------------------------------------------------------------------
    # Evidence assessment
    # ------------------------------------------------------------------

    def assess_evidence_strength(self, freq_score, bayes_score, effect_score):
        """Continuous evidence with an effect-size prerequisite.

        Effect score is a GATE (must be >= effect_threshold) — statistical
        significance without practical significance can never read as Strong.
        """
        signals = [freq_score, bayes_score, effect_score]
        avg = np.mean(signals)
        min_sig = np.min(signals)
        count_strong = sum(1 for s in signals if s > 75)

        if effect_score < self.effect_threshold:
            return "Weak", avg

        if count_strong >= 3 or avg >= 80:
            return "Strong", avg

        if min_sig < 40 or avg < 60:
            return "Weak", avg

        return "Moderate", avg

    # ------------------------------------------------------------------
    # Signal checks
    # ------------------------------------------------------------------

    def check_bh_warning(self, p_value, p_bh):
        """BH warning: significant uncorrected but not after BH correction."""
        return (p_value < 0.05) and (p_bh >= 0.05)

    def check_signal_disagreement(self, freq_score, bayes_score):
        """Disagreement: one signal strong (>75), the other weak (<40)."""
        return (freq_score > 75 and bayes_score < 40) or (bayes_score > 75 and freq_score < 40)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def decide(
        self, evidence_strength, guardrail_status, effect_sign,
        p_value, p_bh, hte_type, hte_rollout_eligible, hte_confidence,
        absolute_lift, freq_score, bayes_score, novelty_detected=False,
    ):
        """Evidence-first decision with structured path tracing.

        Returns: (decision, reasoning, escalate, decision_path_list, decision_path_display)
        """
        steps = []
        display_steps = []

        # HARD GATES
        if guardrail_status == "Violated":
            steps.append("guardrail_violated")
            display_steps.append("Guardrail Violation")
            return "KILL", "Guardrail violation", False, steps, " -> ".join(display_steps)

        if guardrail_status == "Unknown":
            steps.append("guardrail_unknown")
            display_steps.append("Guardrail Status Unknown")
            return "HUMAN REVIEW", "Guardrail status unknown", True, steps, " -> ".join(display_steps)

        steps.append("guardrails_clean")
        display_steps.append("Guardrails Clean")

        if effect_sign < 0:
            steps.append("negative_effect")
            display_steps.append("Negative Effect")
            return "KILL", "Negative treatment effect", False, steps, " -> ".join(display_steps)

        steps.append("positive_effect")
        display_steps.append("Positive Effect")

        # NOVELTY GATE — per docs/PRD.md: "KILL — guardrail failure OR
        # rollback triggered OR novelty detected". An early-period effect
        # that decays (or reverses) by the late period is a false positive
        # regardless of how strong the pooled evidence looks; this overrides
        # evidence strength the same way a guardrail violation does.
        if novelty_detected:
            steps.append("novelty_detected")
            display_steps.append("Novelty Effect Detected")
            return "KILL", "Novelty effect: early lift does not persist late", False, steps, " -> ".join(display_steps)

        # SIGNAL CHECKS
        bh_warning = self.check_bh_warning(p_value, p_bh)
        signal_disagree = self.check_signal_disagreement(freq_score, bayes_score)

        if signal_disagree:
            steps.append("signal_disagreement")
            display_steps.append("Frequentist-Bayesian Disagreement")
            return "HUMAN REVIEW", "Frequentist-Bayesian disagreement", True, steps, " -> ".join(display_steps)

        steps.append(f"evidence_{evidence_strength.lower()}")
        display_steps.append(f"Evidence: {evidence_strength}")

        if bh_warning:
            steps.append("bh_warning")
            display_steps.append("BH Multiple Testing Warning")

        # HTE ELIGIBILITY (only Designed + rollout_eligible + High confidence)
        segment_eligible = (
            hte_type == "Designed" and hte_rollout_eligible and hte_confidence == "High"
        )

        if segment_eligible:
            steps.append("hte_designed_eligible")
            display_steps.append("HTE: Designed + Rollout Eligible + High Confidence")

        # DECISIONS
        if evidence_strength == "Strong":
            if segment_eligible:
                steps.append("decision_segment_rollout")
                display_steps.append("SEGMENT ROLLOUT")
                return "SEGMENT ROLLOUT", "Strong evidence, Designed HTE", False, steps, " -> ".join(display_steps)
            steps.append("decision_ship")
            display_steps.append("SHIP")
            return "SHIP", "Strong evidence", False, steps, " -> ".join(display_steps)

        if evidence_strength == "Moderate":
            if bh_warning:
                if segment_eligible:
                    steps.append("decision_segment_rollout_with_warning")
                    display_steps.append("SEGMENT ROLLOUT (BH Warning)")
                    return (
                        "SEGMENT ROLLOUT", "Moderate evidence, Designed HTE (BH warning)",
                        False, steps, " -> ".join(display_steps),
                    )
                steps.append("decision_extend_bh_warning")
                display_steps.append("EXTEND (BH Warning)")
                return (
                    "EXTEND", "Moderate evidence, BH correction warning",
                    False, steps, " -> ".join(display_steps),
                )
            if segment_eligible:
                steps.append("decision_segment_rollout")
                display_steps.append("SEGMENT ROLLOUT")
                return "SEGMENT ROLLOUT", "Moderate evidence, Designed HTE", False, steps, " -> ".join(display_steps)
            steps.append("decision_ship")
            display_steps.append("SHIP")
            return "SHIP", "Moderate evidence", False, steps, " -> ".join(display_steps)

        if evidence_strength == "Weak":
            # Mirrors the Strong/Moderate segment_eligible check: a Designed,
            # High-confidence, rollout-eligible HTE finding should still be
            # able to produce SEGMENT ROLLOUT even when the pooled/overall
            # effect reads as Weak — that's the entire reason heterogeneous
            # effects analysis exists (the overall signal is diluted by
            # mixed segment effects; this is the recommended fix when that
            # happens, not a fallback).
            if segment_eligible:
                steps.append("decision_segment_rollout_weak")
                display_steps.append("SEGMENT ROLLOUT (Weak Pooled Evidence)")
                return (
                    "SEGMENT ROLLOUT", "Weak pooled evidence, but Designed HTE rescues it",
                    False, steps, " -> ".join(display_steps),
                )
            if absolute_lift > self.mde * 0.5:
                steps.append("decision_extend_weak")
                display_steps.append("EXTEND (Weak but Positive)")
                return (
                    "EXTEND", "Weak evidence, positive effect (power analysis recommended)",
                    False, steps, " -> ".join(display_steps),
                )
            steps.append("decision_human_review_weak")
            display_steps.append("HUMAN REVIEW (Weak Evidence)")
            return "HUMAN REVIEW", "Weak evidence, needs review", True, steps, " -> ".join(display_steps)

        steps.append("decision_human_review_unclassifiable")
        display_steps.append("HUMAN REVIEW (Unclassifiable)")
        return "HUMAN REVIEW", "Unclassifiable", True, steps, " -> ".join(display_steps)

    # ------------------------------------------------------------------
    # DCS (computed ONLY for non-KILL, non-BLOCKED decisions — Rule 7).
    # A "confidence to ship" score doesn't mean anything for a decision
    # that's already KILL, and BLOCKED experiments never reach this method.
    # ------------------------------------------------------------------

    def compute_dcs(self, ers_score, freq_score, bayes_score, guardrails_clean, effect_score, hte_score):
        """40% foundational + 40% statistical + 20% practical."""
        guardrail_score = 100 if guardrails_clean else 0
        w = DCS_WEIGHTS
        dcs = (
            w['ers'] * ers_score
            + w['frequentist'] * freq_score
            + w['bayesian'] * bayes_score
            + w['guardrail'] * guardrail_score
            + w['effect'] * effect_score
            + w['hte'] * hte_score
        )
        return {
            "dcs": dcs,
            "validation_score": ers_score,
            "frequentist_score": freq_score,
            "bayesian_score": bayes_score,
            "guardrail_score": guardrail_score,
            "effect_score": effect_score,
            "hte_score": hte_score,
        }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run_all(self, stat_results_df, hte_summary_df, val_report_df):
        """Synthesize all signals, make decisions."""
        ers_dict = dict(zip(val_report_df["experiment_id"], val_report_df["ers_score"]))

        guardrail_dict = {}
        if "guardrail_status" in val_report_df.columns:
            guardrail_dict = dict(zip(val_report_df["experiment_id"], val_report_df["guardrail_status"]))

        hte_dict = hte_summary_df.set_index("experiment_id").to_dict("index")

        decisions = {}
        audit_rows = []

        for _, row in stat_results_df.iterrows():
            exp_id = row["experiment_id"]

            if row["status"] != "ANALYZED":
                decisions[exp_id] = {"status": "BLOCKED", "reason": row["status"]}
                continue

            # SIGNALS (pre-computed, no re-computation)
            p_value = row["p_value_uncorrected"]
            p_bh = row["p_bh"]
            absolute_lift = row["absolute_lift"]
            effect_sign = np.sign(absolute_lift)
            bayesian_prob = row["bayesian_prob_positive"]
            model_reason = row.get("bayesian_model_selection_reason", "N/A")
            novelty_detected = bool(row.get("novelty_detected", False))

            # Guardrails (READ, default to Unknown if missing)
            guardrail_status = guardrail_dict.get(exp_id, "Unknown")
            if guardrail_status not in ["Clean", "Violated", "Unknown"]:
                guardrail_status = "Unknown"
            guardrails_clean = guardrail_status == "Clean"

            # HTE
            hte_info = hte_dict.get(exp_id, {})
            hte_type = hte_info.get("hte_type", "None")
            hte_confidence = hte_info.get("hte_confidence", "None")
            hte_rollout_eligible = hte_info.get("rollout_eligible", False)

            # ERS
            ers_score = ers_dict.get(exp_id, 50)

            # COMPONENT SCORES
            freq_score = self.score_frequentist_signal(p_value)
            bayes_score = self.score_bayesian_signal(bayesian_prob, absolute_lift)
            effect_score = self.score_effect_size(absolute_lift)
            hte_score = self.score_hte_confidence(hte_confidence)

            # EVIDENCE STRENGTH (with effect prerequisite)
            evidence_strength, evidence_avg = self.assess_evidence_strength(freq_score, bayes_score, effect_score)

            # DECISION (with structured path tracing)
            decision, reasoning, escalate, decision_path_list, decision_path_display = self.decide(
                evidence_strength, guardrail_status, effect_sign,
                p_value, p_bh, hte_type, hte_rollout_eligible, hte_confidence,
                absolute_lift, freq_score, bayes_score, novelty_detected=novelty_detected,
            )

            # DCS only for decisions where a "confidence to ship" reading is
            # meaningful — not KILL (Rule 7).
            if decision != "KILL":
                dcs_components = self.compute_dcs(
                    ers_score, freq_score, bayes_score, guardrails_clean, effect_score, hte_score
                )
            else:
                dcs_components = None

            decisions[exp_id] = {
                "decision": decision,
                "reasoning": reasoning,
                "decision_path_list": decision_path_list,
                "decision_path_display": decision_path_display,
                "escalate": escalate,
                "dcs": dcs_components,
                "evidence_strength": evidence_strength,
                "evidence_avg_score": evidence_avg,
                "p_value": p_value,
                "bayesian_prob_positive": bayesian_prob,
                "guardrail_status": guardrail_status,
                "hte_type": hte_type,
                "hte_confidence": hte_confidence,
                "novelty_detected": novelty_detected,
            }

            # AUDIT TRAIL
            bh_warning = self.check_bh_warning(p_value, p_bh)
            signal_disagree = self.check_signal_disagreement(freq_score, bayes_score)

            audit_rows.append({
                "experiment_id": exp_id,
                "decision": decision,
                "novelty_detected": novelty_detected,
                "decision_path_list": ",".join(decision_path_list),
                "decision_path_display": decision_path_display,
                "escalate_to_review": escalate,
                "evidence_strength": evidence_strength,
                "evidence_avg_score": f"{evidence_avg:.0f}",
                "guardrail_status": guardrail_status,
                "p_value_uncorrected": f"{p_value:.4f}",
                "p_bh": f"{p_bh:.4f}",
                "bh_warning": bh_warning,
                "frequentist_score": f"{freq_score:.0f}",
                "bayesian_prob_positive": f"{bayesian_prob:.2%}",
                "bayesian_score": f"{bayes_score:.0f}",
                "effect_score": f"{effect_score:.0f}",
                "absolute_lift": f"{absolute_lift:.4f}",
                "hte_type": hte_type,
                "hte_confidence": hte_confidence,
                "hte_rollout_eligible": hte_rollout_eligible,
                "signal_disagreement": signal_disagree,
                "dcs_final": f"{dcs_components['dcs']:.0f}" if dcs_components is not None else "N/A",
                "model_selection_reason": model_reason,
                "reasoning": reasoning,
            })

        return decisions, pd.DataFrame(audit_rows)
