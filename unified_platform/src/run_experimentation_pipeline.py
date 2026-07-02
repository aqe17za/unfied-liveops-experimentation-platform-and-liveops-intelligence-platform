"""Runs Experimentation Platform Phases 2-5 as a script.

Phase 1 (experiment_manager.py) has its own script entry point. Phases 2-5
previously existed only as notebook logic (notebooks/02-05 and
phase4_enhance_hte_classifier.ipynb); this module reproduces that same
orchestration as a headless, runnable script so the full pipeline can run
end-to-end without launching Jupyter.

Usage:
    python src/experiment_manager.py            # Phase 1
    python src/run_experimentation_pipeline.py  # Phases 2-5
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.experiment_manager import EXPERIMENT_SPECS, ExperimentCard
from src.experiment_validator import ExperimentValidator
from src.statistical_engine import StatisticalEngine
from src.heterogeneous_effects import HeterogeneousEffects
from src.hte_classifier import HTEClassifier
from src.decision_engine import DecisionEngine
from src.executive_summary import ExecutiveSummary

SIMULATION_DIR = ROOT / "data" / "simulation"

EXPERIMENT_IDS = [spec["experiment_id"] for spec in EXPERIMENT_SPECS]


def run_phase2_validation(experiment_results, experiment_cards):
    """Module 2 — SRM + ERS validation, enriched with guardrail_status."""
    validation_results = []
    for exp_id in EXPERIMENT_IDS:
        exp_data = experiment_results[experiment_results["experiment_id"] == exp_id]
        exp_card = experiment_cards[exp_id]

        validator = ExperimentValidator(exp_data, exp_card)
        result = validator.run()
        guardrail = validator.check_guardrails()
        result["guardrail_status"] = guardrail["status"]
        validation_results.append(result)

        print(f"{exp_id}: SRM p={result['srm']['p_value']:.4f} -> {result['srm']['interpretation']} | "
              f"ERS={result['ers']['ers_score']:.0f} ({result['ers']['ers_label']}) | "
              f"Guardrail={result['guardrail_status']}")

    validation_report = pd.DataFrame([
        {
            "experiment_id": r["experiment_id"],
            "feature_name": r["feature_name"],
            "srm_p_value": r["srm"]["p_value"],
            "srm_passed": r["srm"]["passed"],
            "ers_score": r["ers"]["ers_score"],
            "ers_label": r["ers"]["ers_label"],
            "blocking_reason": r["ers"]["blocking_reason"],
            "guardrail_status": r["guardrail_status"],
        }
        for r in validation_results
    ])
    validation_report.to_csv(SIMULATION_DIR / "validation_report.csv", index=False)
    print(f"Saved {len(validation_report)} rows to {SIMULATION_DIR / 'validation_report.csv'}")
    return validation_report


def run_phase3_statistics(experiment_results, validation_report, experiment_cards, experiment_cards_df):
    """Module 3 — Frequentist + Bayesian + novelty analysis."""
    engine = StatisticalEngine()
    results = engine.run_all(experiment_results, validation_report, experiment_cards=experiment_cards)
    results = engine.enrich_with_bayesian(results, experiment_results, experiment_cards_df,
                                           primary_metric="primary_metric_value")
    results = engine.enrich_with_novelty(results, experiment_results)

    results_export = []
    for exp_id in sorted(results.keys()):
        r = results[exp_id]
        results_export.append({
            "experiment_id": exp_id,
            "status": r["status"],
            "control_mean": r.get("control_mean"),
            "treatment_mean": r.get("treatment_mean"),
            "control_n": r.get("control_n"),
            "treatment_n": r.get("treatment_n"),
            "absolute_lift": r.get("absolute_lift"),
            "relative_lift": r.get("relative_lift"),
            "effect_size_cohens_d": r.get("effect_size_cohens_d"),
            "p_value_uncorrected": r.get("p_value"),
            "p_bonferroni": r.get("p_bonferroni"),
            "p_bh": r.get("p_bh"),
            "ci_lower": r.get("ci_lower"),
            "ci_upper": r.get("ci_upper"),
            "significant_uncorrected": r.get("significant_uncorrected"),
            "significant_bh": r.get("significant_bh"),
            "bayesian_prob_positive": r.get("bayesian_prob_positive"),
            "bayesian_ci_lower": r.get("bayesian_ci_lower"),
            "bayesian_ci_upper": r.get("bayesian_ci_upper"),
            "bayesian_model_suggested": r.get("bayesian_model_suggested"),
            "bayesian_model_final": r.get("bayesian_model_final"),
            "bayesian_model_selection_reason": r.get("bayesian_model_selection_reason"),
            "bayesian_prior": r.get("bayesian_prior"),
            "significant_bayesian": r.get("significant_bayesian"),
            "novelty_decay_ratio": r.get("novelty_decay_ratio"),
            "novelty_detected": r.get("novelty_detected"),
            "novelty_early_lift": r.get("novelty_early_lift"),
            "novelty_late_lift": r.get("novelty_late_lift"),
        })

    results_df = pd.DataFrame(results_export)
    results_df.to_csv(SIMULATION_DIR / "statistical_results.csv", index=False)
    print(f"Saved {len(results_df)} rows to {SIMULATION_DIR / 'statistical_results.csv'}")
    return results_df


def run_phase4_hte(experiment_results, statistical_results):
    """Module 4 — Heterogeneous treatment effects + Designed/Incidental classification."""
    hte = HeterogeneousEffects()
    all_hte_results = hte.run_all(experiment_results, statistical_results, primary_metric="primary_metric_value")

    hte_export = []
    for exp_id in sorted(all_hte_results.keys()):
        for dimension in ["PXI Tier", "Player Type", "Play Style"]:
            df = all_hte_results[exp_id][dimension]
            for _, row in df.iterrows():
                hte_export.append({
                    "experiment_id": exp_id,
                    "dimension": dimension,
                    "segment_value": row["segment_value"],
                    "n_control": row["n_control"],
                    "n_treatment": row["n_treatment"],
                    "control_mean": row["control_mean"],
                    "treatment_mean": row["treatment_mean"],
                    "absolute_lift": row["absolute_lift"],
                    "relative_lift": row["relative_lift"],
                    "effect_size_cohens_d": row["effect_size_cohens_d"],
                    "p_value": row["p_value"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "recommendation": row["recommendation"],
                    "reasoning": row["reasoning"],
                })
    hte_results_df = pd.DataFrame(hte_export)
    hte_results_df.to_csv(SIMULATION_DIR / "heterogeneous_effects_results.csv", index=False)

    rollout_summary = hte.generate_rollout_summary(all_hte_results)
    rollout_summary.to_csv(SIMULATION_DIR / "rollout_strategy_summary.csv", index=False)
    print(f"Saved {len(hte_results_df)} rows to {SIMULATION_DIR / 'heterogeneous_effects_results.csv'}")
    print(f"Saved {len(rollout_summary)} rows to {SIMULATION_DIR / 'rollout_strategy_summary.csv'}")

    classifier = HTEClassifier()
    classifications, hte_summary = classifier.run(hte_results_df, rollout_summary)
    classifications.to_csv(SIMULATION_DIR / "hte_classification.csv", index=False)
    hte_summary.to_csv(SIMULATION_DIR / "hte_summary.csv", index=False)
    print(f"Saved {len(classifications)} rows to {SIMULATION_DIR / 'hte_classification.csv'}")
    print(f"Saved {len(hte_summary)} rows to {SIMULATION_DIR / 'hte_summary.csv'}")

    return hte_results_df, hte_summary


def run_phase5_decision(statistical_results, hte_summary, validation_report):
    """Module 5 — Decision synthesis + executive summary."""
    engine = DecisionEngine()
    all_decisions, audit_df = engine.run_all(statistical_results, hte_summary, validation_report)

    decisions_export = []
    for exp_id in sorted(all_decisions.keys()):
        dec = all_decisions[exp_id]
        if dec.get("status") == "BLOCKED":
            decisions_export.append({
                "experiment_id": exp_id,
                "decision": "BLOCKED",
                "reasoning": dec["reason"],
                "evidence_strength": None,
                "dcs": None,
                "escalate_to_review": False,
                "decision_path": None,
                "p_value_uncorrected": None,
                "bayesian_prob_positive": None,
                "guardrail_status": None,
                "novelty_detected": None,
                "hte_type": None,
            })
        else:
            decisions_export.append({
                "experiment_id": exp_id,
                "decision": dec["decision"],
                "reasoning": dec["reasoning"],
                "evidence_strength": dec["evidence_strength"],
                "dcs": dec["dcs"]["dcs"] if dec["dcs"] else None,
                "escalate_to_review": dec["escalate"],
                "decision_path": dec["decision_path_display"],
                "p_value_uncorrected": dec["p_value"],
                "bayesian_prob_positive": dec["bayesian_prob_positive"],
                "guardrail_status": dec["guardrail_status"],
                "novelty_detected": dec["novelty_detected"],
                "hte_type": dec["hte_type"],
            })

    decisions_df = pd.DataFrame(decisions_export)
    decisions_df.to_csv(SIMULATION_DIR / "decisions.csv", index=False)
    audit_df.to_csv(SIMULATION_DIR / "decision_audit.csv", index=False)
    print(f"Saved {len(decisions_df)} rows to {SIMULATION_DIR / 'decisions.csv'}")
    print(f"Saved {len(audit_df)} rows to {SIMULATION_DIR / 'decision_audit.csv'}")

    exec_summary = ExecutiveSummary()
    executive_summary_df = exec_summary.generate(all_decisions, audit_df)
    executive_summary_df.to_csv(SIMULATION_DIR / "executive_summary.csv", index=False)
    print(f"Saved {len(executive_summary_df)} rows to {SIMULATION_DIR / 'executive_summary.csv'}")

    return decisions_df, audit_df, executive_summary_df


def main():
    print("=" * 60)
    print("PHASE 2-5: VALIDATION -> STATISTICS -> HTE -> DECISION")
    print("=" * 60)

    experiment_results = pd.read_csv(SIMULATION_DIR / "experiment_results.csv")
    experiment_cards_df = pd.read_csv(SIMULATION_DIR / "experiment_cards.csv")
    experiment_cards = {spec["experiment_id"]: ExperimentCard(spec) for spec in EXPERIMENT_SPECS}

    print("\n--- Phase 2: Experiment Validator (SRM + ERS) ---")
    validation_report = run_phase2_validation(experiment_results, experiment_cards)

    print("\n--- Phase 3: Statistical Engine ---")
    statistical_results = run_phase3_statistics(
        experiment_results, validation_report, experiment_cards, experiment_cards_df
    )

    print("\n--- Phase 4: Heterogeneous Effects + HTE Classifier ---")
    hte_results, hte_summary = run_phase4_hte(experiment_results, statistical_results)

    print("\n--- Phase 5: Decision Engine + Executive Summary ---")
    decisions_df, audit_df, executive_summary_df = run_phase5_decision(
        statistical_results, hte_summary, validation_report
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(executive_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
