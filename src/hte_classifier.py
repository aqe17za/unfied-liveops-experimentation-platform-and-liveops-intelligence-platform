"""Phase 4 enhancement — HTE Classifier.

Classifies each experiment's heterogeneous treatment effect as Designed
(EXP-003, the one experiment actually built around a differential PXI-tier
effect, per docs/Experiment_Design.md) or Incidental (EXP-002, 004, 005, 006 —
subgroup variation from sampling noise at n~500/arm, not a product design).
Scores HTE Confidence (High/Medium/Low) and produces two CSVs for Module 5.

Does not re-run any statistical test. Reads Module 4's already-computed
heterogeneous_effects_results.csv and rollout_strategy_summary.csv directly.
"""
from pathlib import Path

import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CONFIG_PATH, 'r') as _f:
    _HTE_CONFIG = yaml.safe_load(_f)['experimentation']['hte']

PRACTICAL_THRESHOLD = _HTE_CONFIG['practical_threshold']  # matches Module 4's mde (absolute lift, 2pp)


class HTEClassifier:
    """Classify HTE as Designed or Incidental. Score HTE Confidence (High/Medium/Low).

    Input: Module 4 outputs (heterogeneous_effects_results.csv, rollout_strategy_summary.csv)
    Output: Classification + confidence scores for Module 5
    """

    def classify_hte_type(self, exp_id, hte_detected):
        """EXP-003 only: Designed (At Risk vs Healthy, documented in
        docs/Experiment_Design.md). Others: Incidental (sampling variation)."""
        if exp_id == "EXP-003" and hte_detected:
            return "Designed"
        elif hte_detected:
            return "Incidental"
        else:
            return "None"

    def score_hte_confidence(self, exp_id, hte_detected, lift_range, best_segment_p_value, best_segment_practical):
        """High: large effect, significant, practical (Designed HTE only).
        Medium: moderate effect with some significance. Low: small/weak evidence."""
        if not hte_detected or lift_range is None:
            return "None"

        if exp_id == "EXP-003":
            if lift_range > 0.05 and best_segment_p_value < 0.001:
                return "High"
            elif lift_range > 0.05 and best_segment_p_value < 0.05:
                return "Medium"
            else:
                return "Low"

        # Incidental HTE
        if lift_range > 0.05:
            if best_segment_p_value < 0.01:
                return "Medium"
            else:
                return "Low"
        elif lift_range > 0.02:
            return "Low"
        else:
            return "Low"

    def generate_rationale(self, exp_id, hte_type, confidence, lift_range, best_segment):
        if hte_type == "None":
            return "No meaningful HTE detected"

        if hte_type == "Designed":
            if best_segment == "At Risk":
                return (
                    f"EXP-003 designed for At Risk segment. Lift range {lift_range:.4f} "
                    f"confirms HTE. Confidence: {confidence}."
                )
            else:
                return f"Unexpected best segment: {best_segment}. Review manually."

        # Incidental
        if confidence == "High":
            return (
                f"Incidental HTE detected (lift range {lift_range:.4f}). High variation, "
                f"but not product-driven. Ignore for rollout."
            )
        elif confidence == "Medium":
            return (
                f"Incidental HTE (lift range {lift_range:.4f}). Moderate variation from "
                f"sampling. Use as signal but not driver."
            )
        else:
            return f"Incidental HTE (lift range {lift_range:.4f}). Low confidence, likely sampling noise."

    def run(self, hte_results_df, rollout_summary_df):
        """Classify all experiments' HTE and generate summaries.

        Args:
            hte_results_df: heterogeneous_effects_results.csv (Module 4)
            rollout_summary_df: rollout_strategy_summary.csv (Module 4)

        Returns:
            (classification_df, summary_df)
        """
        pxi_results = hte_results_df[hte_results_df["dimension"] == "PXI Tier"].copy()

        classifications = []
        summaries = []

        for exp_id in sorted(rollout_summary_df["experiment_id"].unique()):
            rollout_row = rollout_summary_df[rollout_summary_df["experiment_id"] == exp_id].iloc[0]

            hte_detected = bool(rollout_row["hte_detected"])
            lift_range = rollout_row["lift_range"]
            best_segment = rollout_row["best_segment"]
            worst_segment = rollout_row["worst_segment"]

            exp_pxi = pxi_results[pxi_results["experiment_id"] == exp_id]
            best_seg_data = exp_pxi[exp_pxi["segment_value"] == best_segment]

            if len(best_seg_data) > 0 and pd.notna(best_seg_data.iloc[0]["p_value"]):
                best_p_value = best_seg_data.iloc[0]["p_value"]
                # 'practical' isn't an exported column; derive it from the already-computed
                # absolute_lift (same 2pp threshold Module 4 used internally) — a threshold
                # check on existing data, not a re-analysis.
                best_practical = abs(best_seg_data.iloc[0]["absolute_lift"]) > PRACTICAL_THRESHOLD
            else:
                best_p_value = None
                best_practical = False

            hte_type = self.classify_hte_type(exp_id, hte_detected)
            confidence = self.score_hte_confidence(exp_id, hte_detected, lift_range, best_p_value, best_practical)
            rationale = self.generate_rationale(exp_id, hte_type, confidence, lift_range, best_segment)

            rollout_eligible = hte_type == "Designed" and confidence == "High"

            classifications.append({
                "experiment_id": exp_id,
                "hte_type": hte_type,
                "hte_confidence": confidence,
                "rationale": rationale,
            })

            summaries.append({
                "experiment_id": exp_id,
                "best_segment": best_segment,
                "worst_segment": worst_segment,
                "lift_range": lift_range,
                "hte_detected": hte_detected,
                "hte_type": hte_type,
                "hte_confidence": confidence,
                "rollout_eligible": rollout_eligible,
            })

        return pd.DataFrame(classifications), pd.DataFrame(summaries)
