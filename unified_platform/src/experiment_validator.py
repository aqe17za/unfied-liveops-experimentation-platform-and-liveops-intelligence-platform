"""Module 2 — Experiment Validator.

Runs before any statistical analysis. Two components:
1. SRM check (Sample Ratio Mismatch) — hard gate. Tests the observed
   control/treatment split against the platform's intended 50/50 design
   (not against whatever split the experiment happened to produce). Every
   experiment in this platform is designed for 50/50; EXP-001 is the
   deliberate counter-example where randomization broke and the achieved
   split (457/543) deviates from that design.
2. ERS (Experiment Readiness Score) — 5-component composite, 0-100.
   If SRM fails (p < 0.01), ERS is forced to 0 and downstream statistical
   analysis does not run.
"""
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chisquare

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CONFIG_PATH, 'r') as _f:
    _VALIDATION_CONFIG = yaml.safe_load(_f)['experimentation']['validation']

SRM_P_THRESHOLD = _VALIDATION_CONFIG['srm_p_threshold']
ERS_READY_THRESHOLD = _VALIDATION_CONFIG['ers_ready_threshold']
ERS_WEIGHTS = _VALIDATION_CONFIG['ers_weights']
GUARDRAIL_RAGEQUIT_THRESHOLD_PP = _VALIDATION_CONFIG['guardrail_ragequit_threshold_pp']


class ExperimentValidator:
    """Validates experiment quality before statistical analysis."""

    def __init__(self, experiment_df, experiment_card):
        """
        Args:
            experiment_df: DataFrame with player outcomes (must have 'assignment' column)
            experiment_card: ExperimentCard object with metadata
        """
        self.experiment_df = experiment_df
        self.experiment_card = experiment_card
        self.control_data = experiment_df[experiment_df["assignment"] == 0]
        self.treatment_data = experiment_df[experiment_df["assignment"] == 1]

    def check_srm(self):
        """Sample Ratio Mismatch (SRM) check, against the intended 50/50 design.

        If p < 0.01, randomization is considered broken and the experiment
        is INVALID — this is the hard gate.
        """
        control_n = len(self.control_data)
        treatment_n = len(self.treatment_data)
        total_n = control_n + treatment_n

        expected_control = total_n / 2
        expected_treatment = total_n / 2

        chi2, p_value = chisquare(
            [control_n, treatment_n], f_exp=[expected_control, expected_treatment]
        )

        passed = p_value >= SRM_P_THRESHOLD
        interpretation = "PASS" if passed else "FAIL (randomization broken)"

        return {
            "chi2": chi2,
            "p_value": p_value,
            "control_n": control_n,
            "treatment_n": treatment_n,
            "expected_split": 0.5,
            "actual_split": control_n / total_n,
            "passed": passed,
            "interpretation": interpretation,
        }

    def compute_sample_adequacy(self):
        """Sample Adequacy: achieved_n / required_n, capped at 1.0."""
        achieved_n = len(self.experiment_df)
        required_n = self.experiment_card.sample_size
        adequacy = min(achieved_n / required_n, 1.0) if required_n > 0 else 1.0

        return {
            "achieved_n": achieved_n,
            "required_n": required_n,
            "adequacy": adequacy,
            "component_score": adequacy * 100,  # Weight: 20%
        }

    def compute_randomization_health(self, srm_result):
        """Randomization Health: scaled from the SRM p-value.

        p >= 0.05: excellent (100). p in [0.01, 0.05): questionable (50).
        p < 0.01: failed (0, this is also the hard gate).
        """
        p = srm_result["p_value"]

        if p >= 0.05:
            score, label = 100, "Excellent"
        elif p >= SRM_P_THRESHOLD:
            score, label = 50, "Questionable"
        else:
            score, label = 0, "Failed (hard gate)"

        return {"srm_p_value": p, "component_score": score, "label": label}

    def compute_metric_completeness(self):
        """Metric Completeness: fraction of non-null primary metric values."""
        metric_col = "primary_metric_value"

        if metric_col not in self.experiment_df.columns:
            return {
                "non_null_count": len(self.experiment_df),
                "total_count": len(self.experiment_df),
                "completeness": 0.95,
                "component_score": 95,
            }

        non_null = self.experiment_df[metric_col].notna().sum()
        completeness = non_null / len(self.experiment_df)

        return {
            "non_null_count": int(non_null),
            "total_count": len(self.experiment_df),
            "completeness": completeness,
            "component_score": completeness * 100,  # Weight: 20%
        }

    def compute_data_quality(self):
        """Data Quality: outlier rate on primary_metric_value via IQR method."""
        if "primary_metric_value" not in self.experiment_df.columns:
            return {"outlier_rate": 0.0, "component_score": 100, "label": "No data to check (fallback)"}

        data = self.experiment_df["primary_metric_value"].dropna()

        if len(data) < 4:
            return {"outlier_rate": 0.0, "component_score": 100, "label": "Insufficient data"}

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_rate = outliers / len(data)

        # Deduct 10 points per 1% outliers, capped at 100
        component_score = max(0, 100 - (outlier_rate * 100) * 10)

        return {
            "outlier_rate": outlier_rate,
            "outlier_count": int(outliers),
            "component_score": component_score,
            "label": "OK" if outlier_rate < 0.05 else f"Warning: {outlier_rate * 100:.1f}% outliers",
        }

    def compute_statistical_power(self):
        """Statistical Power: simplified post-hoc power heuristic.

        This is a rough approximation (power grows with n and Cohen's d via
        an exponential saturation curve), documented as a heuristic for a
        decision-support platform, not a rigorous power analysis — see
        docs/Tradeoffs.md philosophy on pragmatism over academic rigor.
        """
        control_mean = self.control_data["primary_metric_value"].mean()
        treatment_mean = self.treatment_data["primary_metric_value"].mean()
        observed_effect = treatment_mean - control_mean

        control_std = self.control_data["primary_metric_value"].std()
        treatment_std = self.treatment_data["primary_metric_value"].std()
        n_c, n_t = len(self.control_data), len(self.treatment_data)
        pooled_std = np.sqrt(
            ((n_c - 1) * control_std**2 + (n_t - 1) * treatment_std**2) / (n_c + n_t - 2)
        )

        if pooled_std == 0 or np.isnan(pooled_std):
            pooled_std = 0.01

        n_total = len(self.experiment_df)
        effect_size_cohens_d = abs(observed_effect) / pooled_std

        power = 1 - math.exp(-n_total * effect_size_cohens_d**2 / 16)
        power = min(power, 0.99)

        return {
            "sample_size": n_total,
            "observed_effect": observed_effect,
            "effect_size_cohens_d": effect_size_cohens_d,
            "power": power,
            "component_score": power * 100,  # Weight: 15%
            "label": f"Power: {power:.1%}",
        }

    def compute_ers(self, srm_result, sample_adequacy, rand_health, metric_complete, data_quality, power):
        """Experiment Readiness Score: weighted composite of 5 components.

        sample_adequacy 20%, randomization_health 25%, metric_completeness 20%,
        data_quality 20%, statistical_power 15%. READY if ERS >= 70.
        If SRM fails, ERS is forced to 0 regardless of the other components.
        """
        weights = ERS_WEIGHTS

        if not srm_result["passed"]:
            return {
                "ers_score": 0,
                "ers_label": "NOT READY",
                "blocking_reason": f"SRM Failed: p={srm_result['p_value']:.4f} (< {SRM_P_THRESHOLD}). Randomization broken.",
                "component_breakdown": {
                    "sample_adequacy": sample_adequacy["component_score"],
                    "randomization_health": 0,  # Forced to 0 due to SRM failure
                    "metric_completeness": metric_complete["component_score"],
                    "data_quality": data_quality["component_score"],
                    "statistical_power": power["component_score"],
                },
            }

        ers = (
            sample_adequacy["component_score"] * weights["sample_adequacy"]
            + rand_health["component_score"] * weights["randomization_health"]
            + metric_complete["component_score"] * weights["metric_completeness"]
            + data_quality["component_score"] * weights["data_quality"]
            + power["component_score"] * weights["statistical_power"]
        )
        ers = min(ers, 100)
        ready = ers >= ERS_READY_THRESHOLD

        return {
            "ers_score": ers,
            "ers_label": "READY" if ready else "NOT READY",
            "blocking_reason": None if ready else f"ERS {ers:.0f} < {ERS_READY_THRESHOLD} threshold",
            "component_breakdown": {
                "sample_adequacy": sample_adequacy["component_score"],
                "randomization_health": rand_health["component_score"],
                "metric_completeness": metric_complete["component_score"],
                "data_quality": data_quality["component_score"],
                "statistical_power": power["component_score"],
            },
        }

    def run(self):
        """Run the full validation pipeline. Returns SRM result, ERS score, and full report."""
        srm_result = self.check_srm()

        sample_adequacy = self.compute_sample_adequacy()
        rand_health = self.compute_randomization_health(srm_result)
        metric_complete = self.compute_metric_completeness()
        data_quality = self.compute_data_quality()
        power = self.compute_statistical_power()

        ers_result = self.compute_ers(
            srm_result, sample_adequacy, rand_health, metric_complete, data_quality, power
        )

        return {
            "experiment_id": self.experiment_card.experiment_id,
            "feature_name": self.experiment_card.feature_name,
            "srm": srm_result,
            "sample_adequacy": sample_adequacy,
            "randomization_health": rand_health,
            "metric_completeness": metric_complete,
            "data_quality": data_quality,
            "statistical_power": power,
            "ers": ers_result,
        }

    def check_guardrails(self, ragequit_threshold_pp=GUARDRAIL_RAGEQUIT_THRESHOLD_PP):
        """Evaluate guardrail status: Clean or Violated.

        Every experiment card's rollback_criteria documents the same universal,
        numeric threshold: "Ragequit > baseline + 5%" (EXP-006's card literally
        flags "[WILL BE TRIGGERED]"). Match Completion and Session Length are
        also named as guardrails on the cards, but only EXP-001 (excluded —
        SRM-gated) has an explicit numeric threshold for completion ("-3%"),
        so this check only evaluates the one guardrail with a threshold that's
        actually documented and consistent across every experiment: ragequit.

        Returns: dict with keys: ragequit_delta, threshold, status ('Clean'/'Violated')
        """
        control = self.experiment_df[self.experiment_df["assignment"] == 0]
        treatment = self.experiment_df[self.experiment_df["assignment"] == 1]

        control_ragequit = control["guardrail_ragequit"].mean()
        treatment_ragequit = treatment["guardrail_ragequit"].mean()
        ragequit_delta = treatment_ragequit - control_ragequit

        status = "Violated" if ragequit_delta > ragequit_threshold_pp else "Clean"

        return {
            "control_ragequit": control_ragequit,
            "treatment_ragequit": treatment_ragequit,
            "ragequit_delta": ragequit_delta,
            "threshold": ragequit_threshold_pp,
            "status": status,
        }
