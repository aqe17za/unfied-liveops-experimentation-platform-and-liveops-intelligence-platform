"""Module 3 — Statistical Engine.

Runs frequentist statistical tests on validated experiments only. EXP-001
failed the SRM hard gate in Module 2 (p=0.0065 < 0.01) and is recorded as
BLOCKED here with null statistics — it is never tested.

Multiple-testing correction (Benjamini-Hochberg, primary; Bonferroni, for
reference) uses statsmodels.stats.multitest, not a manual implementation.
Confidence intervals use the t-distribution via scipy.stats.t, not a
hardcoded 1.96 z-critical value.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_ind, t, beta, norm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CONFIG_PATH, 'r') as _f:
    _STATS_CONFIG = yaml.safe_load(_f)['experimentation']['statistics']

# Maps an ExperimentCard's primary_metric label to the actual outcome column
# in experiment_results. Every one of the 6 experiments currently shares the
# same metric ("D7 Retention Rate"), but resolving through this map (rather
# than hardcoding 'primary_metric_value' inside the test functions) means a
# future experiment with a different primary metric only needs an entry here.
METRIC_COLUMN_MAP = {
    "D7 Retention Rate": "primary_metric_value",
}

DEFAULT_METRIC_COLUMN = "primary_metric_value"

# --- Bayesian layer constants ---
MC_SAMPLES = _STATS_CONFIG['mc_samples']
RANDOM_SEED = _STATS_CONFIG['random_seed']
BAYES_SIGNIFICANT_THRESHOLD = _STATS_CONFIG['bayes_significant_threshold']
DEFAULT_ALPHA = _STATS_CONFIG['alpha']
NOVELTY_DECAY_THRESHOLD = _STATS_CONFIG['novelty_decay_threshold']

# A metric name suggesting a binary/proportion outcome (see
# suggest_model_from_metadata) is only an informative starting hypothesis.
# validate_model_from_data is authoritative and decides the final model —
# our simulated "D7 Retention Rate" values are calibrated continuous floats
# (baseline + effect + noise), not true 0/1 draws, and routinely fall
# outside [0, 1], so they validate to normal_normal despite the binary-
# sounding name.


class StatisticalEngine:
    """Runs frequentist statistical tests on validated experiments.

    GATE: experiments blocked by Module 2 validation (ers_label != 'READY')
    are NOT analyzed.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha

    def run_frequentist(self, control_data, treatment_data, primary_metric_col, experiment_id):
        """Welch's t-test (unequal variances) on control vs treatment.

        Args:
            control_data: DataFrame (assignment==0 rows)
            treatment_data: DataFrame (assignment==1 rows)
            primary_metric_col: str, the outcome column name to test
            experiment_id: str, for error reporting
        """
        control = control_data[primary_metric_col].dropna()
        treatment = treatment_data[primary_metric_col].dropna()

        if len(control) == 0 or len(treatment) == 0:
            raise ValueError(f"{experiment_id}: No data in control or treatment")

        control_mean = control.mean()
        treatment_mean = treatment.mean()
        absolute_lift = treatment_mean - control_mean
        relative_lift = absolute_lift / control_mean if control_mean != 0 else 0

        t_stat, p_value = ttest_ind(treatment, control, equal_var=False)

        control_std = control.std()
        treatment_std = treatment.std()

        if len(control) > 1 and len(treatment) > 1:
            pooled_std = np.sqrt(
                ((len(control) - 1) * control_std**2 + (len(treatment) - 1) * treatment_std**2)
                / (len(control) + len(treatment) - 2)
            )
        else:
            pooled_std = 0

        cohens_d = absolute_lift / pooled_std if pooled_std > 0 else 0

        # 95% CI via the t-distribution (scipy.stats.t), not a hardcoded 1.96
        df = len(control) + len(treatment) - 2
        se = pooled_std * np.sqrt(1 / len(control) + 1 / len(treatment))
        t_crit = t.ppf(0.975, df)
        ci_lower = absolute_lift - t_crit * se
        ci_upper = absolute_lift + t_crit * se

        return {
            "experiment_id": experiment_id,
            "test_used": "Welch's t-test",
            "metric_column": primary_metric_col,
            "statistic": t_stat,
            "p_value": p_value,
            "effect_size_cohens_d": cohens_d,
            "absolute_lift": absolute_lift,
            "relative_lift": relative_lift,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "control_n": len(control),
            "treatment_n": len(treatment),
            "significant_uncorrected": p_value < self.alpha,
            "status": "ANALYZED",
        }

    def run_all(self, experiment_results_df, validation_report_df, experiment_cards=None):
        """Run frequentist analysis on every experiment that passed validation.

        GATE: experiments with ers_label != 'READY' are blocked, not analyzed.

        Args:
            experiment_results_df: DataFrame from Module 1 (all experiments + outcomes)
            validation_report_df: DataFrame from Module 2 (validation results per experiment)
            experiment_cards: optional dict[str, ExperimentCard]. If provided, each
                experiment's outcome column is resolved via METRIC_COLUMN_MAP from
                its card's primary_metric label; otherwise DEFAULT_METRIC_COLUMN is used.

        Returns:
            dict of {experiment_id: result_dict}. Blocked experiments have
            status='BLOCKED (<ers_label>)' and null statistics.
        """
        validation_status = dict(zip(validation_report_df["experiment_id"], validation_report_df["ers_label"]))

        experiments = sorted(experiment_results_df["experiment_id"].unique())
        results = {}

        for exp_id in experiments:
            ers_label = validation_status.get(exp_id, "UNKNOWN")

            if ers_label != "READY":
                results[exp_id] = {
                    "experiment_id": exp_id,
                    "status": f"BLOCKED ({ers_label})",
                    "p_value": None,
                    "effect_size_cohens_d": None,
                    "absolute_lift": None,
                    "relative_lift": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "control_mean": None,
                    "treatment_mean": None,
                    "significant_uncorrected": None,
                    "p_bonferroni": None,
                    "significant_bonferroni": None,
                    "p_bh": None,
                    "significant_bh": None,
                    "blocking_reason": f"Failed validation: {ers_label}",
                }
                continue

            if experiment_cards is not None and exp_id in experiment_cards:
                metric_label = experiment_cards[exp_id].primary_metric
                metric_col = METRIC_COLUMN_MAP.get(metric_label, DEFAULT_METRIC_COLUMN)
            else:
                metric_col = DEFAULT_METRIC_COLUMN

            exp_data = experiment_results_df[experiment_results_df["experiment_id"] == exp_id]
            control = exp_data[exp_data["assignment"] == 0]
            treatment = exp_data[exp_data["assignment"] == 1]

            results[exp_id] = self.run_frequentist(control, treatment, metric_col, exp_id)

        analyzed_ids = [exp_id for exp_id in results if results[exp_id]["status"] == "ANALYZED"]

        if len(analyzed_ids) > 1:
            results = self._apply_multiple_testing_correction(results, analyzed_ids)

        return results

    def _apply_multiple_testing_correction(self, results_dict, analyzed_ids):
        """Benjamini-Hochberg FDR correction (statsmodels), Bonferroni shown for reference.

        Only corrects p-values among analyzed experiments — blocked experiments
        are excluded from the family entirely, since they never produced a p-value.
        """
        p_values = [results_dict[exp_id]["p_value"] for exp_id in analyzed_ids]

        reject, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method="fdr_bh")

        for idx, exp_id in enumerate(analyzed_ids):
            results_dict[exp_id]["p_bh"] = p_corrected[idx]
            results_dict[exp_id]["significant_bh"] = bool(reject[idx])

            p_bonf = min(p_values[idx] * len(analyzed_ids), 1.0)
            results_dict[exp_id]["p_bonferroni"] = p_bonf
            results_dict[exp_id]["significant_bonferroni"] = p_bonf < self.alpha

        return results_dict

    # ------------------------------------------------------------------
    # Bayesian layer (additive — does not touch the frequentist methods above)
    #
    # Model selection is a validation process, not a single lookup: metadata
    # gives an informative *suggestion* from the metric's name; the actual
    # data distribution is *authoritative* and decides the final model. Both
    # the suggested and final model, plus the reasoning between them, are
    # recorded as an audit trail (bayesian_model_suggested / _final /
    # _selection_reason) so the decision is reproducible and explainable.
    # ------------------------------------------------------------------

    def suggest_model_from_metadata(self, metric_name):
        """Suggest a Bayesian model from the metric's name (informative hint, not final).

        Returns (suggested_model, suggestion_reason), or (None, reason) if no
        metadata keyword matches.
        """
        name = metric_name.lower()

        if "retention" in name or "churn" in name:
            return "beta_binomial", "Metadata: retention metric (binary by name)"

        if "conversion" in name or "quit" in name:
            return "beta_binomial", "Metadata: outcome metric (binary by name)"

        if any(term in name for term in ["duration", "length", "count", "average", "score", "rate (percent)"]):
            return "normal_normal", "Metadata: continuous metric"

        return None, "No metadata match"

    def validate_model_from_data(self, control_data, treatment_data, primary_metric):
        """Validate a model choice against the actual data distribution (authoritative).

        Returns (validated_model, validation_reason, overrides_metadata).
        """
        sample = pd.concat([control_data[primary_metric], treatment_data[primary_metric]]).dropna()

        if len(sample) == 0:
            return "normal_normal", "No data available", False

        unique_count = sample.nunique()
        value_min = sample.min()
        value_max = sample.max()

        if unique_count == 2:
            return "beta_binomial", "Data: exactly 2 unique values (true binary)", False

        if unique_count >= 100 or value_min < -0.01 or value_max > 1.01:
            reason = (
                f"Data: continuous ({unique_count} unique values, "
                f"range [{value_min:.2f}, {value_max:.2f}])"
            )
            return "normal_normal", reason, True

        if unique_count > 2 and unique_count < 100:
            return "normal_normal", f"Data: small-count continuous ({unique_count} unique values)", False

        return "normal_normal", "Default to Normal-Normal", False

    def select_bayesian_model(self, metric_name, control_data, treatment_data, primary_metric):
        """Select the Bayesian model: suggest from metadata, validate against
        data, record the full decision path.

        Returns (final_model, suggested_model, selection_reason).
        """
        suggested_model, suggestion_reason = self.suggest_model_from_metadata(metric_name)

        validated_model, validation_reason, overrides_metadata = self.validate_model_from_data(
            control_data, treatment_data, primary_metric
        )

        if suggested_model is None:
            selection_reason = f"{suggestion_reason}. Data validation: {validation_reason}. Selected: {validated_model}."
        elif overrides_metadata and suggested_model != validated_model:
            selection_reason = (
                f"{suggestion_reason}. Data validation: {validation_reason}. "
                f"Selected: {validated_model} (overrides metadata suggestion)."
            )
        else:
            selection_reason = f"{suggestion_reason}. Data confirms: {validation_reason}."

        final_model = validated_model
        return final_model, suggested_model, selection_reason

    def compute_bayesian_posterior(
        self, control_mean, treatment_mean, control_n, treatment_n, control_std, treatment_std, model_type,
    ):
        """Compute a Bayesian posterior using the given model.

        beta_binomial: weak Beta(1,1) prior on each arm's proportion, MC
            sampling (deterministic, seed=RANDOM_SEED, MC_SAMPLES draws).
        normal_normal: conjugate approximation for continuous metrics, using
            each arm's REAL observed sample std (not a guessed Bernoulli-style
            std) — closed-form, no sampling needed, exactly reproducible.
            Documented as an approximation, not a full hierarchical Bayesian
            model — see docs/Tradeoffs.md: pragmatic over academically maximal.

        Returns: (prob_positive, ci_lower, ci_upper, prior_description)
        """
        if model_type == "beta_binomial":
            prior_desc = "Beta(1,1)"

            control_posterior = beta(1 + control_mean * control_n, 1 + (1 - control_mean) * control_n)
            treatment_posterior = beta(1 + treatment_mean * treatment_n, 1 + (1 - treatment_mean) * treatment_n)

            rng = np.random.default_rng(RANDOM_SEED)
            control_samples = control_posterior.rvs(MC_SAMPLES, random_state=rng)
            treatment_samples = treatment_posterior.rvs(MC_SAMPLES, random_state=rng)

            prob_positive = np.mean(treatment_samples > control_samples)

            effect_samples = treatment_samples - control_samples
            ci_lower = np.percentile(effect_samples, 2.5)
            ci_upper = np.percentile(effect_samples, 97.5)

        else:  # normal_normal
            prior_desc = "Normal(0, inf) [improper, scale-invariant]"

            effect_mean = treatment_mean - control_mean
            effect_se = np.sqrt((control_std**2 / control_n) + (treatment_std**2 / treatment_n))
            effect_se = max(effect_se, 1e-6)  # avoid degenerate zero-variance edge case

            prob_positive = 1 - norm.cdf(0, loc=effect_mean, scale=effect_se)

            ci_lower = effect_mean - 1.96 * effect_se
            ci_upper = effect_mean + 1.96 * effect_se

        return prob_positive, ci_lower, ci_upper, prior_desc

    def enrich_with_bayesian(self, results, experiment_results_df, experiment_cards_df, primary_metric="primary_metric_value"):
        """Add a Bayesian layer (with full model-selection audit trail) to an
        existing Module 3 results dict, in place.

        Does not re-run any frequentist test — only reads control/treatment
        subsets from experiment_results_df (the same raw data Module 3 already
        used) to get each arm's real mean/std/n for the Bayesian computation.

        Args:
            experiment_cards_df: DataFrame with at least 'experiment_id' and
                'primary_metric' columns (e.g. data/simulation/experiment_cards.csv),
                used to resolve each experiment's real metric LABEL for the
                metadata suggestion step (the data column name alone, e.g.
                'primary_metric_value', carries no semantic information).

        Returns the same results dict, enriched with bayesian_* keys.
        """
        for exp_id in sorted(experiment_results_df["experiment_id"].unique()):
            if exp_id not in results:
                continue

            if results[exp_id]["status"] != "ANALYZED":
                results[exp_id]["bayesian_prob_positive"] = None
                results[exp_id]["bayesian_ci_lower"] = None
                results[exp_id]["bayesian_ci_upper"] = None
                results[exp_id]["bayesian_model_suggested"] = "N/A"
                results[exp_id]["bayesian_model_final"] = "N/A"
                results[exp_id]["bayesian_model_selection_reason"] = "Experiment blocked"
                results[exp_id]["bayesian_prior"] = "N/A"
                results[exp_id]["significant_bayesian"] = None
                continue

            exp_data = experiment_results_df[experiment_results_df["experiment_id"] == exp_id]
            control = exp_data[exp_data["assignment"] == 0]
            treatment = exp_data[exp_data["assignment"] == 1]

            card_row = experiment_cards_df[experiment_cards_df["experiment_id"] == exp_id]
            if len(card_row) > 0:
                metric_name = card_row.iloc[0].get("primary_metric", primary_metric)
            else:
                metric_name = primary_metric

            final_model, suggested_model, selection_reason = self.select_bayesian_model(
                metric_name, control, treatment, primary_metric
            )

            control_mean = results[exp_id]["control_mean"]
            treatment_mean = results[exp_id]["treatment_mean"]
            control_n = results[exp_id]["control_n"]
            treatment_n = results[exp_id]["treatment_n"]
            control_std = control[primary_metric].std()
            treatment_std = treatment[primary_metric].std()

            prob_positive, ci_lower, ci_upper, prior_desc = self.compute_bayesian_posterior(
                control_mean, treatment_mean, control_n, treatment_n, control_std, treatment_std, final_model,
            )

            results[exp_id]["bayesian_prob_positive"] = prob_positive
            results[exp_id]["bayesian_ci_lower"] = ci_lower
            results[exp_id]["bayesian_ci_upper"] = ci_upper
            results[exp_id]["bayesian_model_suggested"] = suggested_model if suggested_model is not None else "N/A"
            results[exp_id]["bayesian_model_final"] = final_model
            results[exp_id]["bayesian_model_selection_reason"] = selection_reason
            results[exp_id]["bayesian_prior"] = prior_desc
            results[exp_id]["significant_bayesian"] = prob_positive > BAYES_SIGNIFICANT_THRESHOLD

        return results

    # ------------------------------------------------------------------
    # Novelty detection — early (D1-7) vs late (D8-14) period comparison.
    # Per docs/TDD.md's original Module 3 spec ("Novelty: Early period vs
    # late period. Detects when effects disappear"); added here since it
    # was missing from the initial implementation.
    # ------------------------------------------------------------------

    def detect_novelty(self, early_lift, late_lift, decay_threshold=NOVELTY_DECAY_THRESHOLD):
        """Detect a novelty effect: an early-period lift that doesn't persist.

        decay_ratio = late_lift / early_lift — the fraction of the early
        effect still present in the late period. novelty_detected if the
        early effect was positive and decays below decay_threshold. This
        formulation (rather than the early/late ratio used descriptively in
        docs/Simulation_Environment.md) handles the case where the late
        effect reverses sign — EXP-005's late lift is negative, so a plain
        ratio (early/late) would be negative and not cleanly comparable to
        the ">2.0" framing; decay_ratio < 0.5 catches reversal naturally,
        since a negative decay_ratio is always < 0.5.

        Returns: dict with decay_ratio, novelty_detected, early_lift, late_lift
        """
        if early_lift <= 0:
            # No early effect to decay from — novelty isn't a meaningful concept here.
            return {
                "decay_ratio": None,
                "novelty_detected": False,
                "early_lift": early_lift,
                "late_lift": late_lift,
            }

        decay_ratio = late_lift / early_lift
        novelty_detected = decay_ratio < decay_threshold

        return {
            "decay_ratio": decay_ratio,
            "novelty_detected": novelty_detected,
            "early_lift": early_lift,
            "late_lift": late_lift,
        }

    def enrich_with_novelty(self, results, experiment_results_df):
        """Add novelty detection to an existing Module 3 results dict, in place.

        Does not re-run any frequentist test — only computes early/late
        period MEANS per arm directly from experiment_results_df (the same
        raw data Module 1 generated and Module 3 already reads), not a
        statistical test. This is the kind of computation Module 3 is meant
        to own; Module 5 (Decision Engine) only ever reads the resulting
        novelty_detected flag, never computes it.

        Returns the same results dict, enriched with novelty_* keys.
        """
        for exp_id in sorted(experiment_results_df["experiment_id"].unique()):
            if exp_id not in results:
                continue

            if results[exp_id]["status"] != "ANALYZED":
                results[exp_id]["novelty_decay_ratio"] = None
                results[exp_id]["novelty_detected"] = None
                results[exp_id]["novelty_early_lift"] = None
                results[exp_id]["novelty_late_lift"] = None
                continue

            exp_data = experiment_results_df[experiment_results_df["experiment_id"] == exp_id]
            control = exp_data[exp_data["assignment"] == 0]
            treatment = exp_data[exp_data["assignment"] == 1]

            early_lift = treatment["early_period_outcome"].mean() - control["early_period_outcome"].mean()
            late_lift = treatment["late_period_outcome"].mean() - control["late_period_outcome"].mean()

            novelty_result = self.detect_novelty(early_lift, late_lift)

            results[exp_id]["novelty_decay_ratio"] = novelty_result["decay_ratio"]
            results[exp_id]["novelty_detected"] = novelty_result["novelty_detected"]
            results[exp_id]["novelty_early_lift"] = early_lift
            results[exp_id]["novelty_late_lift"] = late_lift

        return results
