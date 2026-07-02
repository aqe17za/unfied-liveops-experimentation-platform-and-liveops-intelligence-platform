"""Module 4 — Heterogeneous Treatment Effects (HTE).

Answers "who did it work for?" by segmenting treatment effects within each
experiment that passed Module 3 (i.e. status == 'ANALYZED'; EXP-001 stays
excluded, same SRM hard gate as Module 3).

PRIMARY segmentation: PXI Tier — real, from Project 1's player features.
SECONDARY (exploratory): Player Type, Play Style — synthetic, since Project 1
has no tenure or game-mode-preference field (see src/experiment_manager.py).

Segment values are read dynamically from the data (no hardcoded tier names),
per the platform's segmentation columns. The Critical PXI tier is only ~0.2%
of Project 1's real population (39 of 19,758 players), so most experiments
sample only 0-3 Critical-tier players — far too few for a reliable t-test.
This module reports that honestly as INSUFFICIENT DATA rather than
fabricating a result.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_ind
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CONFIG_PATH, 'r') as _f:
    _HTE_CONFIG = yaml.safe_load(_f)['experimentation']['hte']

DEFAULT_ALPHA = _HTE_CONFIG['alpha']
DEFAULT_MDE = _HTE_CONFIG['mde']


class HeterogeneousEffects:
    """Analyzes treatment effects across player segments.

    PRIMARY: PXI Tier (real, from Project 1 features).
    SECONDARY: Player Type, Play Style (synthetic, exploratory).
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, mde: float = DEFAULT_MDE):
        """
        Args:
            alpha: significance level (0.05)
            mde: minimum detectable (practical) effect, absolute (0.02 = 2pp),
                matching every experiment card's "D7 Retention +2%" success criterion.
        """
        self.alpha = alpha
        self.mde = mde

    def _analyze_segment(self, segment_data, primary_metric):
        """Analyze the treatment effect within a single segment.

        Returns a dict of stats plus a rich, reasoned recommendation — not a
        single significance threshold.
        """
        control = segment_data[segment_data["assignment"] == 0][primary_metric].dropna()
        treatment = segment_data[segment_data["assignment"] == 1][primary_metric].dropna()

        n_ctrl = len(control)
        n_trt = len(treatment)

        if n_ctrl < 2 or n_trt < 2:
            return {
                "n_control": n_ctrl,
                "n_treatment": n_trt,
                "control_mean": None,
                "treatment_mean": None,
                "absolute_lift": None,
                "relative_lift": None,
                "effect_size_cohens_d": None,
                "p_value": None,
                "t_statistic": None,
                "significant": None,
                "practical": None,
                "ci_lower": None,
                "ci_upper": None,
                "recommendation": "INSUFFICIENT DATA",
                "reasoning": f"n_control={n_ctrl}, n_treatment={n_trt} (minimum 2 each)",
            }

        control_mean = control.mean()
        treatment_mean = treatment.mean()
        absolute_lift = treatment_mean - control_mean
        relative_lift = absolute_lift / control_mean if control_mean != 0 else 0

        t_stat, p_value = ttest_ind(treatment, control, equal_var=False)

        control_std = control.std()
        treatment_std = treatment.std()

        if n_ctrl > 1 and n_trt > 1:
            pooled_std = np.sqrt(
                ((n_ctrl - 1) * control_std**2 + (n_trt - 1) * treatment_std**2) / (n_ctrl + n_trt - 2)
            )
        else:
            pooled_std = 0

        cohens_d = absolute_lift / pooled_std if pooled_std > 0 else 0

        df = n_ctrl + n_trt - 2
        t_crit = t_dist.ppf(0.975, df)
        se = pooled_std * np.sqrt(1 / n_ctrl + 1 / n_trt) if pooled_std > 0 else 0
        ci_lower = absolute_lift - t_crit * se
        ci_upper = absolute_lift + t_crit * se

        significant = p_value < self.alpha
        practical = abs(absolute_lift) > self.mde
        ci_positive = ci_lower > 0  # CI doesn't cross zero

        if not significant:
            recommendation = "EXCLUDE"
            reasoning = f"Not significant (p={p_value:.4f})"
        elif significant and not practical:
            recommendation = "MONITOR"
            reasoning = f"Significant but lift {relative_lift * 100:.2f}% < MDE {self.mde * 100:.0f}%"
        elif significant and practical and absolute_lift > 0 and ci_positive:
            recommendation = "INCLUDE"
            reasoning = f"Significant, practical, and CI positive (CI=[{ci_lower:.4f}, {ci_upper:.4f}])"
        elif significant and practical and absolute_lift > 0 and not ci_positive:
            recommendation = "MONITOR"
            reasoning = f"Significant and practical but CI crosses zero (CI=[{ci_lower:.4f}, {ci_upper:.4f}])"
        elif significant and absolute_lift < 0:
            recommendation = "EXCLUDE"
            reasoning = f"Significant negative effect ({relative_lift * 100:.2f}%)"
        else:
            recommendation = "EXCLUDE"
            reasoning = "Negative or ambiguous effect"

        return {
            "n_control": n_ctrl,
            "n_treatment": n_trt,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "absolute_lift": absolute_lift,
            "relative_lift": relative_lift,
            "effect_size_cohens_d": cohens_d,
            "p_value": p_value,
            "t_statistic": t_stat,
            "significant": significant,
            "practical": practical,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "recommendation": recommendation,
            "reasoning": reasoning,
        }

    def segment_by_dimension(self, experiment_data, primary_metric, dimension_column, dimension_name):
        """Generic segmentation by any dimension. Segment values are read
        dynamically from the data — no hardcoded tier/segment names."""
        segment_values = sorted(experiment_data[dimension_column].unique())

        results = []
        for segment_value in segment_values:
            segment_data = experiment_data[experiment_data[dimension_column] == segment_value]
            segment_result = self._analyze_segment(segment_data, primary_metric)
            segment_result["dimension"] = dimension_name
            segment_result["segment_value"] = segment_value
            results.append(segment_result)

        return pd.DataFrame(results)

    def run_all(self, experiment_results_df, statistical_results_df, primary_metric="primary_metric_value"):
        """Run HTE analysis on every experiment that Module 3 analyzed.

        Returns dict of {experiment_id: {dimension_name: segment_results_df}}.
        """
        analyzed_exps = statistical_results_df[statistical_results_df["status"] == "ANALYZED"][
            "experiment_id"
        ].unique()

        all_hte = {}
        for exp_id in sorted(analyzed_exps):
            exp_data = experiment_results_df[experiment_results_df["experiment_id"] == exp_id]

            hte_by_dim = {
                "PXI Tier": self.segment_by_dimension(exp_data, primary_metric, "segment_pxi_tier", "PXI Tier"),
                "Player Type": self.segment_by_dimension(
                    exp_data, primary_metric, "segment_player_type", "Player Type"
                ),
                "Play Style": self.segment_by_dimension(
                    exp_data, primary_metric, "segment_play_style", "Play Style"
                ),
            }
            all_hte[exp_id] = hte_by_dim

        return all_hte

    def generate_rollout_summary(self, all_hte_results):
        """One row per experiment: best/worst PXI-tier segment and rollout strategy."""
        summaries = []

        for exp_id in sorted(all_hte_results.keys()):
            pxi_results = all_hte_results[exp_id]["PXI Tier"]

            if len(pxi_results) == 0:
                continue

            pxi_valid = pxi_results[pxi_results["absolute_lift"].notna()].copy()

            if len(pxi_valid) == 0:
                summaries.append({
                    "experiment_id": exp_id,
                    "best_segment": None,
                    "best_lift": None,
                    "worst_segment": None,
                    "worst_lift": None,
                    "lift_range": None,
                    "hte_detected": False,
                    "rollout_strategy": "NO SEGMENTS WITH SUFFICIENT DATA",
                })
                continue

            best_idx = pxi_valid["absolute_lift"].idxmax()
            worst_idx = pxi_valid["absolute_lift"].idxmin()
            best_seg = pxi_valid.loc[best_idx]
            worst_seg = pxi_valid.loc[worst_idx]

            include_segments = pxi_results[pxi_results["recommendation"] == "INCLUDE"]["segment_value"].tolist()
            exclude_segments = pxi_results[pxi_results["recommendation"] == "EXCLUDE"]["segment_value"].tolist()

            if len(include_segments) > 0:
                rollout_strategy = f"INCLUDE: {', '.join(include_segments)}"
            elif len(exclude_segments) == len(pxi_results):
                rollout_strategy = "NO SEGMENTS RECOMMENDED"
            else:
                rollout_strategy = "MIXED / MONITOR"

            summaries.append({
                "experiment_id": exp_id,
                "best_segment": best_seg["segment_value"],
                "best_lift": best_seg["absolute_lift"],
                "worst_segment": worst_seg["segment_value"],
                "worst_lift": worst_seg["absolute_lift"],
                "lift_range": best_seg["absolute_lift"] - worst_seg["absolute_lift"],
                "hte_detected": abs(best_seg["absolute_lift"] - worst_seg["absolute_lift"]) > 0.01,
                "rollout_strategy": rollout_strategy,
            })

        return pd.DataFrame(summaries)
