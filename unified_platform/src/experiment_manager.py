"""Module 1 — Experiment Manager.

Loads Project 1's real player population, creates the 6 experiment cards
specified in docs/Experiment_Design.md, assigns control/treatment, and
simulates deterministic outcomes (seed=42) per docs/Simulation_Environment.md.

Real vs synthetic, stated plainly:
- player_id, pxi_score, pxi_tier, losing_streak, matches_this_week,
  ragequit_rate_historical, win_rate_last10, recency_days: REAL (Project 1).
- EXP-001's target population (At Risk tier) and EXP-003's per-tier effect
  sizes are applied using each sampled player's REAL pxi_tier.
- segment_player_type (New/Veteran) and segment_play_style (Casual/Ranked):
  SYNTHETIC. Project 1 has no tenure or game-mode-preference field, so these
  are deterministic (seeded) 50/50 splits, used only so Module 4 (HTE) has
  something to segment on for those two dimensions.
- primary_metric_value / early_period_outcome / late_period_outcome /
  guardrail_ragequit / guardrail_completion: SYNTHETIC, by design — these are
  the controlled simulation outcomes described in docs/Simulation_Environment.md,
  built as baseline + effect + noise so that ground truth is known.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from connectors.project1_loader import load_players_from_project1, load_interventions_from_project1
from connectors.duckdb_loader import save_to_duckdb
from connectors.csv_loader import save_to_csv
from analytics_schema.schema import SCHEMA_DB, init_schema

with open(CONFIG_PATH, 'r') as f:
    _CONFIG = yaml.safe_load(f)['experimentation']

RNG_SEED = _CONFIG['simulation']['rng_seed']

# Real D7 Return Rate by PXI tier, from Project 1's north_star_metrics
# (docs/Metric_Definitions.md Section 2, "North-Star Metrics").
BASELINE_AT_RISK = _CONFIG['simulation']['baselines']['at_risk']
BASELINE_ALL = _CONFIG['simulation']['baselines']['all']  # approximated by the Stable-tier D7 return rate

# Project 2's experiment-level ragequit guardrail baseline (distinct from
# Project 1's match-level ragequit proxy, which is ~1.5% of matches).
# This baseline and the EXP-006 effect are fixed values from
# docs/Experiment_Design.md (EXP-006: baseline 18%, treatment 24.2%).
BASELINE_RAGEQUIT = _CONFIG['simulation']['baselines']['ragequit']
BASELINE_COMPLETION = _CONFIG['simulation']['baselines']['completion']

# Per-tier effect sizes for EXP-003 (docs/Experiment_Design.md / Simulation_Environment.md)
EXP003_TIER_EFFECTS = _CONFIG['simulation']['exp003_tier_effects']

SIMULATION_DIR = ROOT / "data" / "simulation"

# The 6 experiment specs, verbatim from docs/Experiment_Design.md.
EXPERIMENT_SPECS = [
    dict(
        experiment_id="EXP-001",
        intervention_id="INT-01",
        feature_name="Relaxed Matchmaking",
        hypothesis="Relaxed skill-based matchmaking -> increased D7 retention for At Risk players",
        primary_metric="D7 Retention Rate",
        guardrail_metrics="Ragequit Rate, Match Completion Rate, Session Length",
        success_criteria="D7 Retention +2% (absolute), no increase in ragequit",
        rollback_criteria="Ragequit > baseline + 5%, Match Completion < baseline - 3%",
        engineering_cost="High",
        expected_impact="High",
        target_segment="At Risk",
        sample_size=1000,
        control_split=0.457,  # SRM scenario: 457/543 split. Chosen (not literal 46/54)
        # so chi-square SRM test reliably trips the p<0.01 hard gate: 460/540 gives
        # p=0.0114 (passes!), 457/543 gives p=0.0065, matching docs/Experiment_Design.md.
        baseline=BASELINE_AT_RISK,
        effect=0.008,
        early_effect=0.008,
        late_effect=0.008,
        ragequit_effect=0.0,
        noise_std=0.07490,  # calibrated so an (unused, SRM-gated) t-test would land ~p=0.30
    ),
    dict(
        experiment_id="EXP-002",
        intervention_id="INT-02",
        feature_name="XP Boost Notification",
        hypothesis="In-game XP boost + notification -> increased session frequency",
        primary_metric="D7 Retention Rate",
        guardrail_metrics="Ragequit Rate, Match Completion Rate, Session Length",
        success_criteria="D7 Retention +2% (absolute)",
        rollback_criteria="Ragequit > baseline + 5%",
        engineering_cost="Low",
        expected_impact="Medium",
        target_segment="All",
        sample_size=1000,
        control_split=0.5,
        baseline=BASELINE_ALL,
        effect=0.031,
        early_effect=0.031,
        late_effect=0.031,
        ragequit_effect=0.0,
        noise_std=0.20258,  # calibrated so t-test lands at p=0.021, per docs/Experiment_Design.md
    ),
    dict(
        experiment_id="EXP-003",
        intervention_id="INT-03",
        feature_name="Mode Switch Prompt",
        hypothesis="Prompt players to try different modes -> increase engagement variety",
        primary_metric="D7 Retention Rate",
        guardrail_metrics="Ragequit Rate, Match Completion Rate",
        success_criteria="D7 Retention +2% overall (or significant in subset)",
        rollback_criteria="Ragequit > baseline + 5%",
        engineering_cost="Low",
        expected_impact="Medium",
        target_segment="All",
        sample_size=1000,
        control_split=0.5,
        baseline=BASELINE_ALL,
        effect=None,  # per-tier, see EXP003_TIER_EFFECTS
        early_effect=None,
        late_effect=None,
        ragequit_effect=0.0,
        noise_std=0.07752,  # calibrated so overall (pooled) t-test lands at p=0.067
    ),
    dict(
        experiment_id="EXP-004",
        intervention_id="INT-04",
        feature_name="Weekend Event Invite",
        hypothesis="Targeted weekend event invites -> increase weekend engagement and D7 retention",
        primary_metric="D7 Retention Rate",
        guardrail_metrics="Ragequit Rate, Match Completion Rate, Session Length",
        success_criteria="D7 Retention +2% (absolute)",
        rollback_criteria="Ragequit > baseline + 5%",
        engineering_cost="Medium",
        expected_impact="High",
        target_segment="Casual",
        sample_size=1000,
        control_split=0.5,
        baseline=BASELINE_ALL,
        effect=0.028,
        early_effect=0.029,
        late_effect=0.027,
        ragequit_effect=0.0,
        noise_std=0.41707,  # calibrated so t-test lands at p=0.034
    ),
    dict(
        experiment_id="EXP-005",
        intervention_id="INT-05",
        feature_name="Reward Progression Highlight",
        hypothesis="Highlight reward progression -> increase engagement and D7 retention",
        primary_metric="D7 Retention Rate",
        guardrail_metrics="Ragequit Rate, Match Completion Rate",
        success_criteria="D7 Retention +2% sustained",
        rollback_criteria="Ragequit > baseline + 5% OR effect decays > 80% after Week 1",
        engineering_cost="Low",
        expected_impact="Medium",
        target_segment="All",
        sample_size=1000,
        control_split=0.5,
        baseline=BASELINE_ALL,
        effect=0.0305,  # blended early/late, used for the whole-window primary metric
        early_effect=0.058,
        late_effect=0.003,
        ragequit_effect=0.0,
        noise_std=0.35477,  # calibrated so overall t-test lands at p=0.019
    ),
    dict(
        experiment_id="EXP-006",
        intervention_id="INT-06",
        feature_name="Friend Match Suggestion",
        hypothesis="Suggest matches with friends -> increase social engagement and D7 retention",
        primary_metric="D7 Retention Rate",
        guardrail_metrics="Ragequit Rate, Match Completion Rate, Session Length",
        success_criteria="D7 Retention +2%, no ragequit increase",
        rollback_criteria="Ragequit > baseline + 5% [WILL BE TRIGGERED]",
        engineering_cost="Medium",
        expected_impact="High",
        target_segment="All",
        sample_size=1000,
        control_split=0.5,
        baseline=BASELINE_ALL,
        effect=0.022,
        early_effect=0.022,
        late_effect=0.022,
        ragequit_effect=0.062,
        noise_std=0.20595,  # calibrated so t-test lands at p=0.041
    ),
]


class ExperimentCard:
    def __init__(self, spec: dict):
        self.experiment_id = spec["experiment_id"]
        self.intervention_id = spec["intervention_id"]
        self.feature_name = spec["feature_name"]
        self.hypothesis = spec["hypothesis"]
        self.primary_metric = spec["primary_metric"]
        self.guardrail_metrics = spec["guardrail_metrics"]
        self.success_criteria = spec["success_criteria"]
        self.rollback_criteria = spec["rollback_criteria"]
        self.engineering_cost = spec["engineering_cost"]
        self.expected_impact = spec["expected_impact"]
        self.target_segment = spec["target_segment"]
        self.sample_size = spec["sample_size"]
        self.control_split = spec["control_split"]
        self.control_size = int(round(spec["sample_size"] * spec["control_split"]))
        self.treatment_size = spec["sample_size"] - self.control_size
        self.status = "Running"
        self.spec = spec


class ExperimentManager:
    def __init__(self):
        self.players = load_players_from_project1()
        self.interventions = load_interventions_from_project1()
        self.experiments: list[ExperimentCard] = []
        self.experiment_results: pd.DataFrame | None = None
        self.rng = np.random.default_rng(RNG_SEED)

    def create_experiment_cards(self) -> list[ExperimentCard]:
        self.experiments = [ExperimentCard(spec) for spec in EXPERIMENT_SPECS]
        print(f"Created {len(self.experiments)} experiment cards")
        return self.experiments

    def _sample_players(self, experiment: ExperimentCard) -> pd.DataFrame:
        """Sample real players for this experiment from the appropriate population."""
        if experiment.target_segment == "At Risk":
            pool = self.players[self.players["pxi_tier"] == "At Risk"]
        else:
            pool = self.players

        idx = self.rng.choice(len(pool), size=experiment.sample_size, replace=False)
        return pool.iloc[idx].reset_index(drop=True).copy()

    def assign_treatment(self, experiment: ExperimentCard) -> pd.DataFrame:
        """Assign sampled players to control(0)/treatment(1) with seeded randomization.

        Uses an exact-count shuffled array (not a binomial draw) so the
        realized split always matches control_size/treatment_size exactly —
        e.g. EXP-001's 46/54 SRM scenario is a precise fixed imbalance, not
        noisy variation around 46/54."""
        sampled = self._sample_players(experiment)
        assignment = np.concatenate([
            np.zeros(experiment.control_size, dtype=int),
            np.ones(experiment.treatment_size, dtype=int),
        ])
        self.rng.shuffle(assignment)
        sampled["assignment"] = assignment
        return sampled

    def simulate_outcomes(self, experiment: ExperimentCard, players_df: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic outcome = baseline + effect + noise, per
        docs/Simulation_Environment.md. EXP-003 applies a per-player effect
        based on the player's REAL pxi_tier (heterogeneous treatment effect)."""
        spec = experiment.spec
        df = players_df.copy()
        n = len(df)
        control_mask = df["assignment"] == 0
        treatment_mask = df["assignment"] == 1

        # noise_std is calibrated per experiment (see EXPERIMENT_SPECS) so the
        # resulting t-test p-value matches the documented target in
        # docs/Experiment_Design.md, rather than a flat value that would make
        # every experiment absurdly significant at n~500/arm.
        noise_std = spec["noise_std"]
        noise = self.rng.normal(0, noise_std, n)
        early_noise = self.rng.normal(0, noise_std, n)
        late_noise = self.rng.normal(0, noise_std, n)

        if experiment.experiment_id == "EXP-003":
            effect = df["pxi_tier"].map(EXP003_TIER_EFFECTS).fillna(0.0).to_numpy()
            early_effect = effect
            late_effect = effect
        else:
            effect = np.full(n, spec["effect"])
            early_effect = np.full(n, spec["early_effect"])
            late_effect = np.full(n, spec["late_effect"])

        baseline = spec["baseline"]

        df["primary_metric_value"] = baseline + noise
        df.loc[treatment_mask, "primary_metric_value"] = baseline + effect[treatment_mask] + noise[treatment_mask]

        df["early_period_outcome"] = baseline + early_noise
        df.loc[treatment_mask, "early_period_outcome"] = baseline + early_effect[treatment_mask] + early_noise[treatment_mask]

        df["late_period_outcome"] = baseline + late_noise
        df.loc[treatment_mask, "late_period_outcome"] = baseline + late_effect[treatment_mask] + late_noise[treatment_mask]

        # Guardrails: ragequit rate and match completion rate
        ragequit_noise = self.rng.normal(0, 0.02, n)
        completion_noise = self.rng.normal(0, 0.02, n)

        df["guardrail_ragequit"] = BASELINE_RAGEQUIT + ragequit_noise
        df.loc[treatment_mask, "guardrail_ragequit"] = (
            BASELINE_RAGEQUIT + spec["ragequit_effect"] + ragequit_noise[treatment_mask]
        )

        df["guardrail_completion"] = BASELINE_COMPLETION + completion_noise
        df.loc[treatment_mask, "guardrail_completion"] = BASELINE_COMPLETION + completion_noise[treatment_mask]

        # Segments: pxi_tier is REAL; player_type and play_style are SYNTHETIC
        # (Project 1 has no tenure or mode-preference field).
        df["segment_pxi_tier"] = df["pxi_tier"]
        df["segment_player_type"] = np.where(self.rng.random(n) < 0.5, "New", "Veteran")
        df["segment_play_style"] = np.where(self.rng.random(n) < 0.5, "Casual", "Ranked")

        df["experiment_id"] = experiment.experiment_id
        return df

    def run_all_experiments(self) -> pd.DataFrame:
        all_results = []
        for experiment in self.experiments:
            print(f"\nRunning {experiment.experiment_id}: {experiment.feature_name}")
            sampled = self.assign_treatment(experiment)
            simulated = self.simulate_outcomes(experiment, sampled)
            all_results.append(simulated)

            ctrl_mean = simulated.loc[simulated["assignment"] == 0, "primary_metric_value"].mean()
            trt_mean = simulated.loc[simulated["assignment"] == 1, "primary_metric_value"].mean()
            print(f"   Control: {experiment.control_size} | Treatment: {experiment.treatment_size}")
            print(f"   Control avg outcome: {ctrl_mean:.4f} | Treatment avg outcome: {trt_mean:.4f}")

        self.experiment_results = pd.concat(all_results, ignore_index=True)
        print(f"\nSimulated {len(self.experiment_results)} experiment outcomes across {len(self.experiments)} experiments")
        return self.experiment_results

    def validate(self, log_path: Path) -> None:
        lines = []

        def log(msg):
            print(msg)
            lines.append(msg)

        log("=" * 60)
        log("VALIDATION")
        log("=" * 60)

        expected_total = sum(e.sample_size for e in self.experiments)
        assert len(self.experiment_results) == expected_total, (
            f"Expected {expected_total} rows, got {len(self.experiment_results)}"
        )
        log(f"Total rows: {len(self.experiment_results)} ({len(self.experiments)} experiments)")

        for experiment in self.experiments:
            exp_data = self.experiment_results[self.experiment_results["experiment_id"] == experiment.experiment_id]
            control_count = int((exp_data["assignment"] == 0).sum())
            treatment_count = int((exp_data["assignment"] == 1).sum())

            assert control_count == experiment.control_size, (
                f"{experiment.experiment_id}: expected {experiment.control_size} control, got {control_count}"
            )
            assert treatment_count == experiment.treatment_size, (
                f"{experiment.experiment_id}: expected {experiment.treatment_size} treatment, got {treatment_count}"
            )
            log(f"{experiment.experiment_id}: Control={control_count}, Treatment={treatment_count}")

        for experiment in self.experiments:
            exp_data = self.experiment_results[self.experiment_results["experiment_id"] == experiment.experiment_id]
            ctrl_mean = exp_data.loc[exp_data["assignment"] == 0, "primary_metric_value"].mean()
            trt_mean = exp_data.loc[exp_data["assignment"] == 1, "primary_metric_value"].mean()
            lift = trt_mean - ctrl_mean
            log(f"   {experiment.experiment_id}: Control={ctrl_mean:.4f}, Treatment={trt_mean:.4f}, Lift={lift:+.4f}")

        log("\nAll validations passed")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines), encoding="utf-8")

    def save_outputs(self):
        cards_data = []
        for exp in self.experiments:
            cards_data.append({
                "experiment_id": exp.experiment_id,
                "intervention_id": exp.intervention_id,
                "feature_name": exp.feature_name,
                "hypothesis": exp.hypothesis,
                "primary_metric": exp.primary_metric,
                "guardrail_metrics": exp.guardrail_metrics,
                "success_criteria": exp.success_criteria,
                "rollback_criteria": exp.rollback_criteria,
                "engineering_cost": exp.engineering_cost,
                "expected_product_impact": exp.expected_impact,
                "target_segment": exp.target_segment,
                "sample_size": exp.sample_size,
                "control_size": exp.control_size,
                "treatment_size": exp.treatment_size,
                "status": exp.status,
            })
        cards_df = pd.DataFrame(cards_data)
        save_to_csv(SIMULATION_DIR / "experiment_cards.csv", cards_df)
        save_to_csv(SIMULATION_DIR / "experiment_results.csv", self.experiment_results)

        save_to_duckdb(str(SCHEMA_DB), "experiments", cards_df)
        save_to_duckdb(str(SCHEMA_DB), "experiment_results", self.experiment_results)
        save_to_duckdb(str(SCHEMA_DB), "players", self.players)
        save_to_duckdb(str(SCHEMA_DB), "interventions", self.interventions)

        print("\n" + "=" * 60)
        print("OUTPUT FILES")
        print("=" * 60)
        print(SIMULATION_DIR / "experiment_cards.csv")
        print(SIMULATION_DIR / "experiment_results.csv")
        print(f"{SCHEMA_DB} (players, interventions, experiments, experiment_results)")


def main():
    print("=" * 60)
    print("PHASE 1: PROJECT 1 CONNECTOR + EXPERIMENT MANAGER")
    print("=" * 60)

    init_schema()

    manager = ExperimentManager()
    manager.create_experiment_cards()
    manager.run_all_experiments()
    manager.validate(SIMULATION_DIR / "validation_log.txt")
    manager.save_outputs()

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("\nNext: Phase 2 - Experiment Validator (SRM + ERS)")


if __name__ == "__main__":
    main()
