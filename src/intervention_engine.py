"""Intervention engine: Confidence-scored policy engine ranking LiveOps actions per player."""

import logging
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ----------------------------------------------------------------------
# Intervention definitions
# ----------------------------------------------------------------------

@dataclass
class Intervention:
    id: str
    name: str
    description: str
    applicable_tiers: List[str]
    primary_trigger: str
    expected_d7_lift_range: tuple
    experiment_metric: str
    guardrail_metric: str


INT_01 = Intervention(
    id="INT-01",
    name="Relaxed Matchmaking",
    description=(
        "Reduce skill gap threshold by 15% for the "
        "next match to improve perceived fairness"
    ),
    applicable_tiers=["Critical", "At Risk"],
    primary_trigger="skill_gap",
    expected_d7_lift_range=(0.06, 0.12),
    experiment_metric="D7 Retention Rate",
    guardrail_metric="Average MQI (must stay above 50)"
)

INT_02 = Intervention(
    id="INT-02",
    name="XP Boost Notification",
    description=(
        "Send in-game notification of 2x XP "
        "for the next session"
    ),
    applicable_tiers=["At Risk", "Stable"],
    primary_trigger="engagement_trend",
    expected_d7_lift_range=(0.04, 0.09),
    experiment_metric="D7 Retention Rate",
    guardrail_metric="Session Length (must not decrease)"
)

INT_03 = Intervention(
    id="INT-03",
    name="Mode Switch Prompt",
    description=(
        "Suggest Squad Battles or casual mode "
        "after 3+ consecutive ranked losses"
    ),
    applicable_tiers=["Critical", "At Risk"],
    primary_trigger="losing_streak",
    expected_d7_lift_range=(0.05, 0.10),
    experiment_metric="D7 Retention Rate",
    guardrail_metric="Ranked Mode Return Rate"
)

INT_04 = Intervention(
    id="INT-04",
    name="Weekend Event Invite",
    description=(
        "Send FUT Champions Weekend League invitation "
        "with bonus reward multiplier"
    ),
    applicable_tiers=["Critical"],
    primary_trigger="recency_days",
    expected_d7_lift_range=(0.08, 0.15),
    experiment_metric="D7 Retention Rate",
    guardrail_metric="Average Session Length"
)

INT_05 = Intervention(
    id="INT-05",
    name="Reward Progression Highlight",
    description=(
        "Show player their progress toward next "
        "reward tier with personalised milestone prompt"
    ),
    applicable_tiers=["At Risk", "Stable"],
    primary_trigger="session_consistency",
    expected_d7_lift_range=(0.03, 0.07),
    experiment_metric="D7 Retention Rate",
    guardrail_metric="Reward Claim Rate"
)

INT_06 = Intervention(
    id="INT-06",
    name="Friend Match Suggestion",
    description=(
        "Prompt player to invite a friend for "
        "a co-op or head-to-head match"
    ),
    applicable_tiers=["At Risk", "Critical"],
    primary_trigger="recency_days",
    expected_d7_lift_range=(0.05, 0.11),
    experiment_metric="D7 Retention Rate",
    guardrail_metric="Solo vs Social Match Ratio"
)

ALL_INTERVENTIONS = [INT_01, INT_02, INT_03, INT_04, INT_05, INT_06]


# ----------------------------------------------------------------------
# Intervention engine
# ----------------------------------------------------------------------

class InterventionEngine:
    """Converts PXI tier, quit probability, and SHAP drivers into
    ranked, confidence-scored LiveOps interventions and auto-generated
    experiment briefs."""

    def __init__(self, config_path=CONFIG_PATH):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['liveops']

        self.config_intervention = self.config['intervention']
        self.confidence_threshold = self.config_intervention['confidence_threshold']
        self.top_n = self.config_intervention['top_n_recommendations']
        self.experiment_run_days = self.config_intervention['experiment_run_days']
        self.confidence_target = self.config_intervention['experiment_confidence_target']
        self.min_d7_lift = self.config_intervention['min_d7_lift_to_rollout']
        self.db_path = str(ROOT / self.config['data']['db_path'])
        self.processed_path = ROOT / self.config['data']['processed_path']
        self.interventions = ALL_INTERVENTIONS

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_player_data(self) -> pd.DataFrame:
        """Load and merge PXI scores with quit predictions."""
        pxi_df = pd.read_csv(self.processed_path / 'pxi_scores.csv')
        quit_df = pd.read_csv(self.processed_path / 'quit_predictions.csv')

        quit_cols = [
            'player_id', 'match_id', 'quit_probability', 'is_high_risk',
            'skill_gap', 'driver_1_feature', 'driver_1_shap',
            'driver_2_feature', 'driver_2_shap', 'driver_3_feature', 'driver_3_shap'
        ]
        quit_subset = quit_df[quit_cols]

        df = pxi_df.merge(quit_subset, on=['player_id', 'match_id'], how='left')

        df['quit_probability'] = df['quit_probability'].fillna(df['quit_probability'].median())
        df['is_high_risk'] = df['is_high_risk'].fillna(False)

        self.logger.info(f"Shape after merge: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Confidence scoring engine
    # ------------------------------------------------------------------

    def compute_confidence(self, player: pd.Series, intervention: Intervention) -> float:
        """Score how suitable an intervention is for this player profile."""
        if player['pxi_tier'] not in intervention.applicable_tiers:
            return 0.0

        base_score = 0.30

        urgency = (1 - (player['pxi_score'] / 100)) * 0.25
        base_score += urgency

        quit_prob = player.get('quit_probability', 0.0)
        if pd.isna(quit_prob):
            quit_prob = 0.0
        base_score += quit_prob * 0.20

        trigger_relevance = {
            'skill_gap': player['normalized_skill_gap'] > 0.5,
            'losing_streak': player['losing_streak'] >= 3,
            'engagement_trend': player['pxi_engagement_trend'] < 40,
            'session_consistency': player['pxi_session_consistency'] < 40,
            'recency_days': player['recency_days'] > 7,
        }
        if trigger_relevance.get(intervention.primary_trigger, False):
            base_score += 0.15

        driver_1 = str(player.get('driver_1_feature', ''))
        driver_2 = str(player.get('driver_2_feature', ''))
        if intervention.primary_trigger in driver_1 or intervention.primary_trigger in driver_2:
            base_score += 0.10

        base_score = min(max(base_score, 0.0), 1.0)
        return round(base_score, 3)

    # ------------------------------------------------------------------
    # Recommendation engine
    # ------------------------------------------------------------------

    def recommend(self, player: pd.Series) -> List[Dict]:
        """Generate ranked intervention recommendations for one player."""
        scored = []
        for intervention in self.interventions:
            confidence = self.compute_confidence(player, intervention)
            if confidence >= self.confidence_threshold:
                scored.append((intervention, confidence))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:self.top_n]

        if not scored:
            return []

        results = []
        for rank, (intervention, confidence) in enumerate(scored, 1):
            if confidence >= 0.70:
                priority = '🔴 HIGH'
            elif confidence >= 0.50:
                priority = '🟠 MEDIUM'
            else:
                priority = '🟡 LOW'

            reasons = []
            if player['normalized_skill_gap'] > 0.5:
                reasons.append(f"Skill gap {player['skill_gap']:.2f} above comfort zone")
            if player['losing_streak'] >= 3:
                reasons.append(f"{player['losing_streak']}-match losing streak")
            if player['pxi_score'] < 40:
                reasons.append(f"PXI score {player['pxi_score']:.0f} (Critical tier)")
            quit_probability = player.get('quit_probability', 0.0)
            if pd.notna(quit_probability) and quit_probability > 0.60:
                reasons.append(f"Quit probability {quit_probability:.0%}")
            if player['recency_days'] > 7:
                reasons.append(f"Inactive for {player['recency_days']:.0f} days")

            if reasons:
                reason = ', '.join(reasons)
            else:
                reason = f"PXI {player['pxi_score']:.0f}, tier {player['pxi_tier']}"

            results.append({
                'rank': rank,
                'intervention_id': intervention.id,
                'intervention_name': intervention.name,
                'description': intervention.description,
                'confidence': confidence,
                'priority': priority,
                'reason': reason,
                'expected_d7_lift': (
                    f"{intervention.expected_d7_lift_range[0]:.0%}-"
                    f"{intervention.expected_d7_lift_range[1]:.0%}"
                ),
                'experiment_metric': intervention.experiment_metric,
                'guardrail_metric': intervention.guardrail_metric,
            })

        return results

    def recommend_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run recommendations for all players. Returns a flat
        DataFrame with one row per player x intervention recommendation."""
        results = []
        n_players_with_recs = 0

        for _, player in df.iterrows():
            recs = self.recommend(player)
            if recs:
                n_players_with_recs += 1
            for rec in recs:
                rec['player_id'] = player['player_id']
                rec['match_id'] = player['match_id']
                rec['pxi_tier'] = player['pxi_tier']
                rec['pxi_score'] = player['pxi_score']
                rec['primary_risk_factor'] = player['primary_risk_factor']
                results.append(rec)

        results_df = pd.DataFrame(results)
        self.logger.info(
            f"Generated {len(results_df)} recommendations for {n_players_with_recs} players"
        )
        return results_df

    # ------------------------------------------------------------------
    # Experiment brief generator
    # ------------------------------------------------------------------

    def _box_line(self, content: str, width: int = 48) -> str:
        content = content[:width]
        return f"║  {content.ljust(width - 2)}║"

    def _box_wrapped(self, content: str, width: int = 48) -> List[str]:
        wrapped = textwrap.wrap(content, width=width - 2) or ['']
        return [self._box_line(line, width) for line in wrapped]

    def generate_experiment_brief(
        self,
        intervention: Intervention,
        target_cohort_size: int,
        current_d7_rate: float = 0.45
    ) -> str:
        """Auto-generate a formatted experiment brief for a given intervention."""
        min_lift = self.min_d7_lift

        z_alpha = 1.645
        z_beta = 0.842

        p1 = current_d7_rate
        p2 = current_d7_rate + min_lift
        p_pool = (p1 + p2) / 2

        n = math.ceil(
            (z_alpha + z_beta) ** 2 * 2 * p_pool * (1 - p_pool) / (p2 - p1) ** 2
        )
        n = max(n, 100)

        available = target_cohort_size // 2
        cohort_note = f"{n} per arm required | {available} available per arm"
        if available < n:
            cohort_note += " ⚠️ underpowered, extend run time"

        tier = "/".join(intervention.applicable_tiers)
        lift_low, lift_high = intervention.expected_d7_lift_range

        width = 48
        top = "╔" + "═" * width + "╗"
        bottom = "╚" + "═" * width + "╝"
        sep = "╠" + "═" * width + "╣"
        blank = "║" + " " * width + "║"

        lines = [
            top,
            self._box_line(f"EXPERIMENT BRIEF — {intervention.id}", width),
            self._box_line(intervention.name, width),
            sep,
            blank,
            self._box_line("HYPOTHESIS", width),
            *self._box_wrapped(f"{intervention.description} will improve D7 retention among {tier} players.", width),
            blank,
            self._box_line("TARGET SEGMENT", width),
            self._box_line(f"Tiers: {tier}", width),
            self._box_line(f"Cohort size: {target_cohort_size} players", width),
            blank,
            self._box_line("DESIGN", width),
            self._box_line("Control:   Current configuration", width),
            *self._box_wrapped(f"Treatment: {intervention.description}", width),
            self._box_line("Split:     50/50 random assignment", width),
            blank,
            self._box_line("METRICS", width),
            self._box_line(f"Primary:   {intervention.experiment_metric}", width),
            self._box_line(f"Guardrail: {intervention.guardrail_metric}", width),
            blank,
            self._box_line("STATISTICS", width),
            self._box_line(cohort_note, width),
            self._box_line(f"Run time: {self.experiment_run_days} days", width),
            self._box_line(f"Confidence target: {self.confidence_target:.0%}", width),
            blank,
            self._box_line("EXPECTED OUTCOME", width),
            self._box_line(f"D7 lift: {lift_low:.0%} - {lift_high:.0%}", width),
            blank,
            self._box_line("DECISION CRITERIA", width),
            self._box_line(f"Roll out if D7 lift > {min_lift:.0%}", width),
            self._box_line(f"with {self.confidence_target:.0%} confidence", width),
            self._box_line("Kill if guardrail metric degrades", width),
            blank,
            bottom,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action queue
    # ------------------------------------------------------------------

    def compute_action_queue(self, recommendations_df: pd.DataFrame, pxi_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate individual recommendations into a LiveOps action queue."""
        total_at_risk = pxi_df['pxi_tier'].isin(['At Risk', 'Critical']).sum()

        intervention_lookup = {iv.id: iv for iv in self.interventions}
        priority_rank = {'🔴 HIGH': 0, '🟠 MEDIUM': 1, '🟡 LOW': 2}

        rows = []
        for intervention_id, group in recommendations_df.groupby('intervention_id'):
            iv = intervention_lookup[intervention_id]
            affected_players = group['player_id'].nunique()
            mean_confidence = group['confidence'].mean()
            overall_priority = group['priority'].value_counts().index[0]
            status = 'Ready' if mean_confidence >= 0.60 else 'Draft'

            rows.append({
                'priority_icon': overall_priority,
                'intervention_id': intervention_id,
                'intervention_name': iv.name,
                'description': iv.description,
                'affected_players': affected_players,
                'pct_of_at_risk': affected_players / total_at_risk if total_at_risk > 0 else 0.0,
                'mean_confidence': round(mean_confidence, 3),
                'expected_d7_lift': f"{iv.expected_d7_lift_range[0]:.0%}-{iv.expected_d7_lift_range[1]:.0%}",
                'status': status,
                'experiment_metric': iv.experiment_metric,
            })

        queue_df = pd.DataFrame(rows)
        queue_df['_priority_sort'] = queue_df['priority_icon'].map(priority_rank)
        queue_df = queue_df.sort_values(
            ['_priority_sort', 'mean_confidence'], ascending=[True, False]
        ).drop(columns=['_priority_sort']).reset_index(drop=True)

        return queue_df

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, recommendations_df: pd.DataFrame, action_queue_df: pd.DataFrame) -> None:
        """Save recommendations and action queue."""
        rec_path = self.processed_path / 'recommendations.csv'
        queue_path = self.processed_path / 'action_queue.csv'

        recommendations_df.to_csv(rec_path, index=False)
        action_queue_df.to_csv(queue_path, index=False)

        con = duckdb.connect(self.db_path)
        con.execute("CREATE OR REPLACE TABLE recommendations AS SELECT * FROM recommendations_df")
        con.execute("CREATE OR REPLACE TABLE action_queue AS SELECT * FROM action_queue_df")
        con.close()

        self.logger.info(f"Saved {len(recommendations_df)} rows to {rec_path}")
        self.logger.info(f"Saved {len(action_queue_df)} rows to {queue_path}")

    def run(self) -> tuple:
        """Full intervention engine pipeline. Returns (recommendations_df, action_queue_df)."""
        self.logger.info("Starting intervention engine pipeline...")

        df = self.load_player_data()

        self.logger.info("Generating recommendations...")
        recommendations_df = self.recommend_batch(df)

        self.logger.info("Building action queue...")
        action_queue_df = self.compute_action_queue(recommendations_df, df)

        self.save_results(recommendations_df, action_queue_df)

        self.logger.info("Intervention engine pipeline complete.")
        return recommendations_df, action_queue_df


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    engine = InterventionEngine()
    recs_df, queue_df = engine.run()

    print("\n=== ACTION QUEUE ===")
    print(queue_df[[
        'priority_icon', 'intervention_name',
        'affected_players', 'mean_confidence', 'status'
    ]].to_string(index=False))

    print("\n=== SAMPLE RECOMMENDATIONS ===")
    sample = recs_df.head(9)
    for player_id in sample['player_id'].unique()[:3]:
        player_recs = sample[sample['player_id'] == player_id]
        print(f"\nPlayer: {player_id}")
        for _, rec in player_recs.iterrows():
            print(
                f"  {rec['rank']}. "
                f"{rec['intervention_name']} "
                f"(confidence: {rec['confidence']:.2f}) "
                f"{rec['priority']}"
            )
            print(f"     Reason: {rec['reason']}")

    print("\n=== SAMPLE EXPERIMENT BRIEF ===")
    top_intervention = engine.interventions[0]
    at_risk_count = len(recs_df[recs_df['intervention_id'] == 'INT-01'])
    brief = engine.generate_experiment_brief(
        top_intervention,
        target_cohort_size=at_risk_count,
        current_d7_rate=0.31
    )
    print(brief)
