"""MQI engine: Match Quality Index scoring (0-100) with 4 components and validation."""

import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MQIEngine:
    """Computes the Match Quality Index (MQI) — a 0-100 composite score
    rating every match across competitiveness, skill balance, quit
    behavior, and comeback factor."""

    def __init__(self, config_path=CONFIG_PATH):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['liveops']

        self.weights = self.config['mqi']['weights']
        self.tiers = self.config['mqi']['tiers']
        self.db_path = str(ROOT / self.config['data']['db_path'])
        self.processed_path = ROOT / self.config['data']['processed_path']
        self.feature_store_path = ROOT / self.config['data']['feature_store_path']

    # ------------------------------------------------------------------
    # Component methods
    # ------------------------------------------------------------------

    def compute_competitiveness(self, df: pd.DataFrame) -> pd.Series:
        """
        Measures how close the match was.
        High score = low goal difference = competitive match.
        """
        max_diff = max(df['score_diff_abs'].max(), 1)
        raw_1 = 1 - (df['score_diff_abs'] / max_diff)
        raw_2 = df['match_competitiveness']
        combined = ((raw_1 + raw_2) / 2 * 100).clip(0, 100)
        return combined.rename('comp_competitiveness')

    def compute_skill_balance(self, df: pd.DataFrame) -> pd.Series:
        """
        Measures how evenly matched the squads were.
        High score = small skill gap = fair matchmaking.
        """
        raw = 1 - df['normalized_skill_gap']
        return (raw * 100).clip(0, 100).rename('comp_skill_balance')

    def compute_quit_penalty(self, df: pd.DataFrame) -> pd.Series:
        """
        Penalizes matches where a ragequit proxy condition is met.
        match_features is match-level, not player-level, so a
        match-level ragequit proxy is used: a large pre-match skill
        gap combined with a large final score gap.
        """
        match_ragequit = (
            (df['normalized_skill_gap'] > 0.6) &
            (df['score_diff_abs'] > 3)
        ).astype(int)
        return ((1 - match_ragequit) * 100).rename('comp_quit_penalty')

    def compute_comeback_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Measures whether the losing team had a fighting chance.
        High score = comeback was possible = engaging match.
        """
        score = pd.Series(15, index=df.index, dtype=float)
        score = score + (25 * (df['score_diff_abs'] <= 2).astype(float))
        score = score + (25 * (df['score_diff_abs'] <= 1).astype(float))
        score[df['comeback_flag'] == 1] = 85
        return score.clip(0, 100).rename('comp_comeback_factor')

    # ------------------------------------------------------------------
    # Main scoring method
    # ------------------------------------------------------------------

    def compute_mqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute full MQI for every match.
        Returns df with MQI score, tier, explanation columns added.
        """
        working = df.copy()

        working['comp_competitiveness'] = self.compute_competitiveness(df)
        working['comp_skill_balance'] = self.compute_skill_balance(df)
        working['comp_quit_penalty'] = self.compute_quit_penalty(df)
        working['comp_comeback_factor'] = self.compute_comeback_factor(df)

        w = self.weights
        working['mqi_score'] = (
            w['competitiveness'] * working['comp_competitiveness'] +
            w['skill_balance'] * working['comp_skill_balance'] +
            w['quit_penalty'] * working['comp_quit_penalty'] +
            w['comeback_factor'] * working['comp_comeback_factor']
        ).round(2)

        working['mqi_tier'] = working['mqi_score'].apply(self.assign_tier)

        component_cols = [
            'comp_competitiveness', 'comp_skill_balance',
            'comp_quit_penalty', 'comp_comeback_factor'
        ]
        component_labels = {
            'comp_competitiveness': 'Competitiveness',
            'comp_skill_balance': 'Skill Balance',
            'comp_quit_penalty': 'Quit Penalty',
            'comp_comeback_factor': 'Comeback Factor'
        }
        working['primary_drag'] = working[component_cols].idxmin(axis=1).map(component_labels)

        working['mqi_explanation'] = working.apply(self.explain_match, axis=1)

        self.logger.info(
            f"MQI computed for {len(working)} matches. "
            f"Mean MQI: {working['mqi_score'].mean():.1f} | "
            f"Tier distribution: {working['mqi_tier'].value_counts().to_dict()}"
        )

        return working

    def assign_tier(self, score) -> str:
        """Assign MQI tier label based on score thresholds."""
        tiers = self.tiers
        if score >= tiers['elite']:
            return 'Elite'
        elif score >= tiers['good']:
            return 'Good'
        elif score >= tiers['average']:
            return 'Average'
        elif score >= tiers['below_average']:
            return 'Below Average'
        else:
            return 'Poor'

    def explain_match(self, row) -> str:
        """Generate a human-readable explanation string for one match."""
        return (
            f"MQI: {row['mqi_score']:.1f} ({row['mqi_tier']}) | "
            f"Competitiveness: {row['comp_competitiveness']:.0f} | "
            f"Skill Balance: {row['comp_skill_balance']:.0f} | "
            f"Quit Penalty: {row['comp_quit_penalty']:.0f} | "
            f"Comeback Factor: {row['comp_comeback_factor']:.0f} | "
            f"Primary drag: {row['primary_drag']}"
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Test how MQI correlates with a ragequit proxy across
        candidate weight combinations.

        Higher MQI should mean LOWER ragequit risk, so the desired
        relationship is a negative Pearson correlation. The weight
        set producing the strongest (most negative) correlation is
        recommended.
        """
        comp_competitiveness = self.compute_competitiveness(df)
        comp_skill_balance = self.compute_skill_balance(df)
        comp_quit_penalty = self.compute_quit_penalty(df)
        comp_comeback_factor = self.compute_comeback_factor(df)

        ragequit_proxy = (
            (df['normalized_skill_gap'] > 0.6) & (df['score_diff_abs'] > 3)
        ).astype(int)

        weight_sets = {
            'Set A (base)': {
                'competitiveness': 0.32, 'skill_balance': 0.28,
                'quit_penalty': 0.20, 'comeback_factor': 0.20
            },
            'Set B (competitiveness heavy)': {
                'competitiveness': 0.40, 'skill_balance': 0.25,
                'quit_penalty': 0.20, 'comeback_factor': 0.15
            },
            'Set C (skill balance heavy)': {
                'competitiveness': 0.25, 'skill_balance': 0.40,
                'quit_penalty': 0.20, 'comeback_factor': 0.15
            },
            'Set D (quit penalty heavy)': {
                'competitiveness': 0.25, 'skill_balance': 0.25,
                'quit_penalty': 0.35, 'comeback_factor': 0.15
            },
            'Set E (comeback heavy)': {
                'competitiveness': 0.25, 'skill_balance': 0.25,
                'quit_penalty': 0.20, 'comeback_factor': 0.30
            },
            'Set F (equal)': {
                'competitiveness': 0.25, 'skill_balance': 0.25,
                'quit_penalty': 0.25, 'comeback_factor': 0.25
            },
        }

        base_r = None
        results = []
        for label, w in weight_sets.items():
            mqi_score = (
                w['competitiveness'] * comp_competitiveness +
                w['skill_balance'] * comp_skill_balance +
                w['quit_penalty'] * comp_quit_penalty +
                w['comeback_factor'] * comp_comeback_factor
            )
            r, p = stats.pearsonr(mqi_score, ragequit_proxy)

            if label == 'Set A (base)':
                base_r = r
                interpretation = 'Base case'
            elif r < base_r:
                interpretation = 'Stronger negative correlation than base'
            elif r > base_r:
                interpretation = 'Weaker negative correlation than base'
            else:
                interpretation = 'Equivalent to base'

            results.append({
                'weight_set': label,
                'w_competitiveness': w['competitiveness'],
                'w_skill_balance': w['skill_balance'],
                'w_quit_penalty': w['quit_penalty'],
                'w_comeback_factor': w['comeback_factor'],
                'pearson_r': round(r, 4),
                'pearson_p': p,
                'interpretation': interpretation,
            })

        results_df = pd.DataFrame(results).sort_values('pearson_r').reset_index(drop=True)

        best = results_df.iloc[0]
        self.logger.info(
            f"Recommended weight set: {best['weight_set']} "
            f"(pearson_r={best['pearson_r']}, p={best['pearson_p']:.4g})"
        )

        return results_df

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save_mqi_results(self, df: pd.DataFrame) -> None:
        """Save MQI-scored matches to processed/ and DuckDB."""
        cols = [
            'match_id', 'mqi_score', 'mqi_tier', 'mqi_explanation',
            'primary_drag', 'comp_competitiveness', 'comp_skill_balance',
            'comp_quit_penalty', 'comp_comeback_factor',
            'skill_gap', 'normalized_skill_gap', 'score_diff_abs',
            'match_competitiveness', 'comeback_flag',
            'home_win', 'home_goals', 'away_goals',
            'time_bucket', 'day_of_week', 'is_weekend'
        ]
        save_df = df[cols]

        out_path = self.processed_path / 'mqi_scores.csv'
        save_df.to_csv(out_path, index=False)

        con = duckdb.connect(self.db_path)
        con.execute("CREATE OR REPLACE TABLE mqi_scores AS SELECT * FROM save_df")
        con.close()

        self.logger.info(f"Saved {len(save_df)} rows to mqi_scores")

    def run(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Full MQI pipeline.
        If df is None, load from feature store.
        Returns MQI-scored DataFrame.
        """
        if df is None:
            df = pd.read_csv(self.feature_store_path / 'match_features.csv')
        mqi_df = self.compute_mqi(df)
        self.save_mqi_results(mqi_df)
        return mqi_df


if __name__ == "__main__":
    engine = MQIEngine()
    mqi_df = engine.run()
    print(mqi_df[['match_id', 'mqi_score', 'mqi_tier', 'primary_drag']].head(10))
    print("\nTier distribution:")
    print(mqi_df['mqi_tier'].value_counts())
    print(f"\nMean MQI: {mqi_df['mqi_score'].mean():.2f}")
