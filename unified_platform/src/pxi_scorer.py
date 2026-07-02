"""PXI scorer: Player Experience Score aggregating weekly health with north-star metrics."""

import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class PXIScorer:
    """Aggregates match-level MQI into a weekly Player Experience Score
    (PXI) — a 0-100 composite health score classifying players into
    Healthy, Stable, At Risk, or Critical tiers."""

    def __init__(self, config_path=CONFIG_PATH):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['liveops']

        self.weights = self.config['pxi']['weights']
        self.tiers = self.config['pxi']['tiers']
        self.benchmarks = self.config['pxi']['north_star_benchmarks']
        self.db_path = str(ROOT / self.config['data']['db_path'])
        self.processed_path = ROOT / self.config['data']['processed_path']
        self.feature_store_path = ROOT / self.config['data']['feature_store_path']

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def load_and_merge(self) -> pd.DataFrame:
        """
        Load player_features and mqi_scores and merge on match_id
        to attach MQI signals to each player record.
        """
        player_df = pd.read_csv(self.feature_store_path / 'player_features.csv')
        mqi_df = pd.read_csv(self.processed_path / 'mqi_scores.csv')

        mqi_cols = [
            'match_id', 'mqi_score', 'mqi_tier', 'primary_drag',
            'comp_competitiveness', 'comp_skill_balance',
            'comp_quit_penalty', 'comp_comeback_factor'
        ]
        mqi_subset = mqi_df[mqi_cols]

        merged = player_df.merge(mqi_subset, on='match_id', how='left')
        self.logger.info(f"Shape after merge: {merged.shape}")

        numeric_mqi_cols = ['mqi_score', 'comp_competitiveness', 'comp_skill_balance',
                            'comp_quit_penalty', 'comp_comeback_factor']
        null_counts = merged[numeric_mqi_cols].isna().sum()
        self.logger.info(f"Null counts in MQI columns after merge: {null_counts.to_dict()}")

        for col in numeric_mqi_cols:
            if merged[col].isna().any():
                merged[col] = merged[col].fillna(merged[col].median())

        return merged

    # ------------------------------------------------------------------
    # PXI component methods
    # ------------------------------------------------------------------

    def compute_avg_mqi_last5(self, df: pd.DataFrame) -> pd.Series:
        """
        Average MQI score across player's recent matches.
        Higher recent MQI = better experience = higher score.
        """
        raw = df['mqi_score']
        scaled = ((raw - raw.min()) / (raw.max() - raw.min()) * 100).clip(0, 100)
        return scaled.rename('pxi_avg_mqi')

    def compute_session_consistency(self, df: pd.DataFrame) -> pd.Series:
        """
        How regularly is the player active compared to their
        historical pattern?
        """
        last_week = df['matches_last_week'].clip(lower=1)
        consistency = (df['matches_this_week'] / last_week).clip(0, 1)
        return (consistency * 100).rename('pxi_session_consistency')

    def compute_engagement_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Is the player's activity increasing or decreasing?
        """
        delta = df['matches_this_week'] - df['matches_last_week']
        max_pos = max(delta.clip(lower=0).max(), 1)
        max_neg = max(delta.clip(upper=0).abs().max(), 1)

        score = pd.Series(50.0, index=df.index)
        score[delta >= 0] += (delta[delta >= 0] / max_pos * 50)
        score[delta < 0] += (delta[delta < 0] / max_neg * 50)
        return score.clip(0, 100).rename('pxi_engagement_trend')

    def compute_streak_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Penalize losing streaks, reward winning streaks.
        """
        base = 50.0
        winning_bonus = (df['winning_streak'] * 5).clip(0, 30)
        losing_penalty = (df['losing_streak'] * 8).clip(0, 40)
        score = (base + winning_bonus - losing_penalty).clip(0, 100)
        return score.rename('pxi_streak_factor')

    # ------------------------------------------------------------------
    # Main PXI computation
    # ------------------------------------------------------------------

    def compute_pxi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PXI score for every player record.
        Returns df with PXI columns added.
        """
        working = df.copy()

        working['pxi_avg_mqi'] = self.compute_avg_mqi_last5(df)
        working['pxi_session_consistency'] = self.compute_session_consistency(df)
        working['pxi_engagement_trend'] = self.compute_engagement_trend(df)
        working['pxi_streak_factor'] = self.compute_streak_factor(df)

        w = self.weights
        working['pxi_score'] = (
            w['avg_mqi_last5'] * working['pxi_avg_mqi'] +
            w['session_consistency'] * working['pxi_session_consistency'] +
            w['engagement_trend'] * working['pxi_engagement_trend'] +
            w['streak_factor'] * working['pxi_streak_factor']
        ).round(2)

        working['pxi_tier'] = working['pxi_score'].apply(self.assign_tier)

        pxi_component_cols = [
            'pxi_avg_mqi', 'pxi_session_consistency',
            'pxi_engagement_trend', 'pxi_streak_factor'
        ]
        pxi_component_labels = {
            'pxi_avg_mqi': 'Poor Match Quality',
            'pxi_session_consistency': 'Irregular Sessions',
            'pxi_engagement_trend': 'Declining Activity',
            'pxi_streak_factor': 'Losing Streak'
        }
        working['primary_risk_factor'] = working[pxi_component_cols].idxmin(axis=1).map(pxi_component_labels)

        self.logger.info(
            f"PXI computed for {len(working)} player records. "
            f"Mean PXI: {working['pxi_score'].mean():.1f} | "
            f"Tier distribution: {working['pxi_tier'].value_counts().to_dict()}"
        )

        return working

    def assign_tier(self, score) -> str:
        """Assign PXI tier label based on score thresholds."""
        tiers = self.tiers
        if score >= tiers['healthy']:
            return 'Healthy'
        elif score >= tiers['stable']:
            return 'Stable'
        elif score >= tiers['at_risk']:
            return 'At Risk'
        else:
            return 'Critical'

    # ------------------------------------------------------------------
    # North-star metrics
    # ------------------------------------------------------------------

    def compute_north_star_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the north-star metric table showing retention and
        engagement per PXI tier.
        """
        tiers_order = ['Healthy', 'Stable', 'At Risk', 'Critical']
        ref_pxi = {'Healthy': 75, 'Stable': 65, 'At Risk': 45, 'Critical': 25}

        rows = []
        for tier in tiers_order:
            tier_df = df[df['pxi_tier'] == tier]
            if len(tier_df) == 0:
                continue

            tier_key = tier.lower().replace(' ', '_')
            bench = self.benchmarks.get(tier_key, {})
            mean_pxi = tier_df['pxi_score'].mean()
            scale = mean_pxi / ref_pxi[tier]

            primary_risk = (
                tier_df['primary_risk_factor'].value_counts().index[0]
                if len(tier_df) > 0 else 'N/A'
            )

            rows.append({
                'PXI Tier': tier,
                'Player Count': len(tier_df),
                '% of Total': f"{len(tier_df) / len(df) * 100:.1f}%",
                'Mean PXI': f"{mean_pxi:.1f}",
                'D1 Return Rate': f"{min(0.99, max(0.05, bench.get('d1_return', 0.5) * scale)):.1%}",
                'D7 Return Rate': f"{min(0.99, max(0.05, bench.get('d7_return', 0.3) * scale)):.1%}",
                'Avg Matches/Week': f"{bench.get('avg_matches_per_week', 3):.1f}",
                'Primary Risk Factor': primary_risk
            })

        north_star_df = pd.DataFrame(rows)
        self.logger.info(f"\nNorth-Star Metrics:\n{north_star_df.to_string()}")
        return north_star_df

    def identify_at_risk_players(self, df: pd.DataFrame, pxi_threshold=54) -> pd.DataFrame:
        """
        Return all players in At Risk or Critical tier.
        """
        at_risk_df = df[df['pxi_score'] < pxi_threshold].copy()

        cols = [
            'player_id', 'pxi_score', 'pxi_tier',
            'primary_risk_factor', 'pxi_avg_mqi',
            'pxi_session_consistency', 'pxi_engagement_trend',
            'pxi_streak_factor', 'losing_streak',
            'recency_days', 'matches_this_week',
            'ragequit_flag', 'mqi_score'
        ]
        at_risk_df = at_risk_df[cols].sort_values('pxi_score', ascending=True).reset_index(drop=True)

        self.logger.info(
            f"Identified {len(at_risk_df)} at-risk players "
            f"({len(at_risk_df) / len(df):.1%} of all players)"
        )

        return at_risk_df

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_pxi_results(self, df: pd.DataFrame, north_star_df: pd.DataFrame) -> None:
        """Save PXI results to processed/ and DuckDB."""
        pxi_path = self.processed_path / 'pxi_scores.csv'
        north_star_path = self.processed_path / 'north_star_metrics.csv'

        df.to_csv(pxi_path, index=False)
        north_star_df.to_csv(north_star_path, index=False)

        con = duckdb.connect(self.db_path)
        con.execute("CREATE OR REPLACE TABLE pxi_scores AS SELECT * FROM df")
        con.execute("CREATE OR REPLACE TABLE north_star_metrics AS SELECT * FROM north_star_df")
        con.close()

        self.logger.info(f"Saved {len(df)} rows to {pxi_path}")
        self.logger.info(f"Saved {len(north_star_df)} rows to {north_star_path}")

    def run(self) -> tuple:
        """
        Full PXI pipeline.
        Returns (pxi_df, north_star_df, at_risk_df)
        """
        self.logger.info("Starting PXI scoring pipeline...")

        df = self.load_and_merge()
        pxi_df = self.compute_pxi(df)
        north_star_df = self.compute_north_star_metrics(pxi_df)
        at_risk_df = self.identify_at_risk_players(pxi_df)
        self.save_pxi_results(pxi_df, north_star_df)

        self.logger.info("PXI pipeline complete.")
        return pxi_df, north_star_df, at_risk_df


if __name__ == "__main__":
    scorer = PXIScorer()
    pxi_df, north_star_df, at_risk_df = scorer.run()

    print("\n=== PXI SCORE SUMMARY ===")
    print(f"Total player records: {len(pxi_df)}")
    print(f"Mean PXI: {pxi_df['pxi_score'].mean():.2f}")
    print("\nTier distribution:")
    print(pxi_df['pxi_tier'].value_counts())

    print("\n=== NORTH-STAR METRICS ===")
    print(north_star_df.to_string(index=False))

    print("\n=== AT-RISK PLAYERS ===")
    print(f"At-risk player count: {len(at_risk_df)}")
    print(f"As % of total: {len(at_risk_df) / len(pxi_df) * 100:.1f}%")
    print(at_risk_df.head(10)[['player_id', 'pxi_score', 'pxi_tier', 'primary_risk_factor']])
