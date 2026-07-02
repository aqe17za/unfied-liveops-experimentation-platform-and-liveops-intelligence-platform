"""Feature engineering: Create match-level and simulated player-level features for ML."""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from data_pipeline import DataPipeline

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class FeatureEngineer:
    """Derives match-level and simulated player-level features from
    cleaned Division Rivals match data and persists them to the
    feature store and DuckDB warehouse."""

    DAY_MAP = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday',
    }

    def __init__(self, config_path=CONFIG_PATH):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['liveops']

        data_cfg = self.config['data']
        self.raw_path = ROOT / data_cfg['raw_path']
        self.processed_path = ROOT / data_cfg['processed_path']
        self.feature_store_path = ROOT / data_cfg['feature_store_path']
        self.db_path = str(ROOT / data_cfg['db_path'])

        self.feature_store_path.mkdir(parents=True, exist_ok=True)
        self.random_state = 42

    def engineer_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create match-level features from cleaned match data."""
        df = df.copy()

        df['skill_gap'] = (df['home_avg_rating'] - df['away_avg_rating']).abs()
        df['normalized_skill_gap'] = (df['skill_gap'] / df['skill_gap'].max()).clip(0, 1)

        df['score_diff_abs'] = (df['home_goals'] - df['away_goals']).abs()
        df['match_competitiveness'] = (1 - (df['score_diff_abs'] / df['score_diff_abs'].max())).clip(0, 1)

        possession_total = (df['home_possession_score'] + df['away_possession_score']).clip(lower=1)
        df['possession_balance'] = (
            1 - (df['home_possession_score'] - df['away_possession_score']).abs() / possession_total
        ).clip(0, 1)

        df['home_first_goal_advantage'] = (
            (df['home_first_goal'] == 1) & (df['home_win'] == 1)
        ).astype(int)

        df['comeback_flag'] = (df['possession_differential'].abs() < 500).astype(int)

        raw_intensity = (df['home_goals'] + df['away_goals'] + df['home_assists'] + df['away_assists']) / 4
        df['match_intensity'] = (raw_intensity / raw_intensity.max() * 100).fillna(0)

        objectives_gap = (df['home_objectives'] - df['away_objectives']).abs()
        max_gap = objectives_gap.max() if objectives_gap.max() > 0 else 1
        df['objectives_balance'] = ((max_gap - objectives_gap) / max_gap * 100)

        np.random.seed(self.random_state)
        df['match_hour'] = np.random.randint(0, 24, size=len(df))

        df['time_bucket'] = pd.cut(
            df['match_hour'],
            bins=[-1, 5, 11, 17, 23],
            labels=['night', 'morning', 'afternoon', 'evening'],
        ).astype(str)

        df['day_of_week'] = (df['match_id'] % 7).map(self.DAY_MAP)
        df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['goal_rate_diff'] = df['home_goals'] - df['away_goals']

        engineered_count = 15
        self.logger.info(f"Engineered {engineered_count} match features")
        return df

    def engineer_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate player-level features from match data.

        Since we have match-level data without persistent player IDs,
        we simulate a player perspective by treating home/away team
        positions as player proxies across matches. Each match
        generates two player records: home and away.
        """
        home_df = pd.DataFrame({
            'player_id': 'P' + df['match_id'].astype(str) + '_H',
            'match_id': df['match_id'],
            'rating': df['home_avg_rating'],
            'won': df['home_win'],
            'goals_scored': df['home_goals'],
            'goals_conceded': df['away_goals'],
            'possession': df['home_possession_score'],
            'normalized_skill_gap': df['normalized_skill_gap'],
            'ragequit_flag': 0,
        })

        away_df = pd.DataFrame({
            'player_id': 'P' + df['match_id'].astype(str) + '_A',
            'match_id': df['match_id'],
            'rating': df['away_avg_rating'],
            'won': 1 - df['home_win'],
            'goals_scored': df['away_goals'],
            'goals_conceded': df['home_goals'],
            'possession': df['away_possession_score'],
            'normalized_skill_gap': df['normalized_skill_gap'],
            'ragequit_flag': 0,
        })

        player_df = pd.concat([home_df, away_df], ignore_index=True)
        player_df = player_df.sort_values(['player_id', 'match_id']).reset_index(drop=True)

        np.random.seed(self.random_state)
        base_prob = 0.05
        losing_penalty = 0.15 * (1 - player_df['won'])
        skill_penalty = 0.10 * player_df['normalized_skill_gap']
        scoreline_penalty = 0.10 * ((player_df['goals_conceded'] - player_df['goals_scored']).clip(lower=0) / 5)
        ragequit_prob = (base_prob + losing_penalty + skill_penalty + scoreline_penalty).clip(0, 0.45)
        player_df['ragequit_flag'] = (np.random.random(len(player_df)) < ragequit_prob).astype(int)

        self.logger.info(f"Simulated ragequit rate: {player_df['ragequit_flag'].mean():.1%}")

        np.random.seed(self.random_state)
        n = len(player_df)
        player_df['matches_this_week'] = np.random.randint(1, 13, size=n)
        player_df['session_match_number'] = np.random.randint(1, 9, size=n)

        losing_streak_raw = np.random.randint(0, 6, size=n)
        player_df['losing_streak'] = np.where(player_df['won'] == 0, losing_streak_raw, 0)

        winning_streak_raw = np.random.randint(0, 6, size=n)
        player_df['winning_streak'] = np.where(player_df['won'] == 1, winning_streak_raw, 0)

        player_df['ragequit_rate_historical'] = np.random.uniform(0, 0.30, size=n)
        player_df['win_rate_last10'] = np.random.uniform(0.2, 0.8, size=n)
        player_df['recency_days'] = np.random.randint(0, 31, size=n)
        player_df['matches_last_week'] = np.random.randint(0, 11, size=n)

        self.logger.info(f"Created {len(player_df)} player match records")
        return player_df

    def save_feature_store(self, match_df: pd.DataFrame, player_df: pd.DataFrame) -> None:
        """Save engineered features to feature store and DuckDB."""
        match_path = self.feature_store_path / 'match_features.csv'
        player_path = self.feature_store_path / 'player_features.csv'

        match_df.to_csv(match_path, index=False)
        player_df.to_csv(player_path, index=False)

        con = duckdb.connect(self.db_path)
        con.execute("CREATE OR REPLACE TABLE match_features AS SELECT * FROM match_df")
        con.execute("CREATE OR REPLACE TABLE player_features AS SELECT * FROM player_df")
        con.close()

        self.logger.info(f"Saved match features to {match_path} ({len(match_df)} rows)")
        self.logger.info(f"Saved player features to {player_path} ({len(player_df)} rows)")

    def run(self, df: pd.DataFrame) -> tuple:
        """Run full feature engineering. Returns (match_df, player_df)."""
        match_df = self.engineer_match_features(df)
        player_df = self.engineer_player_features(match_df)
        self.save_feature_store(match_df, player_df)
        return match_df, player_df


if __name__ == "__main__":
    pipeline = DataPipeline()
    clean_df = pipeline.run()

    engineer = FeatureEngineer()
    match_df, player_df = engineer.run(clean_df)

    print("Match features shape:", match_df.shape)
    print("Match feature columns:", len(match_df.columns))
    print("Player features shape:", player_df.shape)
    print("Player feature columns:", len(player_df.columns))
