"""Data pipeline: Load raw CSV, apply FC language mapping, clean, and save to DuckDB."""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DataPipeline:
    """Loads raw Division Rivals match telemetry, applies FC language
    mapping, cleans it, and persists it to CSV and DuckDB."""

    FC_COLUMN_MAP = {
        'blueWins': 'home_win',
        'blueWardsPlaced': 'home_coverage_placed',
        'blueWardsDestroyed': 'home_coverage_destroyed',
        'blueFirstBlood': 'home_first_goal',
        'blueKills': 'home_goals',
        'blueDeaths': 'home_conceded',
        'blueAssists': 'home_assists',
        'blueEliteMonsters': 'home_objectives',
        'blueDragons': 'home_major_objectives',
        'blueHeralds': 'home_secondary_objectives',
        'blueTowersDestroyed': 'home_structures_taken',
        'blueTotalGold': 'home_possession_score',
        'blueAvgLevel': 'home_avg_rating',
        'blueTotalExperience': 'home_total_rating_points',
        'blueTotalMinionsKilled': 'home_total_actions',
        'blueTotalJungleMinionsKilled': 'home_contested_actions',
        'blueGoldDiff': 'possession_differential',
        'blueExperienceDiff': 'home_momentum_diff',
        'blueCSPerMin': 'home_actions_per_min',
        'blueGoldPerMin': 'home_possession_per_min',
        'redWardsPlaced': 'away_coverage_placed',
        'redWardsDestroyed': 'away_coverage_destroyed',
        'redFirstBlood': 'away_first_goal',
        'redKills': 'away_goals',
        'redDeaths': 'away_conceded',
        'redAssists': 'away_assists',
        'redEliteMonsters': 'away_objectives',
        'redDragons': 'away_major_objectives',
        'redHeralds': 'away_secondary_objectives',
        'redTowersDestroyed': 'away_structures_taken',
        'redTotalGold': 'away_possession_score',
        'redAvgLevel': 'away_avg_rating',
        'redTotalExperience': 'away_total_rating_points',
        'redTotalMinionsKilled': 'away_total_actions',
        'redTotalJungleMinionsKilled': 'away_contested_actions',
        'redGoldDiff': 'away_possession_differential',
        'redExperienceDiff': 'away_momentum_diff',
        'redCSPerMin': 'away_actions_per_min',
        'redGoldPerMin': 'away_possession_per_min',
    }

    def __init__(self, config_path=CONFIG_PATH):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['liveops']

        data_cfg = self.config['data']
        self.raw_path = ROOT / data_cfg['raw_path']
        self.processed_path = ROOT / data_cfg['processed_path']
        self.db_path = str(ROOT / data_cfg['db_path'])
        self.filename = data_cfg['filename']

        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_raw(self) -> pd.DataFrame:
        """Load raw CSV from data/raw/ and apply FC language mapping."""
        raw_file = self.raw_path / self.filename
        df = pd.read_csv(raw_file)
        self.logger.info(f"Loaded {len(df)} matches from raw source")

        df = df.rename(columns=self.FC_COLUMN_MAP)
        df = df.drop(columns=['gameId'], errors='ignore')
        df.insert(0, 'match_id', range(1, len(df) + 1))

        self.logger.info(f"Columns after FC mapping: {df.columns.tolist()}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe — remove duplicates, handle nulls,
        filter incomplete matches."""
        self.logger.info(f"Initial shape: {df.shape}")

        before = len(df)
        df = df.drop_duplicates(subset='match_id')
        duplicates_removed = before - len(df)
        self.logger.info(f"Removed {duplicates_removed} duplicate match_ids")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                df[col] = df[col].fillna(df[col].median())
                self.logger.info(f"Filled {n_nulls} nulls in '{col}' with median")

        before = len(df)
        df = df[(df['home_goals'] >= 0) & (df['away_goals'] >= 0)]
        df = df[df['home_possession_score'] > 0]
        rows_removed = before - len(df)
        self.logger.info(f"Removed {rows_removed} invalid match rows")

        df['data_quality_flag'] = 1

        if 'home_win' not in df.columns:
            df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)

        self.logger.info(f"Final shape: {df.shape}")
        return df

    def save_processed(self, df: pd.DataFrame) -> None:
        """Save cleaned data to CSV and DuckDB."""
        csv_path = self.processed_path / 'fc_matches_clean.csv'
        df.to_csv(csv_path, index=False)

        con = duckdb.connect(self.db_path)
        con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM df")
        self.logger.info(f"Saved {len(df)} matches to DuckDB table 'matches'")
        con.close()

    def run(self) -> pd.DataFrame:
        """Execute full pipeline. Returns clean DataFrame."""
        self.logger.info("Starting data pipeline...")
        df = self.load_raw()
        df = self.clean(df)
        self.save_processed(df)
        self.logger.info("Pipeline complete.")
        return df


if __name__ == "__main__":
    pipeline = DataPipeline()
    df = pipeline.run()
    print(df.head())
    print(df.shape)
    print(df.columns.tolist())
