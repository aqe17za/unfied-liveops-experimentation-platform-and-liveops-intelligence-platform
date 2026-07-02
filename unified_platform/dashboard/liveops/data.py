"""Data loading layer for the LiveOps Intelligence pages.

Loads directly from Project 1's liveops.db. If the pipeline hasn't been run
yet, DB_MISSING is set and every page shows a clear instructional error
instead of crashing.
"""
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "processed" / "liveops.db"

TABLES = [
    'matches', 'match_features', 'player_features',
    'mqi_scores', 'pxi_scores', 'quit_predictions',
    'recommendations', 'action_queue',
    'north_star_metrics'
]


@st.cache_data(ttl=300)
def load_db_table(table_name: str) -> pd.DataFrame:
    """Load any table from liveops.db"""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        df = con.execute(f"SELECT * FROM {table_name}").df()
        con.close()
        return df
    except Exception as e:
        st.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()


def db_missing() -> bool:
    """Checked fresh on every call — reflects the current filesystem state."""
    return not DB_PATH.exists()


@st.cache_data(ttl=300)
def load_all_data() -> dict:
    """Load all pipeline outputs into a dict. Returns empty frames if DB is missing."""
    if not DB_PATH.exists():
        return {table: pd.DataFrame() for table in TABLES}
    return {table: load_db_table(table) for table in TABLES}


def show_missing_db_error():
    st.error(
        "**LiveOps data not found.**\n\n"
        f"`{DB_PATH}` does not exist. Run Project 1's pipeline first:\n\n"
        "```bash\n"
        "python src/data_pipeline.py\n"
        "python src/feature_engineering.py\n"
        "python src/mqi_engine.py\n"
        "python src/pxi_scorer.py\n"
        "python src/quit_predictor.py\n"
        "python src/intervention_engine.py\n"
        "```"
    )
