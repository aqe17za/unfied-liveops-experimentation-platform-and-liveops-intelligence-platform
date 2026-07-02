"""Shared analytics schema for Project 2 (EA SPORTS FC Live Experimentation Platform).

player_id is VARCHAR (e.g. "P1000_A") to match Project 1's liveops.db, not INTEGER.
"""
import duckdb
from pathlib import Path

SCHEMA_DB = Path(__file__).parent / "schema.db"


def init_schema(db_path: Path = SCHEMA_DB):
    """Initialize the shared analytics schema."""
    conn = duckdb.connect(str(db_path))

    # Players table (from Project 1's pxi_scores, one row per player)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id VARCHAR PRIMARY KEY,
            pxi_score FLOAT,
            pxi_tier VARCHAR,
            losing_streak INTEGER,
            matches_this_week INTEGER,
            ragequit_rate_historical FLOAT,
            win_rate_last10 FLOAT,
            recency_days INTEGER
        )
    """)

    # Interventions table (from Project 1's action_queue + recommendations)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            intervention_id VARCHAR PRIMARY KEY,
            feature_name VARCHAR,
            description VARCHAR,
            applicable_tiers VARCHAR,
            expected_d7_lift VARCHAR,
            guardrail_metric VARCHAR,
            affected_players BIGINT,
            mean_confidence DOUBLE,
            status VARCHAR
        )
    """)

    # Experiments table (created by Project 2)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id VARCHAR PRIMARY KEY,
            feature_name VARCHAR,
            hypothesis VARCHAR,
            primary_metric VARCHAR,
            guardrail_metrics VARCHAR,
            success_criteria VARCHAR,
            rollback_criteria VARCHAR,
            engineering_cost VARCHAR,
            expected_product_impact VARCHAR,
            target_segment VARCHAR,
            sample_size INTEGER,
            control_size INTEGER,
            treatment_size INTEGER,
            status VARCHAR,
            ers_score FLOAT,
            decision VARCHAR
        )
    """)

    # Experiment results table (player-level outcomes, created by Project 2)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_results (
            experiment_id VARCHAR,
            player_id VARCHAR,
            assignment INTEGER,
            primary_metric_value DOUBLE,
            early_period_outcome DOUBLE,
            late_period_outcome DOUBLE,
            guardrail_ragequit DOUBLE,
            guardrail_completion DOUBLE,
            segment_pxi_tier VARCHAR,
            segment_player_type VARCHAR,
            segment_play_style VARCHAR
        )
    """)

    # Metrics table (definitions)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id VARCHAR PRIMARY KEY,
            metric_name VARCHAR,
            definition VARCHAR,
            direction VARCHAR,
            baseline FLOAT,
            healthy_threshold FLOAT,
            warning_threshold FLOAT
        )
    """)

    conn.close()


if __name__ == "__main__":
    init_schema()
    print(f"Schema initialized at {SCHEMA_DB}")
