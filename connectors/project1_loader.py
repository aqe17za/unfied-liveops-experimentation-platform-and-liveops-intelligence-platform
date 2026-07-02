"""Loads EA SPORTS FC LiveOps Intelligence Platform (Project 1) pipeline outputs.

Primary source: data/processed/liveops.db — Project 1's live DuckDB output.
Fallback: data/processed/pxi_scores.csv and data/processed/action_queue.csv.

If neither exists, raises FileNotFoundError with an explicit message telling the
user to run Project 1's pipeline first. There is NO silent fallback to any
cached/copied snapshot; that would silently reintroduce the data-drift risk.

Schema notes (verified against the live database):
- pxi_scores: 19,758 rows, one per player_id (VARCHAR, e.g. "P1000_A").
  Contains losing_streak, matches_this_week, ragequit_rate_historical,
  win_rate_last10, recency_days — loaded directly, not hardcoded.
- action_queue: exactly 6 rows (one per intervention).
- recommendations: 26,397 rows (player x intervention). Joined to action_queue
  only to pull guardrail_metric per intervention_id (deduplicated).
- applicable_tiers is not a DB column — it comes from the Metric_Definitions.md
  spec (Section 3) and is hardcoded from that source, not invented.
"""
import duckdb
import pandas as pd
from pathlib import Path

# Live Project 1 output locations (relative-to-file, cwd-independent)
_ROOT = Path(__file__).resolve().parent.parent
_LIVEOPS_DB = _ROOT / "data" / "processed" / "liveops.db"
_PROCESSED_DIR = _ROOT / "data" / "processed"

_PIPELINE_MISSING_MSG = (
    "Project 1 pipeline output not found at {path}.\n"
    "Run Project 1's pipeline first from unified_platform/:\n"
    "  python src/data_pipeline.py\n"
    "  python src/feature_engineering.py\n"
    "  python src/mqi_engine.py\n"
    "  python src/pxi_scorer.py\n"
    "  python src/quit_predictor.py\n"
    "  python src/intervention_engine.py"
)

# From docs/Metric_Definitions.md Section 3 — The 6 Interventions -> Applicable Tiers
APPLICABLE_TIERS = {
    "INT-01": "Critical, At Risk",
    "INT-02": "At Risk, Stable",
    "INT-03": "Critical, At Risk",
    "INT-04": "Critical",
    "INT-05": "At Risk, Stable",
    "INT-06": "At Risk, Critical",
}

PLAYER_COLUMNS = [
    "player_id",
    "pxi_score",
    "pxi_tier",
    "losing_streak",
    "matches_this_week",
    "ragequit_rate_historical",
    "win_rate_last10",
    "recency_days",
]


def load_players_from_project1() -> pd.DataFrame:
    """Load player population from Project 1's live pxi_scores output.

    Tries liveops.db first, falls back to pxi_scores.csv.
    Raises FileNotFoundError with a clear pipeline-run instruction if neither exists.
    """
    if _LIVEOPS_DB.exists():
        conn = duckdb.connect(str(_LIVEOPS_DB), read_only=True)
        try:
            players = conn.execute(
                f"SELECT {', '.join(PLAYER_COLUMNS)} FROM pxi_scores"
            ).df()
            conn.close()
            print(f"Loaded {len(players)} players from {_LIVEOPS_DB} (pxi_scores)")
            return players
        except Exception as exc:
            conn.close()
            print(f"liveops.db exists but pxi_scores query failed: {exc}")

    csv_path = _PROCESSED_DIR / "pxi_scores.csv"
    if csv_path.exists():
        players = pd.read_csv(csv_path)[PLAYER_COLUMNS]
        print(f"Loaded {len(players)} players from {csv_path}")
        return players

    raise FileNotFoundError(
        _PIPELINE_MISSING_MSG.format(path=_LIVEOPS_DB)
    )


def load_interventions_from_project1() -> pd.DataFrame:
    """Load the 6 intervention cards from Project 1's live action_queue output.

    Enriches with guardrail_metric from recommendations (deduplicated on intervention_id).
    Raises FileNotFoundError with a clear pipeline-run instruction if data is missing.
    """
    if _LIVEOPS_DB.exists():
        conn = duckdb.connect(str(_LIVEOPS_DB), read_only=True)
        try:
            interventions = conn.execute("""
                SELECT
                    aq.intervention_id,
                    aq.intervention_name AS feature_name,
                    aq.description,
                    aq.affected_players,
                    aq.mean_confidence,
                    aq.expected_d7_lift,
                    aq.status,
                    aq.experiment_metric,
                    r.guardrail_metric
                FROM action_queue aq
                LEFT JOIN (
                    SELECT DISTINCT intervention_id, guardrail_metric
                    FROM recommendations
                ) r USING (intervention_id)
                ORDER BY aq.intervention_id
            """).df()
            conn.close()
            interventions["applicable_tiers"] = interventions["intervention_id"].map(APPLICABLE_TIERS)
            print(f"Loaded {len(interventions)} interventions from {_LIVEOPS_DB} (action_queue)")
            return interventions
        except Exception as exc:
            conn.close()
            print(f"liveops.db exists but action_queue query failed: {exc}")

    aq_path = _PROCESSED_DIR / "action_queue.csv"
    rec_path = _PROCESSED_DIR / "recommendations.csv"
    if aq_path.exists() and rec_path.exists():
        aq = pd.read_csv(aq_path).rename(columns={"intervention_name": "feature_name"})
        rec = (
            pd.read_csv(rec_path)[["intervention_id", "guardrail_metric"]]
            .drop_duplicates("intervention_id")
        )
        interventions = (
            aq.merge(rec, on="intervention_id", how="left")
            .sort_values("intervention_id")
            .reset_index(drop=True)
        )
        interventions["applicable_tiers"] = interventions["intervention_id"].map(APPLICABLE_TIERS)
        print(f"Loaded {len(interventions)} interventions from {aq_path}")
        return interventions

    raise FileNotFoundError(
        _PIPELINE_MISSING_MSG.format(path=_LIVEOPS_DB)
    )
