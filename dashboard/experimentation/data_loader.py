"""Data loading layer for the Experimentation pages.

The dashboard treats every Phase 1-5 CSV export as an immutable artifact.
This module only reads and normalizes those files, and returns useful empty
states instead of crashing when a file is unavailable.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "simulation"

CSV_FILES = {
    "executive_summary": "executive_summary.csv",
    "decisions": "decisions.csv",
    "decision_audit": "decision_audit.csv",
    "hte_summary": "hte_summary.csv",
    "statistical_results": "statistical_results.csv",
    "heterogeneous_effects_results": "heterogeneous_effects_results.csv",
    "validation_report": "validation_report.csv",
}

_RENAME_MAPS = {
    "executive_summary": {
        "Experiment": "experiment_id",
        "Decision": "decision",
        "Confidence": "confidence",
        "Primary Reason": "primary_reason",
        "Next Action": "next_action",
    },
}


def data_missing() -> bool:
    """Checked fresh on every call — true if none of the pipeline CSVs exist yet."""
    return not any((DATA_DIR / filename).exists() for filename in CSV_FILES.values())


def show_missing_data_error():
    st.error(
        "**Experimentation data not found.**\n\n"
        f"No pipeline CSVs exist in `{DATA_DIR}`. Run the experimentation pipeline first:\n\n"
        "```bash\n"
        "python src/experiment_manager.py\n"
        "python src/run_experimentation_pipeline.py\n"
        "```"
    )


@st.cache_data
def load_csv(key: str):
    """Load a known CSV by key and return (DataFrame, error_message)."""
    filename = CSV_FILES.get(key)
    if filename is None:
        return pd.DataFrame(), f"Unknown dataset key: {key}"

    path = DATA_DIR / filename
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(), f"{filename} was not found in data/simulation."
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), f"{filename} is empty."
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        return pd.DataFrame(), f"Could not read {filename}: {exc}"

    rename_map = _RENAME_MAPS.get(key)
    if rename_map:
        df = df.rename(columns=rename_map)

    return df, None


@st.cache_data
def load_all():
    """Load every dashboard dataset once."""
    return {key: load_csv(key) for key in CSV_FILES}


def safe_get_row(df: pd.DataFrame, id_col: str, value):
    """Return one matching row, or None when the data is missing."""
    if df is None or df.empty or id_col not in df.columns:
        return None
    matches = df[df[id_col] == value]
    if matches.empty:
        return None
    return matches.iloc[0]


def safe_get_rows(df: pd.DataFrame, id_col: str, value):
    """Return all matching rows, or an empty DataFrame."""
    if df is None or df.empty or id_col not in df.columns:
        return pd.DataFrame()
    return df[df[id_col] == value]


def get_all_experiment_ids(executive_summary_df: pd.DataFrame):
    """Return the canonical experiment list from executive_summary.csv."""
    if (
        executive_summary_df is None
        or executive_summary_df.empty
        or "experiment_id" not in executive_summary_df.columns
    ):
        return []
    return sorted(executive_summary_df["experiment_id"].dropna().unique().tolist())


def get_all_decision_types(decisions_df: pd.DataFrame):
    """Return decision types present in the loaded data."""
    if decisions_df is None or decisions_df.empty or "decision" not in decisions_df.columns:
        return []
    return sorted(decisions_df["decision"].dropna().unique().tolist())


def badge_for_decision(decision) -> str:
    """Return a short ASCII marker for a decision."""
    if not isinstance(decision, str) or not decision.strip():
        return "REVIEW"
    decision_upper = decision.upper()
    if "KILL" in decision_upper:
        return "STOP"
    if "ROLLOUT" in decision_upper or "SEGMENT" in decision_upper:
        return "TARGET"
    if "SHIP" in decision_upper:
        return "GO"
    if "BLOCK" in decision_upper:
        return "HOLD"
    return "REVIEW"


def tone_for_decision(decision) -> str:
    """Return a CSS tone name for a decision."""
    if not isinstance(decision, str) or not decision.strip():
        return "neutral"
    decision_upper = decision.upper()
    if "KILL" in decision_upper:
        return "danger"
    if "ROLLOUT" in decision_upper or "SEGMENT" in decision_upper:
        return "info"
    if "SHIP" in decision_upper:
        return "success"
    if "BLOCK" in decision_upper:
        return "warning"
    return "neutral"


def symbol_for_status(value) -> str:
    """Return a readable status marker for pass/fail/unknown fields."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "WARN"
    status = str(value).strip().lower()
    if status in ("clean", "pass", "true", "ready", "yes"):
        return "PASS"
    if status in ("violated", "fail", "false", "no"):
        return "FAIL"
    return "WARN"


def tone_for_status(value) -> str:
    """Return a CSS tone name for generic status values."""
    marker = symbol_for_status(value)
    if marker == "PASS":
        return "success"
    if marker == "FAIL":
        return "danger"
    return "warning"


def format_probability(value) -> str:
    """Render raw float probabilities and preformatted strings consistently."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, str):
        return value if value.strip() else "N/A"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


def format_dcs(value) -> str:
    """Render a Decision Confidence Score."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return str(value)


def format_pvalue(value) -> str:
    """Render a p-value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
