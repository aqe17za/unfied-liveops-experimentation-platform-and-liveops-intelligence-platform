"""Generic DuckDB read/write helpers."""
import duckdb
from pathlib import Path
import pandas as pd


def load_from_duckdb(db_path, table_name: str, limit: int | None = None) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"

    df = conn.execute(query).df()
    conn.close()
    return df


def save_to_duckdb(db_path, table_name: str, df: pd.DataFrame) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    conn.close()
    print(f"Saved {len(df)} rows to {table_name} in {db_path}")
