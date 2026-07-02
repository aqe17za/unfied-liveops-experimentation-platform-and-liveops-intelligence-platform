"""Generic CSV read/write helpers."""
import pandas as pd
from pathlib import Path


def load_from_csv(csv_path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def save_to_csv(csv_path, df: pd.DataFrame) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")
