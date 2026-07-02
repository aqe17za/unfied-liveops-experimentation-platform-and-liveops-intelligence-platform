"""Small shared helpers for LiveOps pages."""
import numpy as np


def simulate_weekly_trend(
    current_value: float,
    n_weeks: int = 8,
    volatility: float = 0.05,
    seed: int = 42
) -> list:
    """Simulate N weeks of historical data for trend charts."""
    np.random.seed(seed)
    values = [current_value]
    for _ in range(n_weeks - 1):
        change = np.random.normal(0, volatility * current_value)
        values.insert(0, max(0, values[0] + change))
    return values
