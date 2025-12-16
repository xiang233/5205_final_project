# src/apportionment/utils.py

import pandas as pd


def load_census_data(path: str) -> pd.Series:
    """
    Load census data from a CSV and return a Series mapping state -> population.

    Expected columns (you can adjust to your actual file):
      - 'state' (or 'State')
      - 'population' (or 'Population')
    """
    df = pd.read_csv(path)

    # Try a couple of reasonable column names
    state_cols = [c for c in df.columns if c.lower() in ("state", "name", "state_name")]
    pop_cols = [c for c in df.columns if "pop" in c.lower()]

    if not state_cols or not pop_cols:
        raise ValueError(
            f"Could not find suitable state/population columns in {df.columns.tolist()}"
        )

    state_col = state_cols[0]
    pop_col = pop_cols[0]

    df = df[[state_col, pop_col]].copy()
    df.columns = ["state", "population"]

    df["population"] = df["population"].astype(float)
    df = df.sort_values("state")

    return df.set_index("state")["population"]


def normalize_population(pop: pd.Series) -> pd.Series:
    """
    Ensure populations are non-negative floats.
    """
    pop = pop.astype(float)
    if (pop < 0).any():
        raise ValueError("Population cannot be negative.")
    return pop


def compute_representation_ratios(pop: pd.Series, seats: pd.Series) -> pd.Series:
    """
    ρ_i = (P_i / s_i) / (total_population / total_seats)
    """
    if not pop.index.equals(seats.index):
        seats = seats.reindex(pop.index)

    total_pop = pop.sum()
    total_seats = seats.sum()

    avg_constituents = total_pop / total_seats
    per_member = pop / seats
    return per_member / avg_constituents


def fairness_deviation(ratios: pd.Series) -> float:
    """
    FD = sum_i |ρ_i - 1|
    """
    return (ratios - 1.0).abs().sum()
