import numpy as np
import pandas as pd

CLASSES_OF_BUSINESS = [
    "general_liability", "commercial_property", "professional_liability",
    "workers_comp", "commercial_auto", "cyber", "management_liability"
]

INDUSTRIES = [
    "construction", "manufacturing", "retail", "technology",
    "healthcare", "hospitality", "real_estate", "financial_services", "transportation"
]

STATES = ["CA", "TX", "FL", "NY", "IL", "GA", "OH", "PA", "AZ", "WA"]

AGENCIES = [f"Agency_{i:03d}" for i in range(1, 31)]


def _acceptance_probability(row: dict, appetite: dict) -> float:
    prob = 0.5

    if row["class_of_business"] in appetite.get("preferred_lines", []):
        prob += 0.20
    if row["class_of_business"] in appetite.get("avoid_lines", []):
        prob -= 0.30

    if row["industry"] in appetite.get("preferred_industries", []):
        prob += 0.22
    if row["industry"] in appetite.get("avoid_industries", []):
        prob -= 0.32

    lo, hi = appetite.get("preferred_premium_range", (0, 999999))
    if lo <= row["premium"] <= hi:
        prob += 0.12
    elif row["premium"] > hi * 2:
        prob -= 0.18

    if row["loss_ratio"] > appetite.get("max_loss_ratio", 0.6):
        prob -= 0.28
    elif row["loss_ratio"] < 0.3:
        prob += 0.10

    if row["state"] in appetite.get("high_risk_states", set()):
        prob -= 0.10

    min_yib = appetite.get("min_years_in_business", 1)
    if row["years_in_business"] < min_yib:
        prob -= 0.20
    elif row["years_in_business"] > 10:
        prob += 0.08

    if row["prior_claims"] > 3:
        prob -= 0.22
    elif row["prior_claims"] == 0:
        prob += 0.08

    return float(np.clip(prob, 0.05, 0.95))


def generate(n: int = 2000, seed: int = 42, appetite: dict | None = None) -> pd.DataFrame:
    from carriers import CARRIERS
    if appetite is None:
        appetite = list(CARRIERS.values())[0]["appetite"]

    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n):
        row = dict(
            class_of_business=str(rng.choice(CLASSES_OF_BUSINESS)),
            industry=str(rng.choice(INDUSTRIES)),
            state=str(rng.choice(STATES)),
            premium=float(rng.lognormal(mean=10.5, sigma=1.2)),
            loss_ratio=float(rng.beta(a=3, b=5)),
            years_in_business=int(rng.integers(1, 40)),
            prior_claims=int(rng.poisson(lam=1.2)),
            num_employees=int(rng.lognormal(mean=4, sigma=1.5)),
            agency=str(rng.choice(AGENCIES)),
        )
        p = _acceptance_probability(row, appetite)
        row["accepted"] = int(rng.random() < p)
        records.append(row)

    return pd.DataFrame(records)
