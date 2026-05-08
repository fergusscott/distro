"""
Generates synthetic book-of-business data for a carrier.
In production this would come from the carrier's actual policy data.
"""
import numpy as np
from data.synthetic import generate, CLASSES_OF_BUSINESS, INDUSTRIES
from carriers import CARRIERS


def get_book(carrier_id: str) -> dict:
    appetite = CARRIERS[carrier_id]["appetite"]
    seed = abs(hash(carrier_id + "book")) % (2**31)
    df = generate(n=1000, seed=seed, appetite=appetite)
    bound = df[df["accepted"] == 1].copy()

    total = len(bound)
    total_premium = float(bound["premium"].sum())

    by_line = {}
    for line in CLASSES_OF_BUSINESS:
        subset = bound[bound["class_of_business"] == line]
        by_line[line] = {
            "count": int(len(subset)),
            "pct": round(len(subset) / total * 100, 1) if total else 0,
            "premium": int(subset["premium"].sum()),
        }

    by_industry = {}
    for ind in INDUSTRIES:
        subset = bound[bound["industry"] == ind]
        by_industry[ind] = {
            "count": int(len(subset)),
            "pct": round(len(subset) / total * 100, 1) if total else 0,
            "premium": int(subset["premium"].sum()),
        }

    total_submissions = len(df)
    bind_rate = round(total / total_submissions * 100, 1) if total_submissions else 0
    avg_premium = int(bound["premium"].mean()) if total else 0

    return {
        "total_bound": total,
        "total_submissions": total_submissions,
        "bind_rate": bind_rate,
        "total_premium": int(total_premium),
        "avg_premium": avg_premium,
        "by_line": by_line,
        "by_industry": by_industry,
    }
