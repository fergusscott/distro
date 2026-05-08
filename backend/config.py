import json
import os
from data.synthetic import CLASSES_OF_BUSINESS, INDUSTRIES, STATES
from carriers import CARRIERS

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "data", "configs")


def _default_config(carrier_id: str) -> dict:
    appetite = CARRIERS[carrier_id]["appetite"]
    lo, hi = appetite.get("preferred_premium_range", (5000, 250000))
    return {
        "lines": {
            line: (
                "preferred" if line in appetite.get("preferred_lines", [])
                else "avoid" if line in appetite.get("avoid_lines", [])
                else "neutral"
            )
            for line in CLASSES_OF_BUSINESS
        },
        "industries": {
            ind: (
                "preferred" if ind in appetite.get("preferred_industries", [])
                else "avoid" if ind in appetite.get("avoid_industries", [])
                else "neutral"
            )
            for ind in INDUSTRIES
        },
        "max_loss_ratio": appetite.get("max_loss_ratio", 0.6),
        "premium_min": lo,
        "premium_max": hi,
        "min_years_in_business": appetite.get("min_years_in_business", 1),
        "high_risk_states": sorted(appetite.get("high_risk_states", set())),
    }


def get_config(carrier_id: str) -> dict:
    path = os.path.join(CONFIG_DIR, f"{carrier_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _default_config(carrier_id)


def save_config(carrier_id: str, config: dict) -> dict:
    path = os.path.join(CONFIG_DIR, f"{carrier_id}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return config


def reset_config(carrier_id: str) -> dict:
    path = os.path.join(CONFIG_DIR, f"{carrier_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return _default_config(carrier_id)
