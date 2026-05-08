import json
import os
from data.synthetic import CLASSES_OF_BUSINESS, INDUSTRIES

GOALS_DIR = os.path.join(os.path.dirname(__file__), "data", "configs")
os.makedirs(GOALS_DIR, exist_ok=True)


def _default_goals() -> dict:
    return {
        "line_targets": {line: 0 for line in CLASSES_OF_BUSINESS},
        "industry_targets": {ind: 0 for ind in INDUSTRIES},
        "target_bind_rate": 0,
        "target_premium_volume": 0,
    }


def get_goals(carrier_id: str) -> dict:
    path = os.path.join(GOALS_DIR, f"goals_{carrier_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _default_goals()


def save_goals(carrier_id: str, goals: dict) -> dict:
    path = os.path.join(GOALS_DIR, f"goals_{carrier_id}.json")
    with open(path, "w") as f:
        json.dump(goals, f, indent=2)
    return goals


def generate_recommendations(carrier_id: str, book: dict, goals: dict) -> list[dict]:
    recs = []

    for line, target in goals.get("line_targets", {}).items():
        if target == 0:
            continue
        current = book["by_line"].get(line, {}).get("pct", 0)
        gap = target - current
        label = line.replace("_", " ").title()
        if gap >= 8:
            recs.append({
                "dimension": "lines",
                "key": line,
                "label": label,
                "action": "preferred",
                "current_pct": current,
                "target_pct": target,
                "gap": round(gap, 1),
                "direction": "under",
                "reason": f"{label} is {current:.0f}% of your book but your target is {target:.0f}% — marking as Preferred will attract more of this business from brokers.",
            })
        elif gap <= -8:
            recs.append({
                "dimension": "lines",
                "key": line,
                "label": label,
                "action": "avoid",
                "current_pct": current,
                "target_pct": target,
                "gap": round(gap, 1),
                "direction": "over",
                "reason": f"{label} is {current:.0f}% of your book but your target is only {target:.0f}% — marking as Avoid will signal lower appetite to brokers.",
            })

    for ind, target in goals.get("industry_targets", {}).items():
        if target == 0:
            continue
        current = book["by_industry"].get(ind, {}).get("pct", 0)
        gap = target - current
        label = ind.replace("_", " ").title()
        if gap >= 8:
            recs.append({
                "dimension": "industries",
                "key": ind,
                "label": label,
                "action": "preferred",
                "current_pct": current,
                "target_pct": target,
                "gap": round(gap, 1),
                "direction": "under",
                "reason": f"{label} is {current:.0f}% of your book but your target is {target:.0f}% — marking as Preferred will attract more of this segment.",
            })
        elif gap <= -8:
            recs.append({
                "dimension": "industries",
                "key": ind,
                "label": label,
                "action": "avoid",
                "current_pct": current,
                "target_pct": target,
                "gap": round(gap, 1),
                "direction": "over",
                "reason": f"{label} is {current:.0f}% of your book but your target is only {target:.0f}% — marking as Avoid will reduce flow in this segment.",
            })

    return sorted(recs, key=lambda r: -abs(r["gap"]))
