import numpy as np
import pandas as pd
from data.synthetic import CLASSES_OF_BUSINESS, INDUSTRIES, STATES, generate
from carriers import CARRIERS
from config import get_config

ALL_FEATURES = ["class_of_business", "industry", "state", "premium", "loss_ratio",
                "years_in_business", "prior_claims", "num_employees"]

ALL_STATES = set(STATES)


def train_all():
    pass  # config-driven scoring needs no pre-training


def _raw_score(features: dict, carrier_id: str) -> float:
    cfg = get_config(carrier_id)
    prob = 0.50

    line_pref = cfg["lines"].get(features["class_of_business"], "neutral")
    if line_pref == "preferred":
        prob += 0.20
    elif line_pref == "avoid":
        prob -= 0.30

    ind_pref = cfg["industries"].get(features["industry"], "neutral")
    if ind_pref == "preferred":
        prob += 0.22
    elif ind_pref == "avoid":
        prob -= 0.32

    if features["loss_ratio"] > cfg["max_loss_ratio"]:
        prob -= 0.28
    elif features["loss_ratio"] < 0.30:
        prob += 0.10

    lo, hi = cfg["premium_min"], cfg["premium_max"]
    if lo <= features["premium"] <= hi:
        prob += 0.12
    elif features["premium"] > hi * 2:
        prob -= 0.18

    if features["years_in_business"] < cfg["min_years_in_business"]:
        prob -= 0.20
    elif features["years_in_business"] > 10:
        prob += 0.08

    if features["prior_claims"] > 3:
        prob -= 0.22
    elif features["prior_claims"] == 0:
        prob += 0.08

    if features["state"] in cfg.get("high_risk_states", []):
        prob -= 0.10

    return float(np.clip(prob, 0.05, 0.95))


def _guidance(features: dict, carrier_id: str) -> dict:
    cfg = get_config(carrier_id)
    strengths, flags = [], []

    cob = features["class_of_business"]
    line_pref = cfg["lines"].get(cob, "neutral")
    if line_pref == "preferred":
        strengths.append(f"{cob.replace('_', ' ').title()} is a preferred line for this carrier")
    elif line_pref == "avoid":
        flags.append(f"{cob.replace('_', ' ').title()} is outside their current appetite")

    ind = features["industry"]
    ind_pref = cfg["industries"].get(ind, "neutral")
    if ind_pref == "preferred":
        strengths.append(f"{ind.title()} is a target industry class")
    elif ind_pref == "avoid":
        flags.append(f"{ind.title()} industry is currently avoided by this carrier")

    lr = features["loss_ratio"]
    if lr > cfg["max_loss_ratio"]:
        flags.append(f"Loss ratio ({lr:.0%}) exceeds their {cfg['max_loss_ratio']:.0%} threshold")
    elif lr < 0.30:
        strengths.append(f"Clean loss history ({lr:.0%}) is well below their threshold")

    lo, hi = cfg["premium_min"], cfg["premium_max"]
    premium = features["premium"]
    if lo <= premium <= hi:
        strengths.append(f"Premium (${premium:,.0f}) fits their target range")
    elif premium > hi:
        flags.append(f"Premium (${premium:,.0f}) is above their sweet spot (up to ${hi:,})")

    if features["state"] in cfg.get("high_risk_states", []):
        flags.append(f"{features['state']} is a higher-scrutiny state for this carrier")

    yib = features["years_in_business"]
    min_yib = cfg["min_years_in_business"]
    if yib < min_yib:
        flags.append(f"Only {yib} year(s) in business — carrier prefers at least {min_yib}")
    elif yib > 10:
        strengths.append(f"{yib} years in business signals stability")

    if features["prior_claims"] > 3:
        flags.append(f"{features['prior_claims']} prior claims will require explanation")
    elif features["prior_claims"] == 0:
        strengths.append("No prior claims — strong submission signal")

    return {"strengths": strengths[:3], "flags": flags[:3]}


def _commission(features: dict, carrier_id: str) -> dict:
    carrier = CARRIERS[carrier_id]
    rate = carrier["commission_rates"].get(features["class_of_business"], carrier["base_commission"])
    return {"rate": rate, "amount": round(features["premium"] * rate)}


def score_for_carrier(features: dict, carrier_id: str) -> dict:
    prob = _raw_score(features, carrier_id)
    score = round(prob * 100)
    carrier = CARRIERS[carrier_id]

    if score >= 70:
        signal, summary = "strong_fit", f"This risk is a strong fit for {carrier['name']}'s appetite."
    elif score >= 45:
        signal, summary = "marginal", f"This risk is in range for {carrier['name']} but may need additional support."
    else:
        signal, summary = "poor_fit", f"This risk falls outside {carrier['name']}'s current appetite."

    guidance = _guidance(features, carrier_id)
    commission = _commission(features, carrier_id)

    return {
        "carrier_id": carrier_id,
        "carrier_name": carrier["name"],
        "carrier_tagline": carrier["tagline"],
        "score": score,
        "signal": signal,
        "summary": summary,
        "strengths": guidance["strengths"],
        "flags": guidance["flags"],
        "commission_rate": commission["rate"],
        "commission_amount": commission["amount"],
    }


def score_all_carriers(features: dict) -> list[dict]:
    return sorted(
        [score_for_carrier(features, cid) for cid in CARRIERS],
        key=lambda x: -x["score"]
    )


def get_agency_stats() -> list[dict]:
    appetite = list(CARRIERS.values())[0]["appetite"]
    df = generate(n=3000, seed=99, appetite=appetite)
    stats = (
        df.groupby("agency")
        .agg(total_submissions=("accepted", "count"), accepted=("accepted", "sum"), avg_premium=("premium", "mean"))
        .reset_index()
    )
    stats["hit_rate"] = (stats["accepted"] / stats["total_submissions"] * 100).round(1)
    stats["avg_premium"] = stats["avg_premium"].round(0).astype(int)
    return stats.drop(columns=["accepted"]).sort_values("total_submissions", ascending=False).to_dict(orient="records")
