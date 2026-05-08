import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import os

from data.synthetic import generate, CLASSES_OF_BUSINESS, INDUSTRIES, STATES
from carriers import CARRIERS

CATEGORICAL_FEATURES = ["class_of_business", "industry", "state"]
NUMERIC_FEATURES = ["premium", "loss_ratio", "years_in_business", "prior_claims", "num_employees"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "data")

_model_cache: dict = {}


def _model_path(carrier_id: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{carrier_id}.pkl")


def _build_pipeline() -> Pipeline:
    cat_transformer = OrdinalEncoder(
        categories=[CLASSES_OF_BUSINESS, INDUSTRIES, STATES],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    preprocessor = ColumnTransformer([
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
    ])


def _train(carrier_id: str) -> Pipeline:
    appetite = CARRIERS[carrier_id]["appetite"]
    df = generate(n=3000, seed=hash(carrier_id) % (2**31), appetite=appetite)
    X, y = df[ALL_FEATURES], df["accepted"]
    pipeline = _build_pipeline()
    pipeline.fit(X, y)
    with open(_model_path(carrier_id), "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


def _load(carrier_id: str) -> Pipeline:
    if carrier_id in _model_cache:
        return _model_cache[carrier_id]
    path = _model_path(carrier_id)
    if os.path.exists(path):
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
    else:
        pipeline = _train(carrier_id)
    _model_cache[carrier_id] = pipeline
    return pipeline


def train_all():
    for carrier_id in CARRIERS:
        _train(carrier_id)


def _guidance(features: dict, carrier_id: str, score: int) -> dict:
    appetite = CARRIERS[carrier_id]["appetite"]
    strengths, flags = [], []

    cob = features["class_of_business"]
    if cob in appetite.get("preferred_lines", []):
        strengths.append(f"{cob.replace('_', ' ').title()} is a core line for this carrier")
    elif cob in appetite.get("avoid_lines", []):
        flags.append(f"{cob.replace('_', ' ').title()} is outside their primary appetite")

    industry = features["industry"]
    if industry in appetite.get("preferred_industries", []):
        strengths.append(f"{industry.title()} is a preferred industry class")
    elif industry in appetite.get("avoid_industries", []):
        flags.append(f"{industry.title()} industry is generally avoided by this carrier")

    lr = features["loss_ratio"]
    max_lr = appetite.get("max_loss_ratio", 0.6)
    if lr > max_lr:
        flags.append(f"Loss ratio ({lr:.0%}) exceeds their {max_lr:.0%} threshold — expect scrutiny")
    elif lr < 0.3:
        strengths.append(f"Clean loss history ({lr:.0%}) is well below their threshold")

    lo, hi = appetite.get("preferred_premium_range", (0, 999999))
    premium = features["premium"]
    if lo <= premium <= hi:
        strengths.append(f"Premium size (${premium:,.0f}) fits squarely in their target range")
    elif premium > hi:
        flags.append(f"Premium (${premium:,.0f}) exceeds their typical sweet spot (up to ${hi:,})")

    if features["state"] in appetite.get("high_risk_states", set()):
        flags.append(f"{features['state']} is a higher-scrutiny state for this carrier")

    yib = features["years_in_business"]
    min_yib = appetite.get("min_years_in_business", 1)
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
    rate = carrier["commission_rates"].get(
        features["class_of_business"], carrier["base_commission"]
    )
    amount = round(features["premium"] * rate)
    return {"rate": rate, "amount": amount}


def score_for_carrier(features: dict, carrier_id: str) -> dict:
    pipeline = _load(carrier_id)
    row = pd.DataFrame([{k: features[k] for k in ALL_FEATURES}])
    prob = float(pipeline.predict_proba(row)[0][1])
    score = round(prob * 100)

    carrier = CARRIERS[carrier_id]
    if score >= 70:
        signal = "strong_fit"
        summary = f"This risk is a strong fit for {carrier['name']}'s appetite."
    elif score >= 45:
        signal = "marginal"
        summary = f"This risk is in range for {carrier['name']} but may require additional underwriting support."
    else:
        signal = "poor_fit"
        summary = f"This risk falls outside {carrier['name']}'s current appetite in several key areas."

    guidance = _guidance(features, carrier_id, score)
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
    results = [score_for_carrier(features, cid) for cid in CARRIERS]
    return sorted(results, key=lambda x: -x["score"])


def get_agency_stats() -> list[dict]:
    from data.synthetic import generate as gen
    appetite = list(CARRIERS.values())[0]["appetite"]
    df = gen(n=3000, seed=99, appetite=appetite)
    stats = (
        df.groupby("agency")
        .agg(total_submissions=("accepted", "count"), accepted=("accepted", "sum"), avg_premium=("premium", "mean"))
        .reset_index()
    )
    stats["hit_rate"] = (stats["accepted"] / stats["total_submissions"] * 100).round(1)
    stats["avg_premium"] = stats["avg_premium"].round(0).astype(int)
    return stats.drop(columns=["accepted"]).sort_values("total_submissions", ascending=False).to_dict(orient="records")
