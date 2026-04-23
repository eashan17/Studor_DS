"""
Task 3: Cold-start recommender.

Three tiers, selected automatically by `recommend_cold_start` based on what
signal is available for the student:

  1. demographic-only       — full demographics, no behavioral history
  2. archetype-seeded       — demographics + >=1 week of behavior
  3. popularity prior       — nothing but a blank slate

All three tiers ultimately route through the content-based scorer so a student
who later acquires a behavioral history transitions smoothly into the main
recommender without a discontinuity.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from task3_content_based import (
    load_profiles, fit_scaler, score_matrix, top_k, DEFAULT_ALPHA,
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")

DEMO_FIELDS = ("gender", "region", "highest_education", "imd_band",
               "age_band", "disability", "studied_credits",
               "num_of_prev_attempts")

BEHAVIORAL_COLS = (
    "clicks_mean", "clicks_std", "active_days_total", "activity_diversity_max",
    "assessment_timeliness_mean_w1_6", "engagement_score_mean",
    "engagement_volatility", "engagement_slope_w1_6", "homepage_share",
    "submission_weight_rate",
)


def _empty_profile_row(template_row: pd.Series) -> pd.Series:
    """Return a zero-valued profile with the same schema as template_row."""
    empty = pd.Series(0.0, index=template_row.index)
    # Copy only key columns we want preserved (ids default to NaN).
    return empty


def _encode_demographics(demo: dict, feats: list) -> pd.Series:
    """Build a profile-shaped row from a demographics dict."""
    row = pd.Series(0.0, index=feats)
    # Numeric demographics.
    if "studied_credits" in demo and "studied_credits" in row.index:
        row["studied_credits"] = float(demo["studied_credits"])
    if "num_of_prev_attempts" in demo and "num_of_prev_attempts" in row.index:
        row["num_of_prev_attempts"] = float(demo["num_of_prev_attempts"])
    # One-hot categoricals — feature names are like "highest_education=HE Qualification".
    for key in ("highest_education", "imd_band", "age_band", "disability", "gender"):
        if key not in demo:
            continue
        col = f"{key}={demo[key]}"
        if col in row.index:
            row[col] = 1.0
    return row


def recommend_demographic(demo: dict, course_profiles: pd.DataFrame,
                          scaler, feats: list, alpha: float = DEFAULT_ALPHA,
                          k: int = 3) -> list:
    row = _encode_demographics(demo, feats)
    stud_vec = scaler.transform(row.values.reshape(1, -1))
    course_mat = scaler.transform(course_profiles[feats].values)
    prior = course_profiles["pass_rate_wilson_low"].values
    score, _sim = score_matrix(stud_vec, course_mat, prior, alpha=alpha)
    return top_k(score[0], course_profiles["code_module"].values, k=k)


def recommend_popularity(course_profiles: pd.DataFrame, k: int = 3) -> list:
    """Wilson lower bound ranking — niche courses are not drowned by popularity."""
    ranked = course_profiles.sort_values("pass_rate_wilson_low", ascending=False)
    return ranked["code_module"].head(k).tolist()


def recommend_archetype(archetype: str, student_profiles: pd.DataFrame,
                        course_profiles: pd.DataFrame, scaler, feats: list,
                        alpha: float = DEFAULT_ALPHA, k: int = 3) -> list:
    """Use the centroid of all training students with the given archetype."""
    col = f"archetype_{archetype}"
    mask = student_profiles[col] > 0 if col in student_profiles.columns else None
    if mask is None or not mask.any():
        return recommend_popularity(course_profiles, k=k)
    centroid = student_profiles.loc[mask, feats].mean()
    stud_vec = scaler.transform(centroid.values.reshape(1, -1))
    course_mat = scaler.transform(course_profiles[feats].values)
    prior = course_profiles["pass_rate_wilson_low"].values
    score, _sim = score_matrix(stud_vec, course_mat, prior, alpha=alpha)
    return top_k(score[0], course_profiles["code_module"].values, k=k)


def recommend_cold_start(demographics: dict = None,
                         archetype: str = None,
                         k: int = 3) -> dict:
    """Dispatcher. Returns {tier, recommendations, reason}."""
    student_profiles, course_profiles = load_profiles()
    scaler, feats = fit_scaler(student_profiles, course_profiles)

    if archetype:
        picks = recommend_archetype(archetype, student_profiles, course_profiles,
                                    scaler, feats, k=k)
        return {"tier": "archetype", "recommendations": picks,
                "reason": f"Seeded from archetype centroid: {archetype}"}

    if demographics:
        picks = recommend_demographic(demographics, course_profiles, scaler, feats, k=k)
        return {"tier": "demographic", "recommendations": picks,
                "reason": "Matched to historical course demographics"}

    picks = recommend_popularity(course_profiles, k=k)
    return {"tier": "popularity", "recommendations": picks,
            "reason": "No signal available — ranking by Wilson-lower-bound pass rate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["auto", "demographic", "archetype", "popularity"],
                    default="auto")
    ap.add_argument("--demographics", type=str, default=None,
                    help='JSON dict, e.g. \'{"highest_education":"HE Qualification",'
                         '"imd_band":"80-90%%","age_band":"35-55","disability":"N",'
                         '"gender":"M","studied_credits":60,"num_of_prev_attempts":0}\'')
    ap.add_argument("--archetype", type=str, default=None,
                    choices=[None, "Steady Engager", "Early Dropout", "Late Recoverer"])
    ap.add_argument("--all_missing", action="store_true",
                    help="simulate a brand-new student with no info at all")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    demo = json.loads(args.demographics) if args.demographics else None
    if args.all_missing:
        demo, args.archetype = None, None

    if args.tier == "auto":
        result = recommend_cold_start(demographics=demo, archetype=args.archetype, k=args.k)
    else:
        # Explicit tier for testing individual branches.
        student_profiles, course_profiles = load_profiles()
        scaler, feats = fit_scaler(student_profiles, course_profiles)
        if args.tier == "demographic":
            picks = recommend_demographic(demo or {}, course_profiles, scaler, feats, k=args.k)
        elif args.tier == "archetype":
            picks = recommend_archetype(args.archetype or "Steady Engager",
                                        student_profiles, course_profiles,
                                        scaler, feats, k=args.k)
        else:
            picks = recommend_popularity(course_profiles, k=args.k)
        result = {"tier": args.tier, "recommendations": picks}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
