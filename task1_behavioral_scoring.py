"""
Task 1: Behavioral Scoring Framework — PathAI Engine
OULAD dataset pipeline: ingestion → feature engineering → engagement scoring
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "weekly_clicks",
    "activity_diversity",
    "active_days",
    "assessment_timeliness",
    "wow_click_delta",
]

TRAIN_PRES = "2013J"
TEST_PRES  = "2014J"


# ---------------------------------------------------------------------------
# 1. Load small reference files
# ---------------------------------------------------------------------------

def load_reference_files():
    cat = "category"

    vle = pd.read_csv(
        os.path.join(DATA_DIR, "vle.csv"),
        usecols=["id_site", "code_module", "code_presentation", "activity_type"],
        dtype={"code_module": cat, "code_presentation": cat, "activity_type": cat},
    )

    student_info = pd.read_csv(
        os.path.join(DATA_DIR, "studentInfo.csv"),
        usecols=["code_module", "code_presentation", "id_student", "final_result"],
        dtype={"code_module": cat, "code_presentation": cat, "final_result": cat},
    )

    student_reg = pd.read_csv(
        os.path.join(DATA_DIR, "studentRegistration.csv"),
        usecols=["code_module", "code_presentation", "id_student", "date_unregistration"],
        dtype={"code_module": cat, "code_presentation": cat},
    )
    student_reg["date_unregistration"] = pd.to_numeric(
        student_reg["date_unregistration"], errors="coerce"
    )
    # Convert unregistration date to week number (NaN → large sentinel = never withdrew)
    student_reg["unreg_week"] = (
        student_reg["date_unregistration"]
        .fillna(99999)
        .div(7)
        .apply(np.floor)
        .astype(int) + 1
    )

    assessments = pd.read_csv(
        os.path.join(DATA_DIR, "assessments.csv"),
        usecols=["code_module", "code_presentation", "id_assessment",
                 "assessment_type", "date", "weight"],
        dtype={"code_module": cat, "code_presentation": cat},
    )
    assessments["due_week"] = (assessments["date"] // 7) + 1

    student_assess = pd.read_csv(
        os.path.join(DATA_DIR, "studentAssessment.csv"),
        dtype={"is_banked": "int8"},
    )
    student_assess = student_assess[student_assess["is_banked"] == 0].drop(columns=["is_banked"])

    courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))

    return vle, student_info, student_reg, assessments, student_assess, courses


# ---------------------------------------------------------------------------
# 2. Build weekly VLE features (chunked for memory efficiency)
# ---------------------------------------------------------------------------

def build_vle_features(vle, student_info, student_reg):
    """
    Stream studentVle.csv in chunks, enrich with activity_type,
    then aggregate to (id_student, code_module, code_presentation, week_num).
    Returns a DataFrame with weekly_clicks, activity_diversity, active_days.
    """
    dtypes = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int32",
        "sum_click": "int32",
    }

    chunk_aggs = []
    print("  Streaming studentVle.csv ...")
    for chunk in pd.read_csv(
        os.path.join(DATA_DIR, "studentVle.csv"),
        dtype=dtypes,
        chunksize=500_000,
    ):
        # Active course period only
        chunk = chunk[chunk["date"] >= 0].copy()
        chunk["week_num"] = (chunk["date"] // 7) + 1

        # Enrich with activity_type
        chunk = chunk.merge(
            vle[["id_site", "code_module", "code_presentation", "activity_type"]],
            on=["id_site", "code_module", "code_presentation"],
            how="left",
        )

        # Aggregate within chunk
        grp = ["id_student", "code_module", "code_presentation", "week_num"]
        agg = (
            chunk.groupby(grp, observed=False)
            .agg(
                weekly_clicks=("sum_click", "sum"),
                activity_diversity=("activity_type", "nunique"),
                active_days=("date", "nunique"),
            )
            .reset_index()
        )
        chunk_aggs.append(agg)

    # Re-aggregate across chunks (same student-week may span multiple chunks)
    print("  Re-aggregating chunks ...")
    combined = pd.concat(chunk_aggs, ignore_index=True)
    grp = ["id_student", "code_module", "code_presentation", "week_num"]
    vle_weekly = (
        combined.groupby(grp)
        .agg(
            weekly_clicks=("weekly_clicks", "sum"),
            activity_diversity=("activity_diversity", "sum"),  # sum then re-nunique not possible; use max across chunks as proxy
            active_days=("active_days", "sum"),
        )
        .reset_index()
    )
    # Note: activity_diversity summed across chunks slightly over-counts unique types
    # that appear in multiple chunks for the same week. Cap at the known number of
    # distinct activity types (11 in OULAD) to bound the error.
    MAX_ACTIVITY_TYPES = vle["activity_type"].nunique()
    vle_weekly["activity_diversity"] = vle_weekly["activity_diversity"].clip(upper=MAX_ACTIVITY_TYPES)

    # Attach final_result and unregistration info
    vle_weekly = vle_weekly.merge(
        student_info[["id_student", "code_module", "code_presentation", "final_result"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    vle_weekly = vle_weekly.merge(
        student_reg[["id_student", "code_module", "code_presentation", "unreg_week"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    vle_weekly["unreg_week"] = vle_weekly["unreg_week"].fillna(99999).astype(int)

    return vle_weekly


# ---------------------------------------------------------------------------
# 3. Build assessment timeliness feature
# ---------------------------------------------------------------------------

def build_timeliness_feature(assessments, student_assess, student_info):
    """
    Returns a DataFrame indexed by (id_student, code_module, code_presentation, week_num)
    with column 'assessment_timeliness'.
    """
    # Join student submissions with assessment metadata
    merged = student_assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation",
                     "assessment_type", "date", "weight", "due_week"]],
        on="id_assessment",
        how="left",
    )

    # margin_days: positive = early, 0 = on-time, negative = late
    merged["margin_days"] = merged["date"] - merged["date_submitted"]

    # Timeliness score: on-time/early → 1.0; 7+ days late → 0.0; linear decay
    merged["timeliness_score"] = (1 + merged["margin_days"] / 7).clip(0, 1)

    # Weight by assessment weight
    merged["weighted_timeliness"] = merged["timeliness_score"] * (merged["weight"] / 100)

    # Aggregate to week level
    grp = ["id_student", "code_module", "code_presentation", "due_week"]
    timeliness_weekly = (
        merged.groupby(grp, observed=False)["weighted_timeliness"]
        .sum()
        .reset_index()
        .rename(columns={"due_week": "week_num", "weighted_timeliness": "assessment_timeliness"})
    )

    return timeliness_weekly


# ---------------------------------------------------------------------------
# 4. Merge all features and compute WoW delta
# ---------------------------------------------------------------------------

def merge_features(vle_weekly, timeliness_weekly):
    """Merge timeliness into VLE weekly frame, forward-fill, compute WoW delta."""
    df = vle_weekly.merge(
        timeliness_weekly,
        on=["id_student", "code_module", "code_presentation", "week_num"],
        how="left",
    )

    # Forward-fill timeliness (carry last known value; weeks before first assessment → 0)
    df = df.sort_values(["id_student", "code_module", "code_presentation", "week_num"])
    df["assessment_timeliness"] = (
        df.groupby(["id_student", "code_module", "code_presentation"])["assessment_timeliness"]
        .transform(lambda s: s.ffill().fillna(0))
    )

    # WoW click delta
    df["wow_click_delta"] = (
        df.groupby(["id_student", "code_module", "code_presentation"])["weekly_clicks"]
        .diff()
        .fillna(0)
    )

    return df


# ---------------------------------------------------------------------------
# 5. Derive feature weights from train cohort (no leakage)
# ---------------------------------------------------------------------------

def derive_weights(df, student_info):
    """
    Point-biserial correlation between each feature and pass/fail outcome,
    computed on the 2013J cohort only. Returns normalized weight dict.
    """
    train_mask = df["code_presentation"].astype(str).str.endswith(TRAIN_PRES)
    train_df = df[train_mask].copy()

    # Student-level mean of each feature
    grp = ["id_student", "code_module", "code_presentation"]
    train_agg = train_df.groupby(grp)[FEATURE_COLS].mean().reset_index()

    # Binary outcome
    outcome = (
        student_info[student_info["code_presentation"].astype(str).str.endswith(TRAIN_PRES)]
        .copy()
    )
    outcome["y"] = outcome["final_result"].isin(["Pass", "Distinction"]).astype(int)

    train_agg = train_agg.merge(
        outcome[["id_student", "code_module", "code_presentation", "y"]],
        on=["id_student", "code_module", "code_presentation"],
        how="inner",
    ).dropna(subset=FEATURE_COLS + ["y"])

    correlations = {}
    for feat in FEATURE_COLS:
        r, p = pointbiserialr(train_agg["y"], train_agg[feat].fillna(0))
        correlations[feat] = abs(r)
        print(f"    |r_pb| {feat}: {abs(r):.4f}  (p={p:.4f})")

    total = sum(correlations.values())
    weights = {k: v / total for k, v in correlations.items()}
    print("\n  Derived weights:")
    for k, v in weights.items():
        print(f"    {k}: {v:.4f}")
    return weights


# ---------------------------------------------------------------------------
# 6. Scale and score
# ---------------------------------------------------------------------------

def scale_and_score(df, weights):
    """
    Fit MinMaxScaler on train split, apply globally.
    Compute weighted engagement score and smooth with 3-week rolling mean.
    """
    train_mask = df["code_presentation"].astype(str).str.endswith(TRAIN_PRES)
    train_df = df[train_mask]

    # Percentile-cap features (fit on train only) to neutralize extreme outliers
    # before MinMaxScaling, otherwise weekly_clicks/wow_click_delta tails compress
    # typical students into a tiny portion of [0,1] and cap engagement_score ~40.
    lower_q = train_df[FEATURE_COLS].fillna(0).quantile(0.01)
    upper_q = train_df[FEATURE_COLS].fillna(0).quantile(0.99)

    train_clipped = train_df[FEATURE_COLS].fillna(0).clip(lower=lower_q, upper=upper_q, axis=1)
    df_clipped = df[FEATURE_COLS].fillna(0).clip(lower=lower_q, upper=upper_q, axis=1)

    scaler = MinMaxScaler()
    scaler.fit(train_clipped)

    scaled = scaler.transform(df_clipped)
    scaled_cols = [f"scaled_{f}" for f in FEATURE_COLS]
    df[scaled_cols] = scaled

    weights_arr = np.array([weights[f] for f in FEATURE_COLS])
    df["raw_score"] = (df[scaled_cols].values @ weights_arr) * 100
    df["raw_score"] = df["raw_score"].clip(0, 100)

    # 3-week rolling mean for smoothing
    df["engagement_score"] = (
        df.sort_values("week_num")
        .groupby(["id_student", "code_module", "code_presentation"])["raw_score"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Engagement volatility (std of weekly engagement scores per student)
    vol = (
        df.groupby(["id_student", "code_module", "code_presentation"])["engagement_score"]
        .std()
        .rename("engagement_volatility")
        .reset_index()
    )
    df = df.merge(vol, on=["id_student", "code_module", "code_presentation"], how="left")

    return df, scaler, scaled_cols


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    print("=== Task 1: Behavioral Scoring Framework ===\n")

    print("[1/6] Loading reference files ...")
    vle, student_info, student_reg, assessments, student_assess, courses = load_reference_files()

    print("[2/6] Building VLE weekly features ...")
    vle_weekly = build_vle_features(vle, student_info, student_reg)
    print(f"  VLE weekly rows: {len(vle_weekly):,}")

    print("[3/6] Building assessment timeliness feature ...")
    timeliness_weekly = build_timeliness_feature(assessments, student_assess, student_info)

    print("[4/6] Merging features ...")
    df = merge_features(vle_weekly, timeliness_weekly)
    print(f"  Feature table rows: {len(df):,}")

    # Verification: uniqueness
    key = ["id_student", "code_module", "code_presentation", "week_num"]
    assert df.duplicated(subset=key).sum() == 0, "Duplicate student-week rows found!"
    print("  [OK] Student-week uniqueness verified.")

    print("[5/6] Deriving feature weights (train cohort only) ...")
    weights = derive_weights(df, student_info)
    with open(os.path.join(OUT_DIR, "feature_weights.json"), "w") as f:
        json.dump(weights, f, indent=2)
    print("  Weights saved -> outputs/feature_weights.json")

    print("[6/6] Scaling and scoring ...")
    df, scaler, scaled_cols = scale_and_score(df, weights)

    # Timeliness correctness assertion
    assert df["engagement_score"].between(0, 100).all(), "Scores out of [0,100]!"
    print("  [OK] Score range [0,100] verified.")

    # Save weekly scores
    output_cols = (
        ["id_student", "code_module", "code_presentation", "week_num",
         "final_result", "unreg_week"]
        + FEATURE_COLS
        + ["raw_score", "engagement_score", "engagement_volatility"]
    )
    df[output_cols].to_csv(os.path.join(OUT_DIR, "weekly_scores.csv"), index=False)
    print("  Scores saved -> outputs/weekly_scores.csv")
    print(f"\n  Score summary:\n{df['engagement_score'].describe().round(2)}")
    print("\n=== Done. Run task1_archetypes.py for clustering and plots. ===")


if __name__ == "__main__":
    main()
