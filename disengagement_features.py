"""
Task 2: Predictive Disengagement — Feature Engineering

Reads outputs/weekly_scores.csv (produced by task1_behavioral_scoring.py) plus
studentInfo.csv and studentRegistration.csv, then builds a Week-6 feature matrix
for the binary classifier.

Leakage contract: every feature in this file is derivable from data observable
on day 42 (end of Week 6) or earlier. Students who unregistered on or before
day 42 are excluded — they are surfaced to advisors by a separate simple rule,
not by the ML model.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_WEEK = 6
CUTOFF_DAY = CUTOFF_WEEK * 7  # 42

KEY = ["id_student", "code_module", "code_presentation"]

# Activity-type buckets for per-type click features. We collapse the long tail of
# rare activity types into semantically meaningful groups so the model gets useful
# signal rather than a bag of sparse columns.
ACTIVITY_BUCKETS = {
    "quiz":     {"quiz", "externalquiz", "questionnaire", "dataplus"},
    "content":  {"oucontent", "subpage", "page", "htmlactivity", "sharedsubpage", "dualpane"},
    "resource": {"resource", "folder", "repeatactivity"},
    "forum":    {"forumng"},
    "collab":   {"oucollaborate", "ouelluminate", "ouwiki", "glossary"},
    "url":      {"url"},
    "homepage": {"homepage"},
}


def load_inputs():
    scores = pd.read_csv(os.path.join(OUT_DIR, "weekly_scores.csv"))
    student_info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"))
    student_reg = pd.read_csv(os.path.join(DATA_DIR, "studentRegistration.csv"))
    student_reg["date_registration"] = pd.to_numeric(
        student_reg["date_registration"], errors="coerce"
    )
    student_reg["date_unregistration"] = pd.to_numeric(
        student_reg["date_unregistration"], errors="coerce"
    )
    return scores, student_info, student_reg


# ---------------------------------------------------------------------------
# Raw-VLE-level features: per-activity-type clicks, recency, inactivity streaks
# ---------------------------------------------------------------------------

def _bucket_for(activity_type):
    for bucket, members in ACTIVITY_BUCKETS.items():
        if activity_type in members:
            return bucket
    return "other"


def build_raw_vle_features():
    """Stream studentVle.csv filtered to date <= 42, produce per-student features
    that weekly_scores.csv doesn't already capture:
      - clicks_<bucket>  per activity-type bucket (quiz, content, forum, ...)
      - days_since_last_click (recency as of day 42; higher = more stale)
      - longest_inactive_streak (max run of consecutive inactive days within days 0-42)
    """
    vle = pd.read_csv(
        os.path.join(DATA_DIR, "vle.csv"),
        usecols=["id_site", "code_module", "code_presentation", "activity_type"],
    )
    vle["bucket"] = vle["activity_type"].map(_bucket_for).fillna("other")
    vle_lookup = vle[["id_site", "code_module", "code_presentation", "bucket"]]

    dtypes = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int32",
        "sum_click": "int32",
    }

    per_bucket_parts = []
    day_activity_parts = []

    print("  Streaming studentVle.csv (days 0-42 only) ...")
    for chunk in pd.read_csv(
        os.path.join(DATA_DIR, "studentVle.csv"),
        dtype=dtypes,
        chunksize=500_000,
    ):
        chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] <= CUTOFF_DAY)]
        if chunk.empty:
            continue
        chunk = chunk.merge(vle_lookup, on=["id_site", "code_module", "code_presentation"],
                            how="left")
        chunk["bucket"] = chunk["bucket"].fillna("other")

        # Per-bucket clicks within this chunk
        per_bucket = (
            chunk.groupby(KEY + ["bucket"], observed=False)["sum_click"]
            .sum()
            .reset_index()
        )
        per_bucket_parts.append(per_bucket)

        # Per-day activity flag (for recency + longest streak)
        day_act = chunk.groupby(KEY + ["date"], observed=False)["sum_click"].sum().reset_index()
        day_activity_parts.append(day_act)

    # Re-aggregate across chunks, then pivot buckets to columns
    per_bucket_all = (
        pd.concat(per_bucket_parts, ignore_index=True)
        .groupby(KEY + ["bucket"])["sum_click"].sum()
        .reset_index()
    )
    bucket_wide = per_bucket_all.pivot_table(
        index=KEY, columns="bucket", values="sum_click", fill_value=0
    )
    bucket_wide.columns = [f"clicks_{c}" for c in bucket_wide.columns]
    bucket_wide = bucket_wide.reset_index()

    # Ensure every bucket column exists (even if empty in this cohort)
    for bucket in list(ACTIVITY_BUCKETS.keys()) + ["other"]:
        col = f"clicks_{bucket}"
        if col not in bucket_wide.columns:
            bucket_wide[col] = 0

    # Share of homepage vs. non-homepage — a student who only logs in without clicking
    # on real content ('homepage-only') is a distinctive disengagement pattern.
    total_clicks = bucket_wide[[c for c in bucket_wide.columns if c.startswith("clicks_")]].sum(axis=1)
    bucket_wide["homepage_share"] = np.where(
        total_clicks > 0, bucket_wide["clicks_homepage"] / total_clicks, 0
    )

    # Day-level activity for recency and longest inactive streak
    day_act_all = (
        pd.concat(day_activity_parts, ignore_index=True)
        .groupby(KEY + ["date"])["sum_click"].sum()
        .reset_index()
    )
    # Keep only active days (sum_click > 0)
    day_act_all = day_act_all[day_act_all["sum_click"] > 0]

    recency = (
        day_act_all.groupby(KEY)["date"]
        .agg(last_active_day="max", first_active_day="min")
        .reset_index()
    )
    recency["days_since_last_click"] = CUTOFF_DAY - recency["last_active_day"]

    # Longest inactive streak within [0, 42]
    streak = _longest_gap(day_act_all)

    raw = bucket_wide.merge(recency[KEY + ["days_since_last_click"]], on=KEY, how="left")
    raw = raw.merge(streak, on=KEY, how="left")

    # Students with zero clicks in the window won't appear in weekly_scores; the main
    # merge is left on weekly_scores aggregate, so they'd be dropped anyway. Fill NaNs
    # for recency features conservatively (treat as "never active in window").
    raw["days_since_last_click"] = raw["days_since_last_click"].fillna(CUTOFF_DAY + 1)
    raw["longest_inactive_streak"] = raw["longest_inactive_streak"].fillna(CUTOFF_DAY + 1)

    return raw


def _longest_gap(day_act):
    """Per-student longest run of consecutive inactive days inside [0, 42]."""
    rows = []
    for key, sub in day_act.groupby(KEY):
        days = np.sort(sub["date"].to_numpy())
        # Build the set of active days and find longest gap using padded boundaries.
        padded = np.concatenate(([-1], days, [CUTOFF_DAY + 1]))
        gaps = np.diff(padded) - 1
        longest = int(gaps.max()) if len(gaps) else CUTOFF_DAY + 1
        rows.append((*key, longest))
    return pd.DataFrame(rows, columns=KEY + ["longest_inactive_streak"])


# ---------------------------------------------------------------------------
# Assessment submission feature — did they actually turn things in?
# ---------------------------------------------------------------------------

def build_submission_features():
    """Count how many of the assessments due by day 42 were submitted by day 42."""
    assessments = pd.read_csv(
        os.path.join(DATA_DIR, "assessments.csv"),
        usecols=["id_assessment", "code_module", "code_presentation", "date", "weight"],
    )
    assessments = assessments[assessments["date"] <= CUTOFF_DAY].copy()

    sub = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"))
    sub = sub[sub["is_banked"] == 0]
    sub = sub[sub["date_submitted"] <= CUTOFF_DAY]

    # Join submissions with assessment metadata
    sub = sub.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
        on="id_assessment", how="inner",
    )
    sub_agg = (
        sub.groupby(KEY)
        .agg(
            n_assessments_submitted_w1_6=("id_assessment", "nunique"),
            submitted_weight_w1_6=("weight", "sum"),
            mean_submission_score_w1_6=("score", "mean"),
        )
        .reset_index()
    )

    # Per-course: how many assessments were due by day 42 — used to compute submission rate.
    due_per_course = (
        assessments.groupby(["code_module", "code_presentation"])
        .agg(n_assessments_due_w1_6=("id_assessment", "nunique"),
             total_weight_due_w1_6=("weight", "sum"))
        .reset_index()
    )
    return sub_agg, due_per_course


def build_weekly_wide(scores):
    """Aggregate per-week features across weeks 1..6 into a single row per student."""
    sw = scores[scores["week_num"] <= CUTOFF_WEEK].copy()

    # Per-week click pivot — one column per week gives the model the full trajectory.
    clicks_wide = (
        sw.pivot_table(
            index=KEY,
            columns="week_num",
            values="weekly_clicks",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=lambda w: f"clicks_wk{int(w)}")
        .reset_index()
    )
    for w in range(1, CUTOFF_WEEK + 1):
        col = f"clicks_wk{w}"
        if col not in clicks_wide.columns:
            clicks_wide[col] = 0

    # Summary stats across the 6-week window.
    summary = (
        sw.groupby(KEY)
        .agg(
            clicks_total_w1_6=("weekly_clicks", "sum"),
            clicks_mean=("weekly_clicks", "mean"),
            clicks_std=("weekly_clicks", "std"),
            active_days_total=("active_days", "sum"),
            activity_diversity_max=("activity_diversity", "max"),
            assessment_timeliness_mean_w1_6=("assessment_timeliness", "mean"),
            engagement_score_w6=("engagement_score", "last"),
            engagement_score_mean=("engagement_score", "mean"),
            engagement_volatility=("engagement_volatility", "first"),
        )
        .reset_index()
    )
    summary["clicks_std"] = summary["clicks_std"].fillna(0)

    # Engagement slope (week 1..6) — OLS slope per student, vectorised for speed.
    slope = _engagement_slope(sw)

    # Weeks with zero activity during the window.
    weeks_inactive = (
        sw.assign(is_zero=(sw["weekly_clicks"] == 0).astype(int))
        .groupby(KEY)["is_zero"]
        .sum()
        .rename("weeks_inactive_w1_6")
        .reset_index()
    )

    out = (
        summary.merge(clicks_wide, on=KEY, how="left")
        .merge(slope, on=KEY, how="left")
        .merge(weeks_inactive, on=KEY, how="left")
    )
    return out


def _engagement_slope(sw):
    """Per-student linear slope of engagement_score vs week_num over weeks 1..6."""
    x = sw["week_num"].to_numpy(dtype=float)
    y = sw["engagement_score"].to_numpy(dtype=float)
    grp = sw.groupby(KEY)

    slopes = {}
    # Numpy polyfit per group — vectorised enough for ~30k students.
    for name, idx in grp.indices.items():
        xs = x[idx]
        ys = y[idx]
        if len(xs) < 2 or xs.std() == 0:
            slopes[name] = 0.0
        else:
            # Use polyfit degree 1 → slope
            slopes[name] = float(np.polyfit(xs, ys, 1)[0])

    slope_df = (
        pd.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in slopes.items()],
            columns=KEY + ["engagement_slope_w1_6"],
        )
    )
    return slope_df


def attach_raw_and_submission(features, raw_vle, sub_agg, due_per_course):
    """Merge raw-VLE and submission features onto the aggregated weekly frame."""
    out = features.merge(raw_vle, on=KEY, how="left")
    out = out.merge(sub_agg, on=KEY, how="left")
    out = out.merge(due_per_course, on=["code_module", "code_presentation"], how="left")

    # Fill sensible defaults for students with no submissions / no VLE activity.
    bucket_cols = [f"clicks_{b}" for b in list(ACTIVITY_BUCKETS.keys()) + ["other"]]
    for col in bucket_cols + ["homepage_share"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    out["days_since_last_click"] = out["days_since_last_click"].fillna(CUTOFF_DAY + 1)
    out["longest_inactive_streak"] = out["longest_inactive_streak"].fillna(CUTOFF_DAY + 1)
    for col in ("n_assessments_submitted_w1_6", "submitted_weight_w1_6"):
        out[col] = out[col].fillna(0)
    # mean_submission_score left as NaN when no submissions — XGBoost handles NaN natively.

    # Submission rate — fraction of available assessment weight the student actually submitted.
    out["submission_weight_rate"] = np.where(
        out["total_weight_due_w1_6"] > 0,
        out["submitted_weight_w1_6"] / out["total_weight_due_w1_6"],
        0,
    )
    return out


def attach_static_features(features, student_info, student_reg):
    """Add demographics (known at enrolment) and registration-timing features."""
    demo_cols = [
        "gender", "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability", "final_result",
    ]
    info = student_info[KEY + demo_cols].copy()

    # Registration timing — negative means registered after course start, positive = how
    # many days before start. NaN registration dates are rare; fill with 0 (course-start day).
    reg = student_reg[KEY + ["date_registration", "date_unregistration"]].copy()
    reg["days_registered_before_start"] = -reg["date_registration"].fillna(0)

    merged = (
        features.merge(info, on=KEY, how="inner")
        .merge(
            reg[KEY + ["days_registered_before_start", "date_unregistration"]],
            on=KEY,
            how="left",
        )
    )
    return merged


def apply_cohort_filter(df):
    """Exclude students who unregistered on or before day 42 (per plan).

    These cases are trivial to flag with a business rule ('student already gone') and
    bias the ML target toward a degenerate signal. Drop `date_unregistration` after
    the filter so it can never leak post-day-42 values into training.
    """
    before = len(df)
    still_registered_at_w6 = (
        df["date_unregistration"].isna() | (df["date_unregistration"] > CUTOFF_DAY)
    )
    df = df.loc[still_registered_at_w6].drop(columns=["date_unregistration"]).copy()
    print(f"  Cohort filter: kept {len(df):,} / {before:,} students "
          f"(dropped {before - len(df):,} who unregistered by day {CUTOFF_DAY}).")
    return df


def build_label(df):
    """Target: 1 if Withdrawn/Fail, 0 if Pass/Distinction."""
    df = df[df["final_result"].isin(["Withdrawn", "Fail", "Pass", "Distinction"])].copy()
    df["y"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    return df


def verify_no_post_week6_columns(df):
    """Defensive check: no column name encodes a week > 6."""
    bad = [c for c in df.columns
           if any(c.startswith(f"clicks_wk{w}") for w in range(CUTOFF_WEEK + 1, 40))]
    assert not bad, f"Leakage: columns derived from week > {CUTOFF_WEEK}: {bad}"


def main():
    print("=== Task 2: Building Week-6 feature matrix ===\n")

    print("[1/7] Loading inputs ...")
    scores, student_info, student_reg = load_inputs()

    print(f"[2/7] Aggregating weekly features (week_num <= {CUTOFF_WEEK}) ...")
    feats = build_weekly_wide(scores)
    print(f"  Weekly-feature rows: {len(feats):,}")

    print("[3/7] Building per-activity-type clicks + recency + inactive-streak features ...")
    raw_vle = build_raw_vle_features()
    print(f"  Raw-VLE feature rows: {len(raw_vle):,}")

    print("[4/7] Building assessment submission features ...")
    sub_agg, due_per_course = build_submission_features()
    feats = attach_raw_and_submission(feats, raw_vle, sub_agg, due_per_course)

    print("[5/7] Attaching demographics + registration timing ...")
    feats = attach_static_features(feats, student_info, student_reg)

    print("[6/7] Applying cohort filter and labeling ...")
    feats = apply_cohort_filter(feats)
    feats = build_label(feats)
    verify_no_post_week6_columns(feats)

    base_rate = feats["y"].mean()
    print(f"  Base rate (withdraw/fail within kept cohort): {base_rate:.3f}")
    print(f"  Final rows: {len(feats):,}")

    out_path = os.path.join(OUT_DIR, "week6_features.csv")
    feats.to_csv(out_path, index=False)
    print(f"[7/7] Saved -> {out_path}")

    # Sanity breakdown by presentation
    print("\n  Rows per presentation:")
    print(feats.groupby("code_presentation").size().to_string())
    print("\n  Base rate per presentation:")
    print(feats.groupby("code_presentation")["y"].mean().round(3).to_string())
    print("\n=== Done. Run disengagement_model.py next. ===")


if __name__ == "__main__":
    main()
