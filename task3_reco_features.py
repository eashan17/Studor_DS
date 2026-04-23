"""
Task 3: Course Recommendation — feature builder.
Produces:
  outputs/course_profiles.csv          one row per code_module
  outputs/student_profiles_reco.csv    one row per (id_student, code_module, code_presentation)

Profiles share the same schema so a student vector and a course vector live in
the same space and can be compared with cosine similarity.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# The 2013 cohort is the training window. 2014 is held out for evaluation
# (matches the temporal split used in Task 1 / Task 2 — no leakage).
# Training window includes 2014B so that CCC (which has no 2013 presentation)
# appears in the course catalog. Evaluation holds out only 2014J, which starts
# strictly after 2014B ends, so this remains a valid temporal split.
TRAIN_PRESENTATIONS = ("2013J", "2013B", "2014B")

NUMERIC_COLS = [
    "clicks_mean", "clicks_std", "active_days_total", "activity_diversity_max",
    "assessment_timeliness_mean_w1_6", "engagement_score_mean",
    "engagement_volatility", "engagement_slope_w1_6", "homepage_share",
    "submission_weight_rate", "studied_credits", "num_of_prev_attempts",
]

# Demographic one-hot categories we aggregate over for the mix vector.
DEMO_CATEGORICAL = {
    "highest_education": [
        "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification",
    ],
    "imd_band": [
        "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
    ],
    "age_band": ["0-35", "35-55", "55<="],
    "disability": ["N", "Y"],
    "gender": ["F", "M"],
}


def _one_hot_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Expand categorical columns into one-hot indicator columns."""
    parts = []
    for col, cats in DEMO_CATEGORICAL.items():
        s = df[col].astype(str)
        oh = pd.DataFrame(
            {f"{col}={c}": (s == c).astype(float) for c in cats},
            index=df.index,
        )
        parts.append(oh)
    return pd.concat(parts, axis=1)


def build_student_profiles(week6: pd.DataFrame, archetypes: pd.DataFrame) -> pd.DataFrame:
    """Per-enrollment student profile vectors."""
    df = week6.copy()
    # Merge archetype labels (may be NaN for students without weekly scores).
    df = df.merge(
        archetypes[["id_student", "code_module", "code_presentation", "archetype"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    df["archetype"] = df["archetype"].fillna("Unknown")

    arche_oh = pd.get_dummies(df["archetype"], prefix="archetype").astype(float)
    demo_oh = _one_hot_mix(df)

    # Continuous columns — fill NA with 0 (missing = no signal).
    numeric = df[NUMERIC_COLS].astype(float).fillna(0.0)

    profile = pd.concat(
        [
            df[["id_student", "code_module", "code_presentation", "final_result"]]
              .reset_index(drop=True),
            numeric.reset_index(drop=True),
            demo_oh.reset_index(drop=True),
            arche_oh.reset_index(drop=True),
        ],
        axis=1,
    )
    return profile


def build_course_profiles(student_profiles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate student profiles into one vector per code_module.

    Uses only the training cohort so evaluation on later presentations is fair.
    """
    train = student_profiles[
        student_profiles["code_presentation"].isin(TRAIN_PRESENTATIONS)
    ].copy()

    value_cols = [c for c in train.columns if c not in
                  ("id_student", "code_module", "code_presentation", "final_result")]

    # Course profile = mean over enrolled students.
    course = train.groupby("code_module")[value_cols].mean().reset_index()

    # Attach outcome summary used as a *prior*, not a similarity axis.
    grp = train.groupby("code_module")["final_result"]
    pass_like = grp.apply(lambda s: s.isin(["Pass", "Distinction"]).mean())
    n_enrolled = grp.size()
    k = grp.apply(lambda s: s.isin(["Pass", "Distinction"]).sum())

    # Wilson lower bound for pass rate — discounts uncertainty, protects niche courses.
    z = 1.96
    n = n_enrolled.astype(float)
    phat = k.astype(float) / n
    denom = 1.0 + z ** 2 / n
    center = phat + z ** 2 / (2.0 * n)
    margin = z * np.sqrt((phat * (1.0 - phat) + z ** 2 / (4.0 * n)) / n)
    wilson_low = (center - margin) / denom

    outcome = pd.DataFrame({
        "code_module": n_enrolled.index,
        "n_enrolled": n_enrolled.values,
        "pass_rate": pass_like.values,
        "pass_rate_wilson_low": wilson_low.values,
        "withdraw_rate": grp.apply(lambda s: (s == "Withdrawn").mean()).values,
    })

    return course.merge(outcome, on="code_module", how="left")


def main():
    week6 = pd.read_csv(os.path.join(OUT_DIR, "week6_features.csv"))
    archetypes = pd.read_csv(os.path.join(OUT_DIR, "student_archetypes.csv"))

    student_profiles = build_student_profiles(week6, archetypes)
    course_profiles = build_course_profiles(student_profiles)

    sp_path = os.path.join(OUT_DIR, "student_profiles_reco.csv")
    cp_path = os.path.join(OUT_DIR, "course_profiles.csv")
    student_profiles.to_csv(sp_path, index=False)
    course_profiles.to_csv(cp_path, index=False)

    print(f"[task3_reco_features] students: {len(student_profiles):,} rows -> {sp_path}")
    print(f"[task3_reco_features] courses : {len(course_profiles):,} rows -> {cp_path}")
    print(course_profiles[["code_module", "n_enrolled", "pass_rate",
                           "pass_rate_wilson_low", "withdraw_rate"]].round(3))


if __name__ == "__main__":
    main()
