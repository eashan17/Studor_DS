"""
Task 3: Content-based course recommender.

score(student, course) = (1 - alpha) * cosine_sim_norm + alpha * pass_rate_prior

* cosine similarity is computed in a standardized behavioral+demographic space
  shared by students and courses.
* the prior is the Wilson lower bound on pass rate — this stops a large-but-
  mediocre course from drowning out a small-but-well-matched course.
* a minimum-similarity floor (sim >= SIM_FLOOR) keeps the prior from pulling
  in courses the student does not actually match.

alpha is tuned in task3_evaluate.py; this module exposes `recommend` as a
library function.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")

KEY_COLS = ("id_student", "code_module", "code_presentation", "final_result")
# sim_floor is applied on the *raw* cosine similarity. With only 6 courses in
# the catalog, requiring sim >= 0.30 eliminates most students (empirically the
# median row-max sim is ~0.1-0.2). We instead drop only *negative*-similarity
# courses — explicit mismatches — and let the combined score rank the rest.
SIM_FLOOR = 0.0
DEFAULT_ALPHA = 0.20


def _feature_columns(df: pd.DataFrame) -> list:
    skip = set(KEY_COLS) | {
        "n_enrolled", "pass_rate", "pass_rate_wilson_low", "withdraw_rate",
    }
    return [c for c in df.columns if c not in skip]


def load_profiles():
    student = pd.read_csv(os.path.join(OUT_DIR, "student_profiles_reco.csv"))
    course = pd.read_csv(os.path.join(OUT_DIR, "course_profiles.csv"))
    return student, course


def fit_scaler(student_profiles: pd.DataFrame, course_profiles: pd.DataFrame):
    """Fit StandardScaler on the student distribution.

    Students are the dense population; fitting the scaler here gives individual
    student vectors non-trivial magnitude, while course vectors (means of
    students) still retain meaningful direction after transform. Fitting on the
    6-row course table instead makes cosine values collapse toward zero.
    """
    feats = [c for c in _feature_columns(course_profiles)
             if c in student_profiles.columns]
    scaler = StandardScaler()
    scaler.fit(student_profiles[feats].values)
    return scaler, feats


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def score_matrix(student_vecs: np.ndarray,
                 course_mat: np.ndarray,
                 pass_prior: np.ndarray,
                 alpha: float = DEFAULT_ALPHA,
                 sim_floor: float = SIM_FLOOR) -> np.ndarray:
    """Return (n_students, n_courses) score matrix."""
    sim = cosine_similarity(student_vecs, course_mat)
    sim_norm = _normalize(sim)
    prior_norm = _normalize(pass_prior)
    score = (1.0 - alpha) * sim_norm + alpha * prior_norm[np.newaxis, :]
    # Zero out courses below the similarity floor so the prior cannot
    # surface a poor match.
    score = np.where(sim >= sim_floor, score, -np.inf)
    return score, sim


def top_k(score_row: np.ndarray, course_ids: np.ndarray, k: int = 3) -> list:
    """Return the top-k course codes for a single student, skipping -inf cells."""
    idx = np.argsort(-score_row)
    picks = []
    for i in idx:
        if not np.isfinite(score_row[i]):
            continue
        picks.append(course_ids[i])
        if len(picks) >= k:
            break
    return picks


def recommend(student_row: pd.Series,
              course_profiles: pd.DataFrame,
              scaler,
              feats: list,
              alpha: float = DEFAULT_ALPHA,
              k: int = 3,
              exclude: set = None) -> list:
    """Library entry: recommend top-k modules for a single student row.

    `exclude` can be a set of code_module values the student has already taken
    and should not be recommended again.
    """
    course_mat = scaler.transform(course_profiles[feats].values)
    stud = scaler.transform(student_row[feats].values.reshape(1, -1))
    prior = course_profiles["pass_rate_wilson_low"].values

    score, _sim = score_matrix(stud, course_mat, prior, alpha=alpha)
    row = score[0].copy()

    ids = course_profiles["code_module"].values
    if exclude:
        for j, cid in enumerate(ids):
            if cid in exclude:
                row[j] = -np.inf
    return top_k(row, ids, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", type=int, default=None)
    ap.add_argument("--presentation", type=str, default="2013J")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    student_profiles, course_profiles = load_profiles()
    scaler, feats = fit_scaler(student_profiles, course_profiles)

    if args.student_id is None:
        # Demo: pick 5 random students from the test cohort (2014J).
        sample = student_profiles[student_profiles["code_presentation"] == "2014J"]\
            .sample(5, random_state=42)
    else:
        sample = student_profiles[
            (student_profiles["id_student"] == args.student_id)
            & (student_profiles["code_presentation"] == args.presentation)
        ]
        if sample.empty:
            raise SystemExit(f"No profile found for student {args.student_id} / {args.presentation}")

    for _, row in sample.iterrows():
        picks = recommend(row, course_profiles, scaler, feats,
                          alpha=args.alpha, k=args.k,
                          exclude={row["code_module"]})
        print(f"student={row['id_student']} (took {row['code_module']} / "
              f"{row['code_presentation']}, outcome={row['final_result']}) -> "
              f"top-{args.k}: {picks}")


if __name__ == "__main__":
    main()
