"""
Task 3: Collaborative filtering recommenders.

Two variants:
  * Item-based k-NN CF on the student x module interaction matrix.
  * User-based k-NN CF in the behavioral profile space (same features as the
    content recommender).

We deliberately skip SVD / matrix factorization — the student x module matrix
has only 7 columns; with that few items, latent-factor methods overfit and
components mostly capture noise. k-NN does not assume low-rank structure and
behaves well on wide-and-shallow catalogs.
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")

TRAIN_PRESENTATIONS = ("2013J", "2013B", "2014B")

OUTCOME_WEIGHT = {
    "Distinction": 1.00,
    "Pass": 0.85,
    "Fail": 0.35,
    "Withdrawn": 0.15,
}


def load_interactions(train_only: bool = False) -> pd.DataFrame:
    """Engagement-weighted student x module interactions.

    For item-item CF we need co-occurrence signal — i.e. the same student
    appearing on multiple modules. Restricting to a single presentation window
    eliminates almost all co-occurrence (most students take 1 module in a given
    semester). We therefore use *all* historical enrollments to learn module-
    module affinity, which is a static catalog property. The evaluation
    harness enforces temporal correctness at recommendation time by only
    passing history strictly prior to the holdout presentation.
    """
    sp = pd.read_csv(os.path.join(OUT_DIR, "student_profiles_reco.csv"))
    if train_only:
        sp = sp[sp["code_presentation"].isin(TRAIN_PRESENTATIONS)].copy()
    else:
        sp = sp.copy()
    sp["interaction"] = sp["final_result"].map(OUTCOME_WEIGHT).fillna(0.5)
    agg = sp.groupby(["id_student", "code_module"], as_index=False)["interaction"].max()
    return agg


def build_matrix(interactions: pd.DataFrame):
    students = np.sort(interactions["id_student"].unique())
    modules = np.sort(interactions["code_module"].unique())
    s_idx = {s: i for i, s in enumerate(students)}
    m_idx = {m: i for i, m in enumerate(modules)}
    rows = interactions["id_student"].map(s_idx).values
    cols = interactions["code_module"].map(m_idx).values
    vals = interactions["interaction"].values
    mat = csr_matrix((vals, (rows, cols)), shape=(len(students), len(modules)))
    return mat, students, modules


class ItemKNN:
    """Item-item cosine CF.

    For a new student s with known interactions v (length n_items),
    scores for unseen items are: sim @ v, masked to items not yet taken.
    """

    def __init__(self):
        self.item_sim_ = None
        self.modules_ = None

    def fit(self, mat: csr_matrix, modules: np.ndarray):
        # cosine between columns (items), computed on the dense transpose —
        # fine here because there are only 6-7 items.
        self.item_sim_ = cosine_similarity(mat.T)
        np.fill_diagonal(self.item_sim_, 0.0)
        self.modules_ = modules
        return self

    def recommend_from_history(self, history: dict, k: int = 3) -> list:
        """history: {code_module: interaction_weight}."""
        scores = np.zeros(len(self.modules_))
        for m, w in history.items():
            if m not in self.modules_:
                continue
            j = int(np.where(self.modules_ == m)[0][0])
            scores += self.item_sim_[:, j] * w
        # Mask already-taken items.
        for m in history:
            if m in self.modules_:
                j = int(np.where(self.modules_ == m)[0][0])
                scores[j] = -np.inf
        order = np.argsort(-scores)
        picks = []
        for i in order:
            if not np.isfinite(scores[i]):
                continue
            picks.append(str(self.modules_[i]))
            if len(picks) >= k:
                break
        return picks


class UserKNN:
    """User-based k-NN in behavioral profile space.

    For a target student, find the k-nearest training students by cosine
    similarity on the profile vector, then aggregate the modules they took
    (weighted by neighbour similarity * outcome weight).
    """

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors
        self.scaler_ = None
        self.feats_ = None
        self.train_vecs_ = None
        self.train_meta_ = None

    def fit(self, student_profiles: pd.DataFrame, feats: list):
        train = student_profiles[
            student_profiles["code_presentation"].isin(TRAIN_PRESENTATIONS)
        ].reset_index(drop=True)
        self.feats_ = feats
        self.scaler_ = StandardScaler().fit(train[feats].values)
        self.train_vecs_ = self.scaler_.transform(train[feats].values)
        self.train_meta_ = train[["id_student", "code_module", "final_result"]].copy()
        self.train_meta_["outcome_w"] = (
            self.train_meta_["final_result"].map(OUTCOME_WEIGHT).fillna(0.5)
        )
        return self

    def recommend(self, student_row: pd.Series, k: int = 3,
                  exclude: set = None) -> list:
        vec = self.scaler_.transform(student_row[self.feats_].values.reshape(1, -1))
        sims = cosine_similarity(vec, self.train_vecs_)[0]
        # Top-K neighbours (excluding self if present).
        if "id_student" in student_row:
            self_mask = self.train_meta_["id_student"].values == student_row["id_student"]
            sims = np.where(self_mask, -np.inf, sims)
        nbr_idx = np.argpartition(-sims, min(self.k_neighbors, len(sims) - 1))[: self.k_neighbors]
        nbr_sims = sims[nbr_idx]
        nbr_meta = self.train_meta_.iloc[nbr_idx].copy()
        nbr_meta["sim"] = np.maximum(nbr_sims, 0.0)  # negative similarity -> zero weight
        nbr_meta["weight"] = nbr_meta["sim"] * nbr_meta["outcome_w"]

        agg = nbr_meta.groupby("code_module")["weight"].sum().sort_values(ascending=False)
        if exclude:
            agg = agg.drop(index=[m for m in exclude if m in agg.index], errors="ignore")
        return agg.head(k).index.tolist()


def main():
    interactions = load_interactions()
    mat, students, modules = build_matrix(interactions)
    print(f"[collab_filter] matrix: {mat.shape}, nnz={mat.nnz}, modules={list(modules)}")

    item_knn = ItemKNN().fit(mat, modules)
    print("[collab_filter] item-item similarity (rounded):")
    print(pd.DataFrame(item_knn.item_sim_, index=modules, columns=modules).round(2))

    # Sanity check: diag is zeroed by construction; symmetry within tolerance.
    assert np.allclose(item_knn.item_sim_, item_knn.item_sim_.T, atol=1e-8)

    sp = pd.read_csv(os.path.join(OUT_DIR, "student_profiles_reco.csv"))
    cp = pd.read_csv(os.path.join(OUT_DIR, "course_profiles.csv"))
    feats = [c for c in cp.columns
             if c not in {"code_module", "n_enrolled", "pass_rate",
                          "pass_rate_wilson_low", "withdraw_rate"}
             and c in sp.columns]
    user_knn = UserKNN(k_neighbors=50).fit(sp, feats)

    # Demo on 3 test-cohort students.
    demo = sp[sp["code_presentation"] == "2014J"].sample(3, random_state=42)
    for _, row in demo.iterrows():
        history = {row["code_module"]: OUTCOME_WEIGHT.get(row["final_result"], 0.5)}
        item_picks = item_knn.recommend_from_history(history, k=3)
        user_picks = user_knn.recommend(row, k=3, exclude={row["code_module"]})
        print(f"student={row['id_student']} took {row['code_module']} "
              f"({row['final_result']}) -> item-KNN: {item_picks}  user-KNN: {user_picks}")


if __name__ == "__main__":
    main()
