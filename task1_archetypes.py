"""
Task 1: Archetype Discovery — PathAI Engine
K-Means clustering on smoothed engagement trajectories + visualization.
Run AFTER task1_behavioral_scoring.py has produced outputs/weekly_scores.csv.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "outputs")


# ---------------------------------------------------------------------------
# 1. Load scored data
# ---------------------------------------------------------------------------

def load_data():
    scores = pd.read_csv(os.path.join(OUT_DIR, "weekly_scores.csv"))
    assessments = pd.read_csv(
        os.path.join(DATA_DIR, "assessments.csv"),
        usecols=["code_module", "code_presentation", "date"],
    )
    assessments["due_week"] = (assessments["date"] // 7) + 1
    return scores, assessments


# ---------------------------------------------------------------------------
# 2. Build trajectory matrix
# ---------------------------------------------------------------------------

def build_trajectory_matrix(scores):
    """
    Pivot to (student × week) matrix.
    Weeks after date_unregistration are forced to 0 (student has left).
    Remaining NaN weeks (enrolled but no clicks) also -> 0.
    """
    df = scores.copy()

    # Force scores to 0 after withdrawal week
    df.loc[df["week_num"] > df["unreg_week"], "engagement_score"] = 0.0

    max_week = int(df["week_num"].max())

    trajectory = df.pivot_table(
        index=["id_student", "code_module", "code_presentation"],
        columns="week_num",
        values="engagement_score",
        fill_value=0.0,
    )
    # Ensure all week columns are present (some may be missing if no student active)
    for w in range(1, max_week + 1):
        if w not in trajectory.columns:
            trajectory[w] = 0.0
    trajectory = trajectory[sorted(trajectory.columns)]

    # Attach final_result for profiling
    meta = (
        df[["id_student", "code_module", "code_presentation", "final_result", "engagement_volatility"]]
        .drop_duplicates(subset=["id_student", "code_module", "code_presentation"])
    )
    return trajectory, meta, max_week


# ---------------------------------------------------------------------------
# 3. Cluster
# ---------------------------------------------------------------------------

def cluster_trajectories(trajectory, k=3):
    X = trajectory.values

    # Elbow + silhouette for k=2..6 (printed for transparency)
    print("  Silhouette scores:")
    for ki in range(2, 7):
        km_tmp = KMeans(n_clusters=ki, random_state=42, n_init=10)
        lbl = km_tmp.fit_predict(X)
        sil = silhouette_score(X, lbl, sample_size=min(5000, len(X)), random_state=42)
        print(f"    k={ki}: silhouette={sil:.4f}")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    print(f"\n  Using k={k}  silhouette={sil:.4f}")
    return labels, km


# ---------------------------------------------------------------------------
# 4. Name archetypes by inspecting cluster centroids
# ---------------------------------------------------------------------------

def name_archetypes(km, max_week):
    """
    Heuristic naming based on centroid shape:
    - Steady Engager:  high mean, low variance
    - Early Dropout:   high early weeks, low late weeks
    - Late Recoverer:  low early weeks, recovering later
    """
    centroids = km.cluster_centers_
    n_weeks = centroids.shape[1]
    early_end = max(1, n_weeks // 4)
    late_start = max(early_end + 1, 3 * n_weeks // 4)

    cluster_stats = []
    for i, c in enumerate(centroids):
        cluster_stats.append({
            "cluster": i,
            "mean": c.mean(),
            "early_mean": c[:early_end].mean(),
            "late_mean": c[late_start:].mean(),
        })

    stats_df = pd.DataFrame(cluster_stats)
    # Steady Engager: highest overall mean
    steady = stats_df.loc[stats_df["mean"].idxmax(), "cluster"]
    # Early Dropout: highest early mean relative to late mean
    stats_df["decay"] = stats_df["early_mean"] - stats_df["late_mean"]
    remaining = stats_df[stats_df["cluster"] != steady]
    dropout = remaining.loc[remaining["decay"].idxmax(), "cluster"]
    # Late Recoverer: the remaining cluster
    recoverer = [c for c in stats_df["cluster"] if c not in (steady, dropout)][0]

    names = {int(steady): "Steady Engager", int(dropout): "Early Dropout", int(recoverer): "Late Recoverer"}
    print(f"\n  Archetype mapping: {names}")
    return names


# ---------------------------------------------------------------------------
# 5. Profile archetypes
# ---------------------------------------------------------------------------

def profile_archetypes(trajectory, meta, labels, names, max_week):
    traj_df = trajectory.copy()
    traj_df["cluster"] = labels
    traj_df = traj_df.reset_index()
    traj_df = traj_df.merge(meta, on=["id_student", "code_module", "code_presentation"], how="left")
    traj_df["archetype"] = traj_df["cluster"].map(names)

    print("\n  Archetype profiles:")
    print(f"  {'Archetype':<20} {'N':>6}  {'Pass%':>7}  {'Withdrawn%':>11}  {'Mean Score':>11}  {'Mean Volatility':>16}")
    print("  " + "-" * 80)
    for cluster_id, name in names.items():
        sub = traj_df[traj_df["cluster"] == cluster_id]
        n = len(sub)
        pass_pct = (sub["final_result"].isin(["Pass", "Distinction"]).sum() / n * 100) if n else 0
        with_pct = (sub["final_result"].eq("Withdrawn").sum() / n * 100) if n else 0
        week_cols = [c for c in trajectory.columns if isinstance(c, (int, np.integer))]
        mean_score = sub[week_cols].values.mean()
        mean_vol = sub["engagement_volatility"].mean() if "engagement_volatility" in sub else float("nan")
        print(f"  {name:<20} {n:>6}  {pass_pct:>6.1f}%  {with_pct:>10.1f}%  {mean_score:>11.1f}  {mean_vol:>15.2f}")

    return traj_df


# ---------------------------------------------------------------------------
# 6. Visualize
# ---------------------------------------------------------------------------

def plot_archetypes(traj_df, names, assessments, max_week, out_path):
    week_cols = sorted([c for c in traj_df.columns if isinstance(c, (int, np.integer))])
    weeks = np.array(week_cols)

    # Representative assessment due weeks (across all modules) for annotation
    assess_weeks = sorted(assessments["due_week"].dropna().unique().astype(int))
    assess_weeks = [w for w in assess_weeks if 1 <= w <= max_week]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Student Engagement Archetypes — PathAI Behavioral Scoring", fontsize=14, y=1.01)

    colors = {"Steady Engager": "#2ecc71", "Early Dropout": "#e74c3c", "Late Recoverer": "#3498db"}
    ordered = ["Steady Engager", "Early Dropout", "Late Recoverer"]

    for row, name in enumerate(ordered):
        cluster_id = {v: k for k, v in names.items()}[name]
        sub = traj_df[traj_df["cluster"] == cluster_id][week_cols].values
        color = colors[name]

        median_traj = np.median(sub, axis=0)
        p25 = np.percentile(sub, 25, axis=0)
        p75 = np.percentile(sub, 75, axis=0)

        # --- Left panel: trajectory ---
        ax_l = axes[row, 0]
        ax_l.plot(weeks, median_traj, color=color, linewidth=2, label="Median")
        ax_l.fill_between(weeks, p25, p75, color=color, alpha=0.25, label="IQR")
        for aw in assess_weeks[:8]:  # annotate up to 8 assessment weeks
            ax_l.axvline(aw, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax_l.set_title(f"{name} — Engagement Trajectory  (n={len(sub):,})", fontsize=11)
        ax_l.set_xlabel("Week")
        ax_l.set_ylabel("Engagement Score (0–100)")
        ax_l.set_ylim(0, 100)
        ax_l.set_xlim(weeks[0], weeks[-1])
        ax_l.legend(fontsize=8)
        ax_l.grid(axis="y", alpha=0.3)

        # --- Right panel: final_result distribution ---
        ax_r = axes[row, 1]
        cluster_outcomes = traj_df.loc[traj_df["cluster"] == cluster_id, "final_result"]
        counts = cluster_outcomes.value_counts()
        bar_colors = {
            "Pass": "#2ecc71", "Distinction": "#27ae60",
            "Fail": "#e67e22", "Withdrawn": "#e74c3c",
        }
        bar_c = [bar_colors.get(r, "#95a5a6") for r in counts.index]
        counts.plot(kind="bar", ax=ax_r, color=bar_c, edgecolor="white")
        ax_r.set_title(f"{name} — Final Result Distribution", fontsize=11)
        ax_r.set_xlabel("")
        ax_r.set_ylabel("# Students")
        ax_r.tick_params(axis="x", rotation=30)
        ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_r.grid(axis="y", alpha=0.3)
        # Annotate percentages
        total = counts.sum()
        for bar, cnt in zip(ax_r.patches, counts):
            ax_r.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{cnt/total*100:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    print("=== Task 1: Archetype Discovery ===\n")

    print("[1/4] Loading scored data ...")
    scores, assessments = load_data()

    print("[2/4] Building trajectory matrix ...")
    trajectory, meta, max_week = build_trajectory_matrix(scores)
    print(f"  Trajectory matrix: {trajectory.shape[0]:,} students × {trajectory.shape[1]} weeks")

    print("[3/4] Clustering (K-Means, k=3) ...")
    labels, km = cluster_trajectories(trajectory, k=3)
    names = name_archetypes(km, max_week)

    print("[4/4] Profiling and plotting ...")
    traj_df = profile_archetypes(trajectory, meta, labels, names, max_week)
    out_path = os.path.join(OUT_DIR, "archetypes.png")
    plot_archetypes(traj_df, names, assessments, max_week, out_path)

    # Save cluster assignments
    cluster_out = traj_df[
        ["id_student", "code_module", "code_presentation", "cluster", "archetype"]
    ].drop_duplicates()
    cluster_out.to_csv(os.path.join(OUT_DIR, "student_archetypes.csv"), index=False)
    print(f"  Cluster labels saved -> outputs/student_archetypes.csv")

    print("\n=== Archetype discovery complete. ===")


if __name__ == "__main__":
    main()
