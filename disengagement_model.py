"""
Task 2: Predictive Disengagement — Train + Calibrate + Evaluate

Reads outputs/week6_features.csv. Trains an XGBoost classifier with a temporal
train/val/test split, calibrates probabilities with isotonic regression, selects
an operating threshold under a usefulness constraint (precision lift floor +
advisor capacity), and emits metrics and plots for the report.

Splits (temporal, matches existing project convention):
  train       = 2013J
  validation  = 2013B, 2014B  (used for early stopping + calibration fit)
  test        = 2014J          (held out for final metrics)
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")

TRAIN_PRES = ["2013J"]
VAL_PRES = ["2013B", "2014B"]
TEST_PRES = ["2014J"]

# Ordinal encodings — order is meaningful.
HIGHEST_ED_ORDER = [
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]
IMD_ORDER = [
    "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]
AGE_ORDER = ["0-35", "35-55", "55<="]

ORDINAL_COLS = {
    "highest_education": HIGHEST_ED_ORDER,
    "imd_band": IMD_ORDER,
    "age_band": AGE_ORDER,
}
ONEHOT_COLS = ["gender", "region", "disability", "code_module"]

# Numeric features
NUMERIC_COLS = [
    # Aggregate engagement from Task 1 weekly frame
    "clicks_total_w1_6", "clicks_mean", "clicks_std",
    "clicks_wk1", "clicks_wk2", "clicks_wk3",
    "clicks_wk4", "clicks_wk5", "clicks_wk6",
    "active_days_total", "activity_diversity_max",
    "assessment_timeliness_mean_w1_6",
    "engagement_score_w6", "engagement_score_mean",
    "engagement_volatility", "engagement_slope_w1_6",
    "weeks_inactive_w1_6",
    # Per-activity-type clicks (raw VLE): quiz/content/forum/etc are semantically distinct.
    "clicks_quiz", "clicks_content", "clicks_resource",
    "clicks_forum", "clicks_collab", "clicks_url", "clicks_homepage", "clicks_other",
    "homepage_share",
    # Recency + inactivity pattern
    "days_since_last_click", "longest_inactive_streak",
    # Assessment submission behavior (not just timeliness)
    "n_assessments_submitted_w1_6", "n_assessments_due_w1_6",
    "submitted_weight_w1_6", "total_weight_due_w1_6",
    "submission_weight_rate", "mean_submission_score_w1_6",
    # Static / registration
    "num_of_prev_attempts", "studied_credits",
    "days_registered_before_start",
]


def load_splits():
    df = pd.read_csv(os.path.join(OUT_DIR, "week6_features.csv"))
    train = df[df["code_presentation"].isin(TRAIN_PRES)].copy()
    val = df[df["code_presentation"].isin(VAL_PRES)].copy()
    test = df[df["code_presentation"].isin(TEST_PRES)].copy()
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"  train base rate={train['y'].mean():.3f}  "
          f"val={val['y'].mean():.3f}  test={test['y'].mean():.3f}")
    return train, val, test


def build_preprocessor():
    ord_cols = list(ORDINAL_COLS.keys())
    ord_categories = [ORDINAL_COLS[c] for c in ord_cols]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("ord", OrdinalEncoder(
                categories=ord_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ord_cols),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLS),
        ],
        remainder="drop",
    )


def make_model(scale_pos_weight: float) -> Pipeline:
    # Deeper trees + more estimators exploit the richer raw-VLE features; the higher
    # scale_pos_weight tilts the loss toward catching positives (recall is the priority).
    xgb_clf = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight * 1.3,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", xgb_clf),
    ])


def feature_matrix(df):
    return df[NUMERIC_COLS + list(ORDINAL_COLS.keys()) + ONEHOT_COLS].copy()


# ---------------------------------------------------------------------------
# Threshold selection — recall under a usefulness constraint (Plan §Modeling.4)
# ---------------------------------------------------------------------------

def pick_threshold(y_true, y_prob, base_rate, lift=1.2):
    """Select operating points, recall-prioritized per the brief.

    The brief mandates optimizing for recall. The naive "flag everyone" solution is
    ruled out by a precision-lift floor, but we set it loosely (1.2x base rate) so
    the floor is a guard-rail against the degenerate case, not a recall ceiling.

    Returned operating points:
      - `max_recall` — highest-recall threshold whose precision still clears the lift
                       floor. **Primary product operating point** (per the brief).
      - `f1_max`     — threshold that maximizes F1 (balanced reference).
      - `f2`         — threshold that maximizes F2 (recall-weighted reference).
      - `default`    — 0.5 (sanity reference).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec_t, rec_t = prec[:-1], rec[:-1]

    floor_value = lift * base_rate
    mask = prec_t >= floor_value
    if mask.any():
        i_maxrec = np.argmax(np.where(mask, rec_t, -1))
        max_recall_thr = float(thr[i_maxrec])
    else:
        max_recall_thr = 0.5
        print(f"  [warn] No threshold clears precision floor {floor_value:.3f}; "
              f"max_recall falls back to 0.5.")

    def _fbeta(beta):
        b2 = beta * beta
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.where(
                (prec_t + rec_t) > 0,
                (1 + b2) * prec_t * rec_t / (b2 * prec_t + rec_t + 1e-12),
                0.0,
            )
        return float(thr[int(np.argmax(vals))])

    return {
        "default": 0.5,
        "max_recall": max_recall_thr,
        "f1_max": _fbeta(1.0),
        "f2": _fbeta(2.0),
        "precision_floor_lift": float(lift),
        "precision_floor_value": float(floor_value),
    }


def metrics_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "flagged_fraction": float(y_pred.mean()),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration(y_true, y_prob_raw, y_prob_cal, path):
    frac_raw, mean_raw = calibration_curve(y_true, y_prob_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, y_prob_cal, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
    ax.plot(mean_raw, frac_raw, "o-", color="#94a3b8", label="Raw XGBoost")
    ax.plot(mean_cal, frac_cal, "o-", color="#2563eb", label="Isotonic-calibrated")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed withdrawal/fail rate")
    ax.set_title("Calibration — Week-6 Disengagement Model (test split)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(cm, thr, path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(np.array(cm)):
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > np.array(cm).max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Pass/Dist", "Pred Withdraw/Fail"])
    ax.set_yticklabels(["Actual Pass/Dist", "Actual Withdraw/Fail"])
    ax.set_title(f"Confusion Matrix @ threshold={thr:.2f} {title_suffix}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_feature_importance(shap_values, feature_names, path, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(names)), vals[::-1], color="#2563eb")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean |SHAP value|  (avg impact on risk prediction)")
    ax.set_title("Top feature drivers of Week-6 disengagement risk")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Task 2: Training disengagement classifier ===\n")

    print("[1/7] Loading splits ...")
    train, val, test = load_splits()

    X_train, y_train = feature_matrix(train), train["y"].to_numpy()
    X_val,   y_val   = feature_matrix(val),   val["y"].to_numpy()
    X_test,  y_test  = feature_matrix(test),  test["y"].to_numpy()

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)
    print(f"  scale_pos_weight = neg/pos = {neg}/{pos} = {spw:.3f}")

    print("\n[2/7] Fitting XGBoost with early stopping on validation AUC ...")
    model = make_model(scale_pos_weight=spw)
    pre = model.named_steps["pre"]
    # Fit preprocessor on train, transform train+val for early stopping.
    X_train_enc = pre.fit_transform(X_train)
    X_val_enc = pre.transform(X_val)
    clf = model.named_steps["clf"]
    clf.set_params(early_stopping_rounds=30)
    clf.fit(
        X_train_enc, y_train,
        eval_set=[(X_val_enc, y_val)],
        verbose=False,
    )
    best_iter = getattr(clf, "best_iteration", None)
    print(f"  best_iteration = {best_iter}")

    print("\n[3/7] Calibrating probabilities (isotonic, frozen on val) ...")
    # Wrap the fitted pipeline in FrozenEstimator so CalibratedClassifierCV trains only
    # the isotonic regressor on top, without refitting the base model (sklearn 1.6+ pattern).
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)

    print("\n[4/7] Scoring splits ...")
    p_train = calibrated.predict_proba(X_train)[:, 1]
    p_val = calibrated.predict_proba(X_val)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]
    # Raw (uncalibrated) test probabilities for the calibration plot comparison.
    p_test_raw = clf.predict_proba(X_test_encode := pre.transform(X_test))[:, 1]

    auc_train = roc_auc_score(y_train, p_train)
    auc_val = roc_auc_score(y_val, p_val)
    auc_test = roc_auc_score(y_test, p_test)
    brier_test = brier_score_loss(y_test, p_test)
    print(f"  ROC-AUC  train={auc_train:.3f}  val={auc_val:.3f}  test={auc_test:.3f}")
    print(f"  Brier (test, calibrated) = {brier_test:.4f}")

    print("\n[5/7] Selecting operating thresholds on validation set ...")
    base_rate_val = float(y_val.mean())
    base_rate_test = float(y_test.mean())
    thr = pick_threshold(y_val, p_val, base_rate=base_rate_val, lift=1.2)
    print(f"  base_rate (val)  = {base_rate_val:.3f}")
    print(f"  precision floor   = {thr['precision_floor_lift']:.1f}x base_rate = "
          f"{thr['precision_floor_value']:.3f}")
    print(f"  thresholds: default={thr['default']:.3f}  "
          f"max_recall={thr['max_recall']:.3f}  "
          f"f1_max={thr['f1_max']:.3f}  f2={thr['f2']:.3f}")

    test_metrics = {
        name: metrics_at(y_test, p_test, thr[name])
        for name in ("default", "max_recall", "f1_max", "f2")
    }
    # Product operating point: max_recall. The brief explicitly requires maximizing
    # recall; the 1.2x lift-floor constraint prevents the degenerate "flag everyone"
    # solution without capping how much recall we can achieve.
    product_thr = thr["max_recall"]
    test_metrics["product"] = metrics_at(y_test, p_test, product_thr)

    metrics_out = {
        "base_rate": {"train": float(y_train.mean()),
                      "val": base_rate_val,
                      "test": base_rate_test},
        "roc_auc": {"train": float(auc_train), "val": float(auc_val), "test": float(auc_test)},
        "brier_test": float(brier_test),
        "thresholds": thr,
        "product_threshold": float(product_thr),
        "test_metrics": test_metrics,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved -> outputs/metrics.json")

    print("\n[6/7] Generating plots ...")
    plot_calibration(y_test, p_test_raw, p_test,
                     os.path.join(OUT_DIR, "calibration.png"))
    cm_product = test_metrics["product"]["confusion_matrix"]
    plot_confusion(cm_product, product_thr,
                   os.path.join(OUT_DIR, "confusion_matrix.png"),
                   title_suffix="(product operating point)")

    # SHAP on the raw XGBoost model (calibration wraps probabilities, doesn't change ranking).
    feature_names = _get_feature_names(pre)
    explainer = shap.TreeExplainer(clf)
    # Sample 2000 rows to keep it fast; SHAP averages are stable at that size.
    sample_idx = np.random.RandomState(42).choice(
        len(X_test_encode), size=min(2000, len(X_test_encode)), replace=False
    )
    shap_values = explainer.shap_values(X_test_encode[sample_idx])
    plot_feature_importance(shap_values, feature_names,
                            os.path.join(OUT_DIR, "feature_importance.png"))
    print("  Plots saved -> outputs/{calibration,confusion_matrix,feature_importance}.png")

    print("\n[7/7] Persisting model ...")
    joblib.dump({
        "calibrated": calibrated,
        "raw_clf": clf,
        "preprocessor": pre,
        "feature_names": feature_names,
        "numeric_cols": NUMERIC_COLS,
        "ordinal_cols": list(ORDINAL_COLS.keys()),
        "onehot_cols": ONEHOT_COLS,
        "product_threshold": product_thr,
        "thresholds": thr,
    }, os.path.join(OUT_DIR, "disengagement_model.joblib"))
    print("  Model saved -> outputs/disengagement_model.joblib")

    print("\n=== Summary ===")
    pm = test_metrics["product"]
    print(f"  Product operating point: threshold={product_thr:.3f}")
    print(f"    precision={pm['precision']:.3f}  recall={pm['recall']:.3f}  "
          f"f1={pm['f1']:.3f}  f2={pm['f2']:.3f}  flagged={pm['flagged_fraction']:.3f}")
    print(f"  ROC-AUC test = {auc_test:.3f}   Brier = {brier_test:.4f}")
    print("\n=== Done. Run disengagement_alert.py next. ===")


def _get_feature_names(preprocessor):
    names = []
    names.extend(NUMERIC_COLS)
    names.extend(list(ORDINAL_COLS.keys()))
    oh: OneHotEncoder = preprocessor.named_transformers_["oh"]
    oh_names = oh.get_feature_names_out(ONEHOT_COLS).tolist()
    names.extend(oh_names)
    return names


if __name__ == "__main__":
    main()
