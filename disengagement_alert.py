"""
Task 2: Advisor Alert Payload Generator

Loads the calibrated disengagement model, scores the test cohort, and writes three
example alert payloads for advisors (true positive, false positive, borderline) so
the product team can sanity-check the format before a real deployment.

Each alert contains the student's risk score, tier, and the top 3 SHAP-derived
drivers translated into plain English — "what to say to the student" evidence,
not raw feature names.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "outputs")
ALERT_DIR = os.path.join(OUT_DIR, "example_alerts")
Path(ALERT_DIR).mkdir(parents=True, exist_ok=True)

TEST_PRES = ["2014J"]


# ---------------------------------------------------------------------------
# Plain-English driver translation
# ---------------------------------------------------------------------------

def driver_explanation(feature_name, feature_value, shap_value, row):
    """Map a (feature, value, SHAP) triple to a human-readable {factor, evidence}.

    Rule of thumb: describe the observed value, not the SHAP direction. SHAP tells us
    *ranking* (why this is a top driver) but for non-monotonic features like
    engagement_volatility, the same SHAP sign can correspond to different
    human-readable stories depending on the value. So we anchor the text on the
    value itself and let SHAP do the selection.
    """
    pos = shap_value > 0  # positive = pushes toward disengagement risk (used for simple monotonic features)

    if feature_name == "clicks_total_w1_6":
        # Low clicks → high risk is monotonic; safe to use SHAP direction.
        direction = "Low" if pos else "High"
        return {
            "factor": f"{direction} VLE engagement in weeks 1-6",
            "evidence": f"{int(feature_value):,} total clicks across 42 days "
                        f"({int(row['active_days_total'])} active days)",
        }
    if feature_name == "assessment_timeliness_mean_w1_6":
        if pos:
            return {
                "factor": "Late or missed assessments",
                "evidence": f"Weighted timeliness = {feature_value:.2f} "
                            f"(0 = missed, 1 = on-time or early)",
            }
        return {
            "factor": "Strong assessment follow-through",
            "evidence": f"Weighted timeliness = {feature_value:.2f}",
        }
    if feature_name == "engagement_slope_w1_6":
        trend = "declining" if pos else "improving"
        return {
            "factor": f"Engagement trajectory {trend}",
            "evidence": f"Slope = {feature_value:+.1f} score-points per week over weeks 1-6",
        }
    if feature_name == "weeks_inactive_w1_6":
        return {
            "factor": f"Silent weeks in the 6-week window",
            "evidence": f"{int(feature_value)} of 6 weeks had zero VLE activity",
        }
    if feature_name == "engagement_score_w6":
        # Anchor on the value, not SHAP direction — "solid" at a 4/100 score is nonsensical.
        label = "Very low" if feature_value < 10 else "Low" if feature_value < 30 else \
                "Moderate" if feature_value < 50 else "Solid"
        return {
            "factor": f"{label} end-of-week-6 engagement score",
            "evidence": f"Smoothed score = {feature_value:.1f}/100",
        }
    if feature_name == "engagement_score_mean":
        label = "Very low" if feature_value < 10 else "Low" if feature_value < 30 else \
                "Moderate" if feature_value < 50 else "Solid"
        return {
            "factor": f"{label} average engagement through weeks 1-6",
            "evidence": f"Mean weekly score = {feature_value:.1f}/100",
        }
    if feature_name == "engagement_volatility":
        # Non-monotonic: low volatility can mean "steady-low" (bad) or "steady-high" (good).
        # Pair the std with the mean score so the advisor can interpret it in context.
        mean_score = row.get("engagement_score_mean", float("nan"))
        if feature_value < 2:
            story = "Steady but low engagement" if mean_score < 30 else "Steady engagement pattern"
        elif feature_value > 8:
            story = "Erratic week-to-week engagement"
        else:
            story = "Moderate week-to-week variability"
        return {
            "factor": story,
            "evidence": f"Std of weekly score = {feature_value:.2f} "
                        f"(mean = {mean_score:.1f}/100)",
        }
    if feature_name in ("clicks_wk1", "clicks_wk2", "clicks_wk3",
                         "clicks_wk4", "clicks_wk5", "clicks_wk6"):
        wk = feature_name.replace("clicks_wk", "")
        return {
            "factor": f"{'Low' if pos else 'High'} activity in week {wk}",
            "evidence": f"{int(feature_value)} clicks in week {wk}",
        }
    if feature_name in ("clicks_mean", "clicks_std"):
        stat = "mean" if "mean" in feature_name else "variability"
        return {
            "factor": f"Weekly click {stat}",
            "evidence": f"{feature_value:.1f} clicks/week",
        }
    if feature_name == "activity_diversity_max":
        return {
            "factor": f"{'Narrow' if pos else 'Broad'} range of resource types used",
            "evidence": f"{int(feature_value)} distinct activity types accessed",
        }
    if feature_name == "active_days_total":
        return {
            "factor": f"{'Few' if pos else 'Many'} active study days",
            "evidence": f"{int(feature_value)} active days across the 6-week window",
        }
    if feature_name == "num_of_prev_attempts":
        if pos:
            return {
                "factor": "Multiple prior attempts at this module",
                "evidence": f"{int(feature_value)} previous attempt(s)",
            }
        return {
            "factor": "First attempt at this module",
            "evidence": "No prior attempts",
        }
    if feature_name == "studied_credits":
        return {
            "factor": f"{'Heavy' if pos else 'Moderate'} credit load",
            "evidence": f"{int(feature_value)} credits studied this presentation",
        }
    if feature_name == "days_registered_before_start":
        if pos:
            return {
                "factor": "Registered late relative to course start",
                "evidence": f"Registered {int(feature_value)} days before course start "
                            f"(negative = registered after start)",
            }
        return {
            "factor": "Registered well ahead of course start",
            "evidence": f"Registered {int(feature_value)} days in advance",
        }
    if feature_name == "imd_band":
        return {
            "factor": "Low-index-of-multiple-deprivation catchment" if pos
                     else "Higher-resource catchment",
            "evidence": f"IMD band = {feature_value}",
        }
    if feature_name == "highest_education":
        return {
            "factor": f"{'Lower' if pos else 'Higher'} prior education level",
            "evidence": f"Level = {feature_value}",
        }
    if feature_name == "age_band":
        return {
            "factor": f"Age band: {feature_value}",
            "evidence": "Age-band effect surfaced by the model",
        }
    if feature_name.startswith("disability"):
        return {
            "factor": "Declared disability on file",
            "evidence": "Check for accessibility support eligibility",
        }
    # Fallback: raw feature name
    return {
        "factor": f"{feature_name} ({'raises' if pos else 'lowers'} risk)",
        "evidence": f"value={feature_value}",
    }


def tier_for(score, threshold, base_rate):
    """Tier bands anchored on the model's decision threshold.

    HIGH   = model fired (score >= threshold) — advisor outreach required.
    MEDIUM = score below threshold but still > threshold/2 — monitor, light-touch nudge.
    LOW    = score below half the threshold — no action.
    Using a threshold-relative band keeps the three tiers populated regardless of
    whether the threshold happens to sit above or below the cohort base rate.
    """
    if score >= threshold:
        return "HIGH"
    if score >= threshold * 0.5:
        return "MEDIUM"
    return "LOW"


def action_for(tier):
    return {
        "HIGH": "Advisor outreach within 72h; offer study-plan reset and check for blockers.",
        "MEDIUM": "Include in weekly digest; send supportive nudge email with study resources.",
        "LOW": "No action — keep monitoring next checkpoint.",
    }[tier]


# ---------------------------------------------------------------------------
# Alert builder
# ---------------------------------------------------------------------------

def build_alert(row, risk_score, feature_names, shap_values, threshold, base_rate):
    tier = tier_for(risk_score, threshold, base_rate)

    # Top 3 drivers by |SHAP| that push toward disengagement (positive SHAP).
    # For a TP/borderline case most drivers are positive; for an FP we still show
    # the top signed drivers so the advisor can see *why* the model fired.
    order = np.argsort(-np.abs(shap_values))
    drivers = []
    for idx in order:
        fname = feature_names[idx]
        fval = _lookup_value(row, fname)
        drivers.append(driver_explanation(fname, fval, shap_values[idx], row))
        if len(drivers) >= 3:
            break

    return {
        "student_id": int(row["id_student"]),
        "course": f"{row['code_module']} {row['code_presentation']}",
        "triggered_at_week": 6,
        "risk_score": round(float(risk_score), 3),
        "risk_tier": tier,
        "threshold_at_generation": round(float(threshold), 3),
        "cohort_base_rate": round(float(base_rate), 3),
        "top_drivers": drivers,
        "recommended_action": action_for(tier),
    }


def _lookup_value(row, feature_name):
    """Best-effort lookup of the original (not one-hot-encoded) feature value for display."""
    if feature_name in row.index:
        return row[feature_name]
    # Handle one-hot: e.g. 'code_module_BBB' → the row's code_module == 'BBB' → 1 else 0
    for base in ("gender", "region", "disability", "code_module"):
        prefix = f"{base}_"
        if feature_name.startswith(prefix):
            return int(row.get(base) == feature_name[len(prefix):])
    return float("nan")


def render_markdown(alert):
    lines = [
        f"# Week-6 Disengagement Alert — {alert['risk_tier']}",
        "",
        f"**Student ID:** {alert['student_id']}  ",
        f"**Course:** {alert['course']}  ",
        f"**Risk score:** {alert['risk_score']:.2f} "
        f"(threshold {alert['threshold_at_generation']:.2f}, "
        f"cohort base rate {alert['cohort_base_rate']:.2f})  ",
        f"**Triggered:** end of Week 6",
        "",
        "## Top drivers",
    ]
    for i, d in enumerate(alert["top_drivers"], 1):
        lines.append(f"{i}. **{d['factor']}** — {d['evidence']}")
    lines += ["", f"**Recommended action:** {alert['recommended_action']}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_examples(df, y_prob, threshold):
    """Pick one TP, one FP, one borderline from the test set."""
    y = df["y"].to_numpy()
    pred = (y_prob >= threshold).astype(int)

    # True positive: high confidence, actually disengaged
    tp_mask = (pred == 1) & (y == 1)
    tp_idx = int(np.argmax(np.where(tp_mask, y_prob, -1)))

    # False positive: model was confident, but student actually passed
    fp_mask = (pred == 1) & (y == 0)
    fp_idx = int(np.argmax(np.where(fp_mask, y_prob, -1)))

    # Borderline: score closest to the threshold
    borderline_idx = int(np.argmin(np.abs(y_prob - threshold)))

    return {"true_positive": tp_idx, "false_positive": fp_idx, "borderline": borderline_idx}


def main():
    print("=== Task 2: Generating example advisor alerts ===\n")

    bundle = joblib.load(os.path.join(OUT_DIR, "disengagement_model.joblib"))
    calibrated = bundle["calibrated"]
    raw_clf = bundle["raw_clf"]
    pre = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    threshold = bundle["product_threshold"]

    features = pd.read_csv(os.path.join(OUT_DIR, "week6_features.csv"))
    test = features[features["code_presentation"].isin(TEST_PRES)].reset_index(drop=True)

    # Build model-ready frame — must match training column set.
    model_cols = (
        bundle["numeric_cols"] + bundle["ordinal_cols"] + bundle["onehot_cols"]
    )
    X_test = test[model_cols]
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    base_rate = float(test["y"].mean())

    print(f"  threshold={threshold:.3f}  base_rate={base_rate:.3f}  "
          f"test_n={len(test):,}")

    # SHAP values on the raw XGBoost model using the preprocessed matrix.
    X_test_enc = pre.transform(X_test)
    explainer = shap.TreeExplainer(raw_clf)
    shap_vals = explainer.shap_values(X_test_enc)

    examples = pick_examples(test, y_prob, threshold)

    for label, idx in examples.items():
        row = test.iloc[idx]
        alert = build_alert(
            row=row,
            risk_score=float(y_prob[idx]),
            feature_names=feature_names,
            shap_values=shap_vals[idx],
            threshold=threshold,
            base_rate=base_rate,
        )
        alert["example_case_type"] = label
        alert["actual_outcome"] = str(row["final_result"])

        json_path = os.path.join(ALERT_DIR, f"{label}.json")
        md_path = os.path.join(ALERT_DIR, f"{label}.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(alert, f, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(alert))
        print(f"  [{label}] student={alert['student_id']}  score={alert['risk_score']:.3f}  "
              f"tier={alert['risk_tier']}  actual={alert['actual_outcome']}")

    print(f"\n  Alerts written to: {ALERT_DIR}")
    print("=== Done. Run generate_task2_report.py next. ===")


if __name__ == "__main__":
    main()
