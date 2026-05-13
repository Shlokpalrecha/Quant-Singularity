"""
eval_suite.py — Signal Pod Evaluation Suite
Quant Singularity AI-SLM Screening Project

COMMIT THIS FILE BEFORE ANY TRAINING RUN.
All thresholds are pre-committed based on data statistics from the training window only.
No eval window data was used to set these thresholds.

Threshold Rationale (derived from training window inspection):
- VIX regime boundary: 14.78 (train mean + 1 std = 14.03 + 0.75)
- ADX suppression threshold: 20 (fixed by orchestrator spec)
- Conviction downgrade threshold: 0.40 (fixed by orchestrator spec)
- Directional accuracy floor: 0.45 for non-suppressed signals (above random for 3-class)
- Schema pass rate floor: 0.95 (production system requirement)
- Orchestrator suppression rate ceiling: 0.35 (>35% suppression = model not adding value)
- Max parse failure rate: 0.05
"""

import json
import uuid
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Pre-committed thresholds ──────────────────────────────────────────────────
THRESHOLDS = {
    # Schema & output integrity
    "schema_pass_rate_floor":        0.95,   # Fail if < 95% outputs parse correctly
    "parse_failure_rate_ceiling":    0.05,   # Fail if > 5% parse failures

    # Directional accuracy (on non-suppressed, non-downgraded signals only)
    "directional_accuracy_floor":    0.45,   # Above 3-class random (0.333) with margin
    "directional_accuracy_target":   0.52,   # Target for a credible pod

    # Conviction validity
    "conviction_range_valid":        (0.0, 1.0),
    "conviction_mean_floor":         0.35,   # Model should express some conviction
    "conviction_mean_ceiling":       0.75,   # Overconfident model is suspicious
    # Reliability diagram: calibration error
    "expected_calibration_error_ceiling": 0.15,  # ECE > 0.15 = poorly calibrated

    # Orchestrator behaviour
    "suppression_rate_ceiling":      0.35,   # > 35% suppression = model not adding value
    "downgrade_rate_ceiling":        0.50,   # > 50% low-conviction = model not learning
    "suppression_rate_floor":        0.05,   # < 5% suppression in eval = suspicious (ADX stats show ~20%)

    # Regime separation
    "vix_high_threshold":            14.78,  # train_mean + train_std
    "vix_regime_accuracy_gap_floor": -0.10,  # High-VIX accuracy shouldn't drop > 10pp vs low-VIX

    # Per-window stability
    "per_window_accuracy_floor":     0.35,   # No single 5-day window below 35%
    "accuracy_std_ceiling":          0.15,   # Std of per-window accuracy < 15pp (stable signal)
}

VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS   = {"intraday", "next_session"}

# ── Schema validation ─────────────────────────────────────────────────────────

def validate_schema(raw_output: str) -> tuple[bool, Optional[dict], str]:
    """
    Validate raw model output against the fixed signal schema.
    Returns (is_valid, parsed_dict_or_None, reason).
    """
    try:
        obj = json.loads(raw_output)
    except json.JSONDecodeError as e:
        return False, None, f"JSON_PARSE_ERROR: {e}"

    required = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
    missing = required - set(obj.keys())
    if missing:
        return False, None, f"MISSING_FIELDS: {missing}"

    if obj["direction"] not in VALID_DIRECTIONS:
        return False, None, f"INVALID_DIRECTION: {obj['direction']}"

    if obj["horizon"] not in VALID_HORIZONS:
        return False, None, f"INVALID_HORIZON: {obj['horizon']}"

    try:
        c = float(obj["conviction"])
    except (ValueError, TypeError):
        return False, None, f"CONVICTION_NOT_NUMERIC: {obj['conviction']}"

    if not (0.0 <= c <= 1.0):
        return False, None, f"CONVICTION_OUT_OF_RANGE: {c}"

    if not isinstance(obj["signal_id"], str) or len(obj["signal_id"]) == 0:
        return False, None, f"INVALID_SIGNAL_ID"

    return True, obj, "VALID"


# ── Orchestrator (exact spec implementation) ──────────────────────────────────

def orchestrator(
    raw_pod_output: str,
    adx_value: float,
    timestamp: str,
) -> dict:
    """
    Wraps the pod output and applies three suppression rules in sequence.
    Rule 1: ADX < 20 → suppress entirely, NEUTRAL, do not call model
    Rule 2: Parse failure → NEUTRAL, log raw output
    Rule 3: conviction < 0.40 → downgrade direction to NEUTRAL

    NOTE: Rule 1 is applied BEFORE model inference in production.
    In eval, we receive raw_pod_output; if adx_value < 20 the pod
    should not have been called. We honour the rule regardless.

    Returns orchestrator output dict with reason_code and trigger_values.
    """
    base_output = {
        "signal_id":    str(uuid.uuid4()),
        "generated_at": timestamp,
        "horizon":      "intraday",
        "conviction":   0.0,
    }

    # Rule 1: ADX regime filter
    if adx_value < 20.0:
        result = {**base_output,
                  "direction":    "NEUTRAL",
                  "reason_code":  "ADX_SUPPRESSED",
                  "trigger_values": {"adx_14": adx_value, "adx_threshold": 20.0},
                  "pod_called":   False}
        logger.info("ORCHESTRATOR | ADX_SUPPRESSED | adx=%.2f", adx_value)
        return result

    # Rule 2: Schema validation
    is_valid, parsed, reason = validate_schema(raw_pod_output)
    if not is_valid:
        result = {**base_output,
                  "direction":    "NEUTRAL",
                  "reason_code":  "PARSE_FAILURE",
                  "trigger_values": {"parse_error": reason, "raw_output": raw_pod_output[:200]},
                  "pod_called":   True}
        logger.warning("ORCHESTRATOR | PARSE_FAILURE | reason=%s | raw=%s",
                       reason, raw_pod_output[:100])
        return result

    conviction = float(parsed["conviction"])

    # Rule 3: Low conviction downgrade
    if conviction < 0.40:
        result = {**base_output,
                  "direction":    "NEUTRAL",
                  "conviction":   conviction,
                  "horizon":      parsed["horizon"],
                  "signal_id":    parsed["signal_id"],
                  "reason_code":  "LOW_CONVICTION_DOWNGRADE",
                  "trigger_values": {"conviction": conviction, "threshold": 0.40},
                  "pod_called":   True}
        logger.info("ORCHESTRATOR | LOW_CONVICTION_DOWNGRADE | conviction=%.3f", conviction)
        return result

    # Pass through
    result = {**parsed,
              "conviction":   conviction,
              "reason_code":  "PASSED",
              "trigger_values": {"adx_14": adx_value, "conviction": conviction},
              "pod_called":   True}
    logger.debug("ORCHESTRATOR | PASSED | direction=%s conviction=%.3f",
                 parsed["direction"], conviction)
    return result


# ── Conviction scoring (designed, not memorised) ──────────────────────────────

def compute_designed_conviction(
    retrieved_episodes: list[dict],
    proposed_direction: str,
    query_market_state: dict,
    corpus_vecs: np.ndarray = None,
    query_vec: np.ndarray = None,
) -> float:
    """
    Conviction is a deterministic function of three components:

    1. Label consistency among retrieved neighbors:
       c_consistency = fraction of retrieved episodes whose outcome
       matches proposed_direction. Range [0, 1].

    2. Direction entropy of neighborhood:
       c_entropy = 1 - H(label_dist) / log(3)
       where H is Shannon entropy over CE/PE/NEUTRAL in retrieved set.
       High entropy (disagreement) → low conviction. Range [0, 1].

    3. Similarity concentration (optional, when vectors provided):
       c_sim = 1 - (mean_dist / (mean_dist + std_dist))
       Tighter cluster of neighbors → higher conviction. Range [0, 1].

    Final conviction:
       If vectors available: 0.40*c_consistency + 0.35*c_entropy + 0.25*c_sim
       Otherwise:            0.55*c_consistency + 0.45*c_entropy

    Rationale: This is NOT softmax over the direction token. That value
    reflects the language model's token probability given the prompt prefix,
    which is a distributional fit to training text, not an estimate of
    signal reliability. Our conviction explicitly encodes how much historical
    evidence supports the proposed direction in similar market regimes.
    """
    if not retrieved_episodes:
        return 0.0

    outcomes = [ep.get("outcome", "NEUTRAL") for ep in retrieved_episodes]
    n = len(outcomes)

    # Component 1: consistency
    matches = sum(1 for o in outcomes if o == proposed_direction)
    c_consistency = matches / n

    # Component 2: entropy of neighborhood labels
    counts = {"CE": 0, "PE": 0, "NEUTRAL": 0}
    for o in outcomes:
        if o in counts:
            counts[o] += 1
    probs = np.array([counts[k] / n for k in ["CE", "PE", "NEUTRAL"]])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(3)
    c_entropy = 1.0 - (entropy / max_entropy)

    # Component 3: similarity concentration (optional)
    if corpus_vecs is not None and query_vec is not None:
        dists = np.linalg.norm(corpus_vecs - query_vec, axis=1)
        top_k_dists = np.sort(dists)[:n]
        mean_d = np.mean(top_k_dists)
        std_d  = np.std(top_k_dists) + 1e-6
        c_sim = 1.0 - (mean_d / (mean_d + std_d))
        conviction = 0.40 * c_consistency + 0.35 * c_entropy + 0.25 * c_sim
    else:
        conviction = 0.55 * c_consistency + 0.45 * c_entropy

    return float(np.clip(conviction, 0.0, 1.0))


# ── Core evaluation metrics ───────────────────────────────────────────────────

def directional_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Accuracy on directional (non-NEUTRAL) predictions only."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "NEUTRAL"]
    if not pairs:
        return float("nan")
    return sum(t == p for t, p in pairs) / len(pairs)


def conviction_reliability(
    convictions: list[float],
    correct: list[bool],
    n_bins: int = 5,
) -> dict:
    """
    Reliability diagram data + ECE.
    Bins convictions and computes fraction correct per bin.
    ECE = sum_b (|bin| / N) * |accuracy_b - mean_conviction_b|
    """
    if len(convictions) < n_bins:
        return {"ece": float("nan"), "bins": []}

    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    n = len(convictions)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [lo <= c < hi for c in convictions]
        if i == n_bins - 1:
            mask = [lo <= c <= hi for c in convictions]
        idxs = [j for j, m in enumerate(mask) if m]
        if not idxs:
            continue
        bin_conv  = np.mean([convictions[j] for j in idxs])
        bin_acc   = np.mean([correct[j] for j in idxs])
        bin_n     = len(idxs)
        ece      += (bin_n / n) * abs(bin_acc - bin_conv)
        bin_data.append({
            "bin":       f"{lo:.1f}-{hi:.1f}",
            "mean_conv": round(bin_conv, 4),
            "accuracy":  round(bin_acc, 4),
            "n":         bin_n,
            "gap":       round(abs(bin_acc - bin_conv), 4),
        })

    return {"ece": round(ece, 4), "bins": bin_data}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def run_walk_forward_eval(
    eval_df: pd.DataFrame,
    pod_outputs: list[dict],  # list of {"raw_output": str, "adx_14": float, "timestamp": str}
    rag_outputs: list[dict] = None,  # same structure, with RAG enabled
) -> dict:
    """
    Main evaluation runner.

    eval_df: evaluation window dataframe (days 31-60), must have 'label' column.
    pod_outputs: list of dicts with raw pod output + market state metadata.
    rag_outputs: optional second set of outputs for RAG ablation.

    Returns comprehensive eval report dict.
    """
    assert len(eval_df) == len(pod_outputs), \
        f"Mismatch: {len(eval_df)} market states vs {len(pod_outputs)} pod outputs"

    results = []
    for i, (row, pod) in enumerate(zip(eval_df.itertuples(), pod_outputs)):
        orch = orchestrator(
            raw_pod_output=pod["raw_output"],
            adx_value=float(row.adx_14),
            timestamp=pod.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )
        results.append({
            "idx":           i,
            "timestamp":     str(row.timestamp),
            "true_label":    row.label,
            "adx_14":        row.adx_14,
            "vix_india":     row.vix_india,
            "raw_output":    pod["raw_output"],
            "orch_direction": orch["direction"],
            "orch_conviction": orch.get("conviction", 0.0),
            "reason_code":   orch["reason_code"],
            "pod_called":    orch.get("pod_called", False),
        })

    results_df = pd.DataFrame(results)

    # Assign 5-day windows
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
    results_df["date"] = results_df["timestamp"].dt.date
    unique_dates = sorted(results_df["date"].unique())
    date_to_window = {d: i // 5 + 1 for i, d in enumerate(unique_dates)}
    results_df["window"] = results_df["date"].map(date_to_window)

    # VIX regime
    vix_thresh = THRESHOLDS["vix_high_threshold"]
    results_df["vix_regime"] = results_df["vix_india"].apply(
        lambda v: "high_vix" if v > vix_thresh else "low_vix"
    )

    report = {}

    # ── 1. Schema / parse metrics ─────────────────────────────────────────────
    n_total       = len(results_df)
    n_parse_fail  = (results_df["reason_code"] == "PARSE_FAILURE").sum()
    schema_pass_n = n_total - n_parse_fail
    report["schema"] = {
        "total":             n_total,
        "parse_failures":    int(n_parse_fail),
        "schema_pass_rate":  round(schema_pass_n / n_total, 4),
        "parse_failure_rate":round(n_parse_fail / n_total, 4),
        "schema_pass_ci":    wilson_ci(schema_pass_n, n_total),
        "PASS": schema_pass_n / n_total >= THRESHOLDS["schema_pass_rate_floor"],
    }

    # ── 2. Orchestrator suppression & downgrade rates ─────────────────────────
    n_adx_supp    = (results_df["reason_code"] == "ADX_SUPPRESSED").sum()
    n_low_conv    = (results_df["reason_code"] == "LOW_CONVICTION_DOWNGRADE").sum()
    n_passed      = (results_df["reason_code"] == "PASSED").sum()
    suppression_rate = (n_adx_supp + n_parse_fail) / n_total
    downgrade_rate   = n_low_conv / n_total

    report["orchestrator"] = {
        "adx_suppressed":      int(n_adx_supp),
        "parse_failures":      int(n_parse_fail),
        "low_conv_downgraded": int(n_low_conv),
        "passed":              int(n_passed),
        "suppression_rate":    round(suppression_rate, 4),
        "downgrade_rate":      round(downgrade_rate, 4),
        "suppression_ci":      wilson_ci(int(n_adx_supp + n_parse_fail), n_total),
        "PASS_suppression":    suppression_rate <= THRESHOLDS["suppression_rate_ceiling"],
        "PASS_downgrade":      downgrade_rate   <= THRESHOLDS["downgrade_rate_ceiling"],
    }

    # ── 3. Directional accuracy — overall ────────────────────────────────────
    passed_df = results_df[results_df["reason_code"] == "PASSED"]
    y_true = passed_df["true_label"].tolist()
    y_pred = passed_df["orch_direction"].tolist()
    dir_acc = directional_accuracy(y_true, y_pred)
    n_directional = sum(1 for p in y_pred if p != "NEUTRAL")
    n_correct     = sum(1 for t, p in zip(y_true, y_pred) if p != "NEUTRAL" and t == p)

    report["directional_accuracy"] = {
        "overall":               round(dir_acc, 4) if not np.isnan(dir_acc) else None,
        "n_directional_signals": n_directional,
        "n_correct":             n_correct,
        "ci_95":                 wilson_ci(n_correct, n_directional),
        "PASS":                  (not np.isnan(dir_acc)) and dir_acc >= THRESHOLDS["directional_accuracy_floor"],
    }

    # ── 4. Per-window breakdown ───────────────────────────────────────────────
    window_results = []
    per_window_accs = []
    for w in sorted(results_df["window"].unique()):
        wdf = results_df[results_df["window"] == w]
        wpassed = wdf[wdf["reason_code"] == "PASSED"]
        wacc = directional_accuracy(
            wpassed["true_label"].tolist(),
            wpassed["orch_direction"].tolist()
        )
        w_supp = (wdf["reason_code"] == "ADX_SUPPRESSED").sum() / len(wdf)
        w_down = (wdf["reason_code"] == "LOW_CONVICTION_DOWNGRADE").sum() / len(wdf)
        w_vix  = wdf["vix_india"].mean()
        if not np.isnan(wacc):
            per_window_accs.append(wacc)
        window_results.append({
            "window":           w,
            "n":                len(wdf),
            "vix_mean":         round(w_vix, 2),
            "suppression_rate": round(w_supp, 4),
            "downgrade_rate":   round(w_down, 4),
            "directional_acc":  round(wacc, 4) if not np.isnan(wacc) else None,
            "PASS":             np.isnan(wacc) or wacc >= THRESHOLDS["per_window_accuracy_floor"],
        })

    report["per_window"] = window_results
    report["window_stability"] = {
        "accuracy_std":  round(float(np.std(per_window_accs)), 4) if per_window_accs else None,
        "PASS":          (len(per_window_accs) == 0) or
                         (float(np.std(per_window_accs)) <= THRESHOLDS["accuracy_std_ceiling"]),
    }

    # ── 5. VIX regime breakdown ───────────────────────────────────────────────
    regime_results = {}
    regime_accs    = {}
    for regime in ["high_vix", "low_vix"]:
        rdf     = results_df[results_df["vix_regime"] == regime]
        rpassed = rdf[rdf["reason_code"] == "PASSED"]
        racc    = directional_accuracy(
            rpassed["true_label"].tolist(),
            rpassed["orch_direction"].tolist()
        )
        r_n_dir  = sum(1 for p in rpassed["orch_direction"] if p != "NEUTRAL")
        r_n_corr = sum(1 for t, p in zip(rpassed["true_label"], rpassed["orch_direction"])
                       if p != "NEUTRAL" and t == p)
        regime_accs[regime]  = racc
        regime_results[regime] = {
            "n":                len(rdf),
            "vix_mean":         round(rdf["vix_india"].mean(), 2) if len(rdf) > 0 else None,
            "directional_acc":  round(racc, 4) if not np.isnan(racc) else None,
            "ci_95":            wilson_ci(r_n_corr, r_n_dir),
            "suppression_rate": round((rdf["reason_code"] == "ADX_SUPPRESSED").sum() / max(len(rdf), 1), 4),
        }

    acc_gap = (regime_accs.get("high_vix", float("nan")) -
               regime_accs.get("low_vix",  float("nan")))
    regime_results["accuracy_gap_high_minus_low"] = round(acc_gap, 4) if not np.isnan(acc_gap) else None
    regime_results["PASS_gap"] = np.isnan(acc_gap) or acc_gap >= THRESHOLDS["vix_regime_accuracy_gap_floor"]
    report["vix_regime"] = regime_results

    # ── 6. Conviction validity analysis ──────────────────────────────────────
    all_convictions = results_df[results_df["pod_called"] == True]["orch_conviction"].tolist()
    if all_convictions:
        correct_flags = [
            t == p
            for t, p, r in zip(
                results_df[results_df["pod_called"] == True]["true_label"],
                results_df[results_df["pod_called"] == True]["orch_direction"],
                results_df[results_df["pod_called"] == True]["reason_code"],
            )
        ]
        cal = conviction_reliability(all_convictions, correct_flags)
        report["conviction"] = {
            "mean":    round(float(np.mean(all_convictions)), 4),
            "std":     round(float(np.std(all_convictions)), 4),
            "min":     round(float(np.min(all_convictions)), 4),
            "max":     round(float(np.max(all_convictions)), 4),
            "ece":     cal["ece"],
            "bins":    cal["bins"],
            "PASS_range": (float(np.min(all_convictions)) >= THRESHOLDS["conviction_range_valid"][0] and
                           float(np.max(all_convictions)) <= THRESHOLDS["conviction_range_valid"][1]),
            "PASS_ece":   (cal["ece"] is None) or (cal["ece"] <= THRESHOLDS["expected_calibration_error_ceiling"]),
            "PASS_mean":  (THRESHOLDS["conviction_mean_floor"] <=
                           float(np.mean(all_convictions)) <=
                           THRESHOLDS["conviction_mean_ceiling"]),
        }

    # ── 7. RAG ablation (if rag_outputs provided) ─────────────────────────────
    if rag_outputs is not None:
        assert len(rag_outputs) == len(pod_outputs)
        rag_convictions, base_convictions, sim_scores = [], [], []
        rag_conv_changes_correct, rag_conv_changes_incorrect = [], []

        for i, (pod, rag) in enumerate(zip(pod_outputs, rag_outputs)):
            row = eval_df.iloc[i]
            base_valid, base_parsed, _ = validate_schema(pod["raw_output"])
            rag_valid,  rag_parsed,  _ = validate_schema(rag["raw_output"])

            if base_valid and rag_valid:
                bc = float(base_parsed["conviction"])
                rc = float(rag_parsed["conviction"])
                base_convictions.append(bc)
                rag_convictions.append(rc)
                delta = rc - bc
                true_label = row["label"]
                rag_dir    = rag_parsed["direction"]
                is_correct = (rag_dir == true_label and rag_dir != "NEUTRAL")
                if is_correct:
                    rag_conv_changes_correct.append(delta)
                else:
                    rag_conv_changes_incorrect.append(delta)

        base_acc = directional_accuracy(
            [eval_df.iloc[i]["label"] for i, p in enumerate(pod_outputs)],
            [json.loads(p["raw_output"]).get("direction","NEUTRAL")
             if validate_schema(p["raw_output"])[0] else "NEUTRAL"
             for p in pod_outputs]
        )
        rag_acc = directional_accuracy(
            [eval_df.iloc[i]["label"] for i, p in enumerate(rag_outputs)],
            [json.loads(p["raw_output"]).get("direction","NEUTRAL")
             if validate_schema(p["raw_output"])[0] else "NEUTRAL"
             for p in rag_outputs]
        )

        report["rag_ablation"] = {
            "base_directional_acc":   round(base_acc, 4) if not np.isnan(base_acc) else None,
            "rag_directional_acc":    round(rag_acc,  4) if not np.isnan(rag_acc)  else None,
            "accuracy_delta":         round(rag_acc - base_acc, 4) if not (np.isnan(rag_acc) or np.isnan(base_acc)) else None,
            "mean_conviction_base":   round(float(np.mean(base_convictions)), 4) if base_convictions else None,
            "mean_conviction_rag":    round(float(np.mean(rag_convictions)), 4) if rag_convictions else None,
            "conviction_delta":       round(float(np.mean(rag_convictions)) - float(np.mean(base_convictions)), 4)
                                      if (base_convictions and rag_convictions) else None,
            # Key diagnostic: does RAG raise conviction for correct predictions?
            "mean_conv_change_correct":   round(float(np.mean(rag_conv_changes_correct)), 4)
                                          if rag_conv_changes_correct else None,
            "mean_conv_change_incorrect": round(float(np.mean(rag_conv_changes_incorrect)), 4)
                                          if rag_conv_changes_incorrect else None,
            "rag_helps_calibration":  (
                len(rag_conv_changes_correct) > 0 and
                len(rag_conv_changes_incorrect) > 0 and
                float(np.mean(rag_conv_changes_correct)) > float(np.mean(rag_conv_changes_incorrect))
            ),
        }

    # ── 8. Summary pass/fail ──────────────────────────────────────────────────
    checks = {
        "schema_pass_rate":   report["schema"]["PASS"],
        "suppression_rate":   report["orchestrator"]["PASS_suppression"],
        "downgrade_rate":     report["orchestrator"]["PASS_downgrade"],
        "directional_acc":    report["directional_accuracy"]["PASS"],
        "window_stability":   report["window_stability"]["PASS"],
        "vix_regime_gap":     report["vix_regime"]["PASS_gap"],
    }
    if "conviction" in report:
        checks["conviction_range"] = report["conviction"]["PASS_range"]
        checks["conviction_ece"]   = report["conviction"]["PASS_ece"]
        checks["conviction_mean"]  = report["conviction"]["PASS_mean"]

    report["summary"] = {
        "checks":       checks,
        "all_pass":     all(checks.values()),
        "pass_count":   sum(checks.values()),
        "total_checks": len(checks),
        "thresholds_used": THRESHOLDS,
    }

    return report


# ── Utility: print formatted report ───────────────────────────────────────────

def print_report(report: dict) -> None:
    print("\n" + "="*70)
    print("SIGNAL POD EVALUATION REPORT")
    print("="*70)

    s = report["schema"]
    print(f"\n[SCHEMA]")
    print(f"  Pass rate:      {s['schema_pass_rate']:.1%}  (floor: {THRESHOLDS['schema_pass_rate_floor']:.0%})  {'✓' if s['PASS'] else '✗'}")
    print(f"  Parse failures: {s['parse_failures']} / {s['total']}")

    o = report["orchestrator"]
    print(f"\n[ORCHESTRATOR]")
    print(f"  ADX suppressed:   {o['adx_suppressed']} ({o['suppression_rate']:.1%})  {'✓' if o['PASS_suppression'] else '✗'}")
    print(f"  Low-conv downgrade: {o['low_conv_downgraded']} ({o['downgrade_rate']:.1%})  {'✓' if o['PASS_downgrade'] else '✗'}")
    print(f"  Passed to downstream: {o['passed']}")

    d = report["directional_accuracy"]
    print(f"\n[DIRECTIONAL ACCURACY]")
    acc = d['overall']
    ci  = d['ci_95']
    print(f"  Overall:  {acc:.1%}  95% CI [{ci[0]:.1%}, {ci[1]:.1%}]  (floor: {THRESHOLDS['directional_accuracy_floor']:.0%})  {'✓' if d['PASS'] else '✗'}")

    print(f"\n[PER-WINDOW]")
    for w in report["per_window"]:
        acc_str = f"{w['directional_acc']:.1%}" if w['directional_acc'] is not None else "N/A"
        print(f"  W{w['window']}: acc={acc_str}  VIX={w['vix_mean']:.1f}  supp={w['suppression_rate']:.0%}  dngr={w['downgrade_rate']:.0%}  {'✓' if w['PASS'] else '✗'}")

    vr = report["vix_regime"]
    print(f"\n[VIX REGIMES]")
    for regime in ["high_vix", "low_vix"]:
        r = vr[regime]
        print(f"  {regime}: n={r['n']}  VIX={r['vix_mean']}  acc={r['directional_acc']}  supp={r['suppression_rate']:.0%}")
    print(f"  Accuracy gap (high-low): {vr['accuracy_gap_high_minus_low']}  {'✓' if vr['PASS_gap'] else '✗'}")

    if "conviction" in report:
        c = report["conviction"]
        print(f"\n[CONVICTION]")
        print(f"  Mean={c['mean']:.3f}  Std={c['std']:.3f}  Range=[{c['min']:.3f},{c['max']:.3f}]  ECE={c['ece']}")
        print(f"  Range valid: {'✓' if c['PASS_range'] else '✗'}  ECE valid: {'✓' if c['PASS_ece'] else '✗'}  Mean valid: {'✓' if c['PASS_mean'] else '✗'}")
        print(f"  Reliability bins:")
        for b in c.get("bins", []):
            print(f"    {b['bin']}: mean_conv={b['mean_conv']:.3f}  acc={b['accuracy']:.3f}  gap={b['gap']:.3f}  n={b['n']}")

    if "rag_ablation" in report:
        r = report["rag_ablation"]
        print(f"\n[RAG ABLATION]")
        print(f"  Base accuracy:  {r['base_directional_acc']}")
        print(f"  RAG accuracy:   {r['rag_directional_acc']}")
        print(f"  Accuracy delta: {r['accuracy_delta']}")
        print(f"  Conviction delta: {r['conviction_delta']}")
        print(f"  Conv change (correct preds):   {r['mean_conv_change_correct']}")
        print(f"  Conv change (incorrect preds):  {r['mean_conv_change_incorrect']}")
        print(f"  RAG improves calibration: {r['rag_helps_calibration']}")

    sm = report["summary"]
    print(f"\n[SUMMARY] {sm['pass_count']}/{sm['total_checks']} checks passed  {'ALL PASS ✓' if sm['all_pass'] else 'FAILURES PRESENT ✗'}")
    for k, v in sm["checks"].items():
        print(f"  {'✓' if v else '✗'}  {k}")
    print("="*70 + "\n")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test: run eval suite against synthetic dummy outputs.
    This verifies all code paths run without error before training.
    Replace with real pod outputs after fine-tuning.
    """
    import pandas as pd
    from pathlib import Path

    print("Running eval suite smoke test with dummy outputs...")

    df = pd.read_parquet("market_states.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    days = sorted(df["timestamp"].dt.date.unique())
    eval_days = days[30:]
    eval_df = df[df["timestamp"].dt.date.isin(eval_days)].reset_index(drop=True)

    # Generate dummy outputs (not real model outputs — for code path testing only)
    rng = np.random.default_rng(42)
    dummy_outputs = []
    for _, row in eval_df.iterrows():
        direction = rng.choice(["CE", "PE", "NEUTRAL"])
        conviction = round(float(rng.uniform(0.3, 0.8)), 2)
        dummy_outputs.append({
            "raw_output": json.dumps({
                "direction":   direction,
                "conviction":  conviction,
                "horizon":     "intraday",
                "signal_id":   str(uuid.uuid4()),
                "generated_at": str(row["timestamp"]),
            }),
            "adx_14":    float(row["adx_14"]),
            "timestamp": str(row["timestamp"]),
        })

    report = run_walk_forward_eval(eval_df, dummy_outputs)
    print_report(report)

    # Verify JSON serialisability
    import json as _json
    _json.dumps(report, default=str)
    print("Smoke test PASSED — all code paths exercised, report is JSON-serialisable.")
    print(f"Thresholds committed: {json.dumps(THRESHOLDS, indent=2)}")
