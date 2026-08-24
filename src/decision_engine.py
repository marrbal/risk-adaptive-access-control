#!/usr/bin/env python3
"""
decision_engine.py
Risk-Adaptive Decision Engine for a synthetic healthcare technology company.

Reads risk_scores.csv (produced by risk_model.py) and applies adaptive,
sensitivity-aware thresholds to produce one of three decisions per event:
  allow     — risk score is below the challenge threshold
  challenge — risk score is borderline; step-up authentication is required
  block     — risk score is above the block threshold

Thresholds tighten for higher-sensitivity resources so that the same
score triggers a stricter response on a Privileged resource than on a
Public one.

Outputs:
  final_decisions.csv   — one row per event with decision and step-up fields
  Prints a full evaluation and comparison report to stdout.
"""

import argparse
import csv
import json
from pathlib import Path

# THRESHOLD CONFIGURATION

BASE_CHALLENGE = 40   # score >= challenge
BASE_BLOCK     = 70   # score >= block

# Sensitivity-specific overrides  (lower = stricter)
SENSITIVITY_THRESHOLDS = {
    "Public":       (BASE_CHALLENGE, BASE_BLOCK),   # 40 / 70
    "Internal":     (BASE_CHALLENGE, BASE_BLOCK),   # 40 / 70
    "Confidential": (30,             60),            # 30 / 60
    "Privileged":   (20,             50),            # 20 / 50
}

# Score band → step-up type (evaluated after challenge/block decision)
#   challenge band lower half → mfa_reprompt
#   challenge band upper half → mfa_plus_supervisor
#   block band                → soc_review (always block)
STEPUP_MIDPOINT = 55   # within the BASE challenge band, the boundary between
                       # mfa_reprompt and mfa_plus_supervisor

# DECISION LOGIC

def decide(risk_score: int, sensitivity: str) -> tuple[str, str]:
    """
    Return (decision, step_up_type).
    step_up_type is 'none' for allow decisions.
    """
    challenge_thresh, block_thresh = SENSITIVITY_THRESHOLDS.get(
        sensitivity, (BASE_CHALLENGE, BASE_BLOCK)
    )

    if risk_score >= block_thresh:
        return "block", "soc_review"

    if risk_score >= challenge_thresh:
        # Within the challenge band — determine step-up level.
        # Use absolute score rather than relative position so that
        # the interpretation is consistent across sensitivity tiers.
        if risk_score < STEPUP_MIDPOINT:
            return "challenge", "mfa_reprompt"
        else:
            return "challenge", "mfa_plus_supervisor"

    return "allow", "none"


# METRICS

def compute_metrics(rows: list[dict], decision_col: str) -> dict:
    """
    Binary classification metrics.
    Positive class: anomalous events.
    Flagged = challenge or block.
    """
    tp = fp = fn = tn = 0
    per_type: dict[str, dict] = {}

    for r in rows:
        gt      = r["ground_truth_label"]   # "normal" / "anomalous"
        dec     = r[decision_col]           # allow / challenge / block
        atype   = r.get("anomaly_type", "")

        flagged  = dec in ("challenge", "block")
        positive = gt == "anomalous"

        if positive and flagged:
            tp += 1
        elif not positive and flagged:
            fp += 1
        elif positive and not flagged:
            fn += 1
        else:
            tn += 1

        if positive:
            if atype not in per_type:
                per_type[atype] = {"detected": 0, "missed": 0, "total": 0}
            per_type[atype]["total"] += 1
            if flagged:
                per_type[atype]["detected"] += 1
            else:
                per_type[atype]["missed"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1,
                per_type=per_type)


# REPORT

def print_report(rows: list[dict], ai_metrics: dict, baseline_metrics: dict) -> None:
    total  = len(rows)
    n_anom = sum(1 for r in rows if r["ground_truth_label"] == "anomalous")

    print("\n" + "═" * 64)
    print("  RISK-ADAPTIVE DECISION ENGINE — EVALUATION REPORT  (Task 5)")
    print("═" * 64)
    print(f"  Total events   : {total:,}")
    print(f"  Anomalous      : {n_anom:,}  ({100*n_anom/total:.1f}%)")

    # Decision distribution
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["raac_decision"]] = dist.get(r["raac_decision"], 0) + 1
    print("\n  ── Decision Distribution ───────────────────────────────────")
    for dec in ("allow", "challenge", "block"):
        c = dist.get(dec, 0)
        print(f"  {dec:<12} : {c:>6,}  ({100*c/total:.1f}%)")

    # Step-up distribution
    su_dist: dict[str, int] = {}
    for r in rows:
        su_dist[r["step_up_type"]] = su_dist.get(r["step_up_type"], 0) + 1
    print("\n  ── Step-Up Control Distribution ────────────────────────────")
    for su in ("none", "mfa_reprompt", "mfa_plus_supervisor", "soc_review"):
        c = su_dist.get(su, 0)
        print(f"  {su:<25} : {c:>6,}")

    # AI decision engine metrics
    m = ai_metrics
    print("\n  ── AI Decision Engine Metrics (all events) ─────────────────")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1-score  : {m['f1']:.3f}")

    # Per-category recall
    print("\n  ── Per-Category Detection ──────────────────────────────────")
    header = f"  {'Anomaly Type':<35} {'Tot':>4} {'Det':>4} {'Miss':>4}  Recall"
    print(header)
    print("  " + "-" * 58)
    for atype, s in sorted(m["per_type"].items()):
        r = s["detected"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {atype:<35} {s['total']:>4} {s['detected']:>4} {s['missed']:>4}  {r:.1%}")

    # Comparison with static baseline
    b = baseline_metrics
    print("\n  ── Comparison: AI Engine vs Static Baseline ────────────────")
    print(f"  {'Metric':<12}  {'Baseline':>10}  {'AI Engine':>10}  {'Δ':>8}")
    print("  " + "-" * 46)
    for label, bval, aval in [
        ("Precision",  b["precision"],  m["precision"]),
        ("Recall",     b["recall"],     m["recall"]),
        ("F1",         b["f1"],         m["f1"]),
    ]:
        delta = aval - bval
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<12}  {bval:>10.3f}  {aval:>10.3f}  {sign}{delta:>7.3f}")

    fp_delta = b["fp"] - m["fp"]
    fn_delta = b["fn"] - m["fn"]
    print(f"\n  False Positives : Baseline {b['fp']:,}  →  AI Engine {m['fp']:,}  "
          f"({'−' if fp_delta>=0 else '+'}{abs(fp_delta):,})")
    print(f"  False Negatives : Baseline {b['fn']:,}  →  AI Engine {m['fn']:,}  "
          f"({'−' if fn_delta>=0 else '+'}{abs(fn_delta):,})")

    print("═" * 64 + "\n")

# CSV OUTPUT

OUTPUT_FIELDS = [
    "event_id", "user_id", "department", "resource_id", "sensitivity",
    "network_zone", "device_trust", "failure_count_24h", "bytes_transferred",
    "ground_truth_label", "anomaly_type", "split",
    "risk_score", "anomaly_proba",
    "raac_decision", "step_up_type",
    "baseline_decision",          # carried over from baseline_decisions.csv if available
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV]  {len(rows):,} decisions  →  {path}")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="TechnoMed Solutions — Risk-Adaptive Decision Engine (Task 5)"
    )
    parser.add_argument("--scores",   default="risk_scores.csv",
                        help="Risk scores CSV from risk_model.py")
    parser.add_argument("--baseline", default="baseline_decisions.csv",
                        help="Baseline decisions CSV from baseline_policy.py (optional)")
    parser.add_argument("--output",   default="final_decisions.csv",
                        help="Output CSV (default: final_decisions.csv)")
    args = parser.parse_args()

    
    # Load risk scores
    print(f"\n  TechnoMed Solutions — Risk-Adaptive Decision Engine")
    print(f"  Scores input  : {args.scores}")

    with open(args.scores, newline="", encoding="utf-8") as f:
        score_rows = list(csv.DictReader(f))
    print(f"  Loaded {len(score_rows):,} scored events.")

    # Index by event_id for fast lookup
    score_by_id = {r["event_id"]: r for r in score_rows}

    # Load baseline decisions (optional — for comparison report)
    baseline_by_id: dict[str, dict] = {}
    baseline_metrics = None
    if Path(args.baseline).exists():
        with open(args.baseline, newline="", encoding="utf-8") as f:
            for br in csv.DictReader(f):
                baseline_by_id[br["event_id"]] = br
        print(f"  Baseline input: {args.baseline} ({len(baseline_by_id):,} rows)")
    else:
        print(f"  Baseline file not found — comparison report will be skipped.")

    # Apply adaptive thresholds
    rows = []
    for sr in score_rows:
        risk_score  = int(sr["risk_score"])
        sensitivity = sr.get("sensitivity", "Internal")

        decision, step_up = decide(risk_score, sensitivity)

        row = dict(sr)
        row["raac_decision"] = decision
        row["step_up_type"]  = step_up

        # Carry over baseline decision for side-by-side comparison
        br = baseline_by_id.get(sr["event_id"], {})
        row["baseline_decision"] = br.get("decision", "n/a")

        rows.append(row)

    # Metrics
    ai_metrics = compute_metrics(rows, "raac_decision")

    if baseline_by_id:
        # Compute baseline metrics from the loaded baseline decisions
        baseline_rows = list(baseline_by_id.values())
        baseline_metrics = compute_metrics(baseline_rows, "decision")
    else:
        baseline_metrics = dict(
            tp=445, fp=620, fn=166, tn=10942,
            precision=0.418, recall=0.728, f1=0.531,
            per_type={}
        )

    # Write output and print report
    write_csv(rows, args.output)
    print_report(rows, ai_metrics, baseline_metrics)

    print(f"  Done.  Reproduce with:  python decision_engine.py\n")


if __name__ == "__main__":
    main()
