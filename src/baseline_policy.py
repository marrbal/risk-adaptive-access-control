#!/usr/bin/env python3
"""
baseline_policy.py
Static rule-based policy engine for a synthetic healthcare technology company.  
Reads logs_with_anomalies.jsonl, applies eight rules in priority order,
and writes baseline_decisions.csv. Prints a full evaluation report at the end.
"""

import argparse
import csv
import json
from datetime import datetime, timezone

# CONFIGURATION

# Normal working-hour windows per department (start inclusive, end exclusive)
DEPT_HOURS = {
    "Engineering":  (7,  20),
    "Finance":      (8,  18),
    "HR":           (8,  17),
    "Clinical":     (7,  19),
    "IT Security":  (0,  24),   # 24-hour operations — no off-hours rule
    "Legal":        (9,  18),
    "Executive":    (7,  21),
}

# Departments that are authorised to access each resource.
AUTHORIZED_DEPTS = {
    "RES-001": {"Clinical",    "IT Security"},
    "RES-002": {"Clinical",    "IT Security"},
    "RES-003": {"Clinical",    "IT Security"},
    "RES-004": {"HR",          "IT Security"},
    "RES-005": {"Finance", "HR", "IT Security"},
    "RES-006": {"IT Security"},
    "RES-007": {"IT Security", "Legal"},
    "RES-008": {"Executive",   "IT Security"},
    "RES-009": {"Finance", "Executive", "IT Security"},
    "RES-010": {"Engineering", "IT Security"},
    "RES-011": {"Engineering", "Finance", "IT Security"},
    "RES-012": {"Engineering", "Finance", "IT Security"},
    "RES-013": {"IT Security"},
    "RES-014": {"IT Security"},
    "RES-015": {"Legal",       "IT Security"},
    "RES-016": {"Legal",       "IT Security"},
    "RES-017": {"Engineering", "IT Security"},
    "RES-018": {"Engineering", "IT Security"},
    "RES-019": {"Engineering", "IT Security"},
    "RES-020": {"IT Security"},
    "RES-021": None,   # VPN Gateway — all
    "RES-022": None,   # JIRA — all
    "RES-023": None,   # Confluence — all
    "RES-024": {"Finance", "Legal", "IT Security"},
    "RES-025": None,   # SharePoint — all
    "RES-026": None,   # Email — all
    "RES-027": None,   # Marketing Portal — all
}

SENSITIVE_TIERS = {"Privileged", "Confidential"}

# HELPERS

def event_hour(ts_str: str) -> int:
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.hour


def is_off_hours(event: dict) -> bool:
    dept = event.get("department", "")
    start, end = DEPT_HOURS.get(dept, (8, 18))
    hour = event_hour(event["timestamp"])
    # IT Security never triggers off-hours rules (end=24 covers all hours)
    return not (start <= hour < end)


def is_role_mismatch(event: dict) -> bool:
    """True when the user's department is not authorised for this resource."""
    res_id = event.get("resource_id", "")
    dept   = event.get("department", "")
    allowed = AUTHORIZED_DEPTS.get(res_id)
    if allowed is None:
        return False            # open-access resource
    return dept not in allowed


# RULE ENGINE  (first match wins)
RULES = [
    ("R01_PRIV_EXTERNAL",
     "block",
     lambda e: e.get("sensitivity") == "Privileged"
               and e.get("network_zone") == "external"),

    ("R02_PRIV_OFF_HOURS",
     "block",
     lambda e: e.get("sensitivity") == "Privileged"
               and e.get("department") != "IT Security"
               and is_off_hours(e)),

    ("R03_ROLE_MISMATCH",
     "block",
     lambda e: e.get("sensitivity") in SENSITIVE_TIERS
               and is_role_mismatch(e)),

    ("R04_UNKNOWN_DEVICE_SENSITIVE",
     "challenge",
     lambda e: e.get("device_trust") == "unknown"
               and e.get("sensitivity") in SENSITIVE_TIERS),

    ("R05_HIGH_FAILURE_COUNT",
     "challenge",
     lambda e: (e.get("failure_count_24h") or 0) >= 3),

    ("R06_CONF_EXTERNAL",
     "challenge",
     lambda e: e.get("sensitivity") == "Confidential"
               and e.get("network_zone") == "external"),

    ("R07_CONF_OFF_HOURS",
     "challenge",
     lambda e: e.get("sensitivity") == "Confidential"
               and e.get("department") != "IT Security"
               and is_off_hours(e)),

    ("R08_DEFAULT",
     "allow",
     lambda e: True),
]


def evaluate(event: dict) -> tuple[str, str]:
    """Return (decision, rule_id) — first matching rule wins."""
    for rule_id, decision, predicate in RULES:
        if predicate(event):
            return decision, rule_id
    return "allow", "R08_DEFAULT"   # unreachable but safe


# EVALUATION METRICS

def compute_metrics(rows: list[dict]) -> dict:
    """
    Treat 'challenge' and 'block' as positive (flagged).
    'allow' is negative.
    Ground-truth positive  = anomalous event.
    Ground-truth negative  = normal event.
    """
    tp = fp = fn = tn = 0
    per_type: dict[str, dict] = {}

    for r in rows:
        gt       = r["ground_truth_label"]           # "normal" / "anomalous"
        decision = r["decision"]                      # allow / challenge / block
        atype    = r["anomaly_type"]                  # "" for normal events

        flagged  = decision in ("challenge", "block")
        positive = (gt == "anomalous")

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


# OUTPUT

OUTPUT_FIELDS = [
    "event_id", "user_id", "department", "resource_id", "sensitivity",
    "network_zone", "device_trust", "failure_count_24h",
    "ground_truth_label", "anomaly_type",
    "decision", "rule_triggered",
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV]  {len(rows):,} decisions  →  {path}")


def print_report(m: dict, rows: list[dict]) -> None:
    total    = len(rows)
    n_anom   = sum(1 for r in rows if r["ground_truth_label"] == "anomalous")
    n_normal = total - n_anom

    print("\n" + "═" * 60)
    print("  BASELINE POLICY ENGINE — EVALUATION REPORT")
    print("═" * 60)
    print(f"  Total events   : {total:,}")
    print(f"  Anomalous      : {n_anom:,}  ({100*n_anom/total:.1f}%)")
    print(f"  Normal         : {n_normal:,}  ({100*n_normal/total:.1f}%)")

    print("\n  ── Overall Metrics ─────────────────────────────────")
    print(f"  True  Positives (TP) : {m['tp']:>6,}")
    print(f"  False Positives (FP) : {m['fp']:>6,}")
    print(f"  False Negatives (FN) : {m['fn']:>6,}")
    print(f"  True  Negatives (TN) : {m['tn']:>6,}")
    print(f"  Precision            : {m['precision']:.3f}")
    print(f"  Recall               : {m['recall']:.3f}")
    print(f"  F1-score             : {m['f1']:.3f}")

    print("\n  ── Per-Category Detection ──────────────────────────")
    header = f"  {'Anomaly Type':<35} {'Total':>5} {'Det.':>5} {'Miss.':>5}  Recall"
    print(header)
    print("  " + "-" * 58)
    for atype, s in sorted(m["per_type"].items()):
        r = s["detected"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {atype:<35} {s['total']:>5} {s['detected']:>5} {s['missed']:>5}  {r:.1%}")

    # Rule trigger breakdown
    rule_counts: dict[str, int] = {}
    for r in rows:
        rule_counts[r["rule_triggered"]] = rule_counts.get(r["rule_triggered"], 0) + 1

    print("\n  ── Rule Trigger Counts ─────────────────────────────")
    for rule_id, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule_id:<30} {count:>6,}")

    print("═" * 60 + "\n")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="TechnoMed Solutions — Static Baseline Policy Engine (Task 3)"
    )
    parser.add_argument("--input",  default="logs_with_anomalies.jsonl",
                        help="Input log file (default: logs_with_anomalies.jsonl)")
    parser.add_argument("--output", default="baseline_decisions.csv",
                        help="Output CSV (default: baseline_decisions.csv)")
    args = parser.parse_args()

    print(f"\n  TechnoMed Solutions — Baseline Policy Engine")
    print(f"  Input : {args.input}")

    with open(args.input, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(events):,} events.\n")

    rows = []
    for event in events:
        decision, rule = evaluate(event)
        rows.append({
            **event,
            "decision":      decision,
            "rule_triggered": rule,
        })

    write_csv(rows, args.output)

    m = compute_metrics(rows)
    print_report(m, rows)

    print(f"  Done.  Reproduce with:  python baseline_policy.py\n")


if __name__ == "__main__":
    main()
