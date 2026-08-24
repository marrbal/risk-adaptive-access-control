#!/usr/bin/env python3
"""
risk_model.py

Approach
--------
Supervised classification with a calibrated Random Forest that outputs a
probability for each event being anomalous.  The probability is scaled to
a 0–100 risk score.

Train / test split
------------------
All events are sorted by timestamp.  The first 10 simulation days are used
for training; the final 4 days form the held-out test set.  Features that
require historical context (peer-group rarity, user resource novelty, byte
volume statistics) are computed exclusively from training data to prevent
temporal leakage.

Output
------
  risk_scores.csv  — one row per event: event_id, risk_score, decision fields
"""

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

# DOMAIN CONSTANTS 

DEPT_HOURS = {
    "Engineering":  (7,  20),
    "Finance":      (8,  18),
    "HR":           (8,  17),
    "Clinical":     (7,  19),
    "IT Security":  (0,  24),
    "Legal":        (9,  18),
    "Executive":    (7,  21),
}

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
    "RES-021": None,
    "RES-022": None,
    "RES-023": None,
    "RES-024": {"Finance", "Legal", "IT Security"},
    "RES-025": None,
    "RES-026": None,
    "RES-027": None,
}

SENSITIVITY_LEVEL = {"Public": 0, "Internal": 1, "Confidential": 2, "Privileged": 3}
SENSITIVE_TIERS   = {"Privileged", "Confidential"}

SIM_START  = datetime(2024, 1, 15, tzinfo=timezone.utc)
TRAIN_DAYS = 10   # days 1–10 for training; days 11–14 for testing

# HELPERS

def parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def is_off_hours(event: dict) -> bool:
    start, end = DEPT_HOURS.get(event.get("department", ""), (8, 18))
    hour = parse_ts(event["timestamp"]).hour
    return not (start <= hour < end)


def is_role_mismatch(event: dict) -> bool:
    allowed = AUTHORIZED_DEPTS.get(event.get("resource_id", ""))
    if allowed is None:
        return False
    return event.get("department", "") not in allowed


# FEATURE NAMES  (20 features total)

FEATURE_NAMES = [
    # Static sensitivity / network / device signals
    "sensitivity_level",          # 0=Public … 3=Privileged
    "is_off_hours",               # 1 outside dept working window
    "is_external",                # 1 if network_zone = external
    "is_vpn",                     # 1 if network_zone = vpn
    "device_trust_unknown",       # 1 if device_trust = unknown
    "device_trust_noncompliant",  # 1 if device_trust = non_compliant
    "is_first_seen_device",       # direct from log schema
    "failure_count_24h",          # direct from log schema (capped at 10)
    "mfa_not_used",               # 1 if mfa_used = False
    "action_is_export",           # 1 if action = export
    "action_write_or_delete",     # 1 if action in {write, delete}
    "hour_of_day_norm",           # hour / 23  (0–1)
    "is_weekend",                 # 1 if day_of_week >= 5
    # Combined risk signals
    "sensitivity_mismatch",       # 1 if dept not authorised AND Priv/Conf
    "external_privileged",        # 1 if external AND Privileged
    "off_hours_privileged",       # 1 if off-hours AND Privileged (non-IT-Sec)
    # Historical / behavioural features  (trained on train set only)
    "peer_resource_rarity",       # 1 − (dept×resource count / dept count)
    "user_resource_novelty",      # 1 if user never accessed resource in training
    "user_hour_deviation_norm",   # |hour − user mean hour| / 12
    "bytes_log_zscore",           # z-score of log(bytes+1) vs user training mean
]


# FEATURE TRANSFORMER  (fit on train set, transform any event)

class FeatureTransformer:

    def fit(self, events: list[dict]) -> "FeatureTransformer":
        # Peer-group rarity
        dept_res_cnt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        dept_cnt: dict[str, int] = defaultdict(int)
        for e in events:
            dept_res_cnt[e["department"]][e["resource_id"]] += 1
            dept_cnt[e["department"]] += 1
        self._dept_res_cnt = dept_res_cnt
        self._dept_cnt     = dept_cnt

        # User–resource novelty 
        user_res: dict[str, set] = defaultdict(set)
        for e in events:
            user_res[e["user_id"]].add(e["resource_id"])
        self._user_res = user_res

        # User mean access hour 
        user_hours: dict[str, list] = defaultdict(list)
        dept_hours: dict[str, list] = defaultdict(list)
        for e in events:
            h = parse_ts(e["timestamp"]).hour
            user_hours[e["user_id"]].append(h)
            dept_hours[e["department"]].append(h)
        self._user_mean_h = {u: float(np.mean(hs)) for u, hs in user_hours.items()}
        self._dept_mean_h = {d: float(np.mean(hs)) for d, hs in dept_hours.items()}

        # User byte-volume statistics (on log-scale) 
        user_logb: dict[str, list] = defaultdict(list)
        for e in events:
            user_logb[e["user_id"]].append(math.log1p(e.get("bytes_transferred", 0)))
        self._user_mean_b = {u: float(np.mean(bs)) for u, bs in user_logb.items()}
        self._user_std_b  = {u: float(np.std(bs)) + 1e-6 for u, bs in user_logb.items()}
        all_lb = [math.log1p(e.get("bytes_transferred", 0)) for e in events]
        self._global_mean_b = float(np.mean(all_lb))
        self._global_std_b  = float(np.std(all_lb)) + 1e-6

        return self

    def transform_one(self, event: dict) -> list[float]:
        uid    = event["user_id"]
        dept   = event["department"]
        res    = event["resource_id"]
        sens   = event.get("sensitivity", "Public")
        ts     = parse_ts(event["timestamp"])
        hour   = ts.hour
        nz     = event.get("network_zone", "corporate")
        dt     = event.get("device_trust", "compliant")
        action = event.get("action", "read")
        bytes_ = event.get("bytes_transferred", 0)

        # Peer resource rarity
        d_total = self._dept_cnt.get(dept, 1)
        r_count = self._dept_res_cnt.get(dept, {}).get(res, 0)
        peer_rarity = 1.0 if r_count == 0 else 1.0 - (r_count / d_total)

        # User resource novelty
        novelty = 0.0 if res in self._user_res.get(uid, set()) else 1.0

        # User hour deviation (normalised)
        mean_h   = self._user_mean_h.get(uid, self._dept_mean_h.get(dept, 12.0))
        hour_dev = abs(hour - mean_h) / 12.0

        # Byte z-score (log-scale)
        log_b = math.log1p(bytes_)
        mu_b  = self._user_mean_b.get(uid, self._global_mean_b)
        sd_b  = self._user_std_b.get(uid,  self._global_std_b)
        b_z   = max((log_b - mu_b) / sd_b, -4.0)  # keep upper tail open

        off_h     = is_off_hours(event)
        mismatch  = is_role_mismatch(event) and sens in SENSITIVE_TIERS
        non_it_sec = dept != "IT Security"

        return [
            float(SENSITIVITY_LEVEL.get(sens, 0)),
            float(off_h),
            float(nz == "external"),
            float(nz == "vpn"),
            float(dt == "unknown"),
            float(dt == "non_compliant"),
            float(event.get("is_first_seen_device", False)),
            float(min(event.get("failure_count_24h", 0), 10)),
            float(not event.get("mfa_used", True)),
            float(action == "export"),
            float(action in ("write", "delete")),
            float(hour) / 23.0,
            float(ts.weekday() >= 5),
            float(mismatch),
            float(nz == "external" and sens == "Privileged"),
            float(off_h and sens == "Privileged" and non_it_sec),
            peer_rarity,
            novelty,
            hour_dev,
            b_z,
        ]

    def transform(self, events: list[dict]) -> np.ndarray:
        return np.array([self.transform_one(e) for e in events], dtype=np.float32)


# PIPELINE

def split_events(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Strict temporal split: first TRAIN_DAYS → train, remainder → test."""
    cutoff = SIM_START.replace(day=SIM_START.day + TRAIN_DAYS)
    train = [e for e in events if parse_ts(e["timestamp"]) < cutoff]
    test  = [e for e in events if parse_ts(e["timestamp"]) >= cutoff]
    return train, test


def build_labels(events: list[dict]) -> np.ndarray:
    return np.array(
        [1 if e.get("ground_truth_label") == "anomalous" else 0 for e in events],
        dtype=np.int32,
    )


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                seed: int = 42) -> CalibratedClassifierCV:
    base = RandomForestClassifier(
        n_estimators    = 400,
        max_depth       = None,
        min_samples_leaf= 2,
        class_weight    = "balanced",
        random_state    = seed,
        n_jobs          = -1,
    )
    # Isotonic calibration improves probability estimates from RF
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    return model


def scale_to_risk(proba: np.ndarray) -> np.ndarray:
    """Map [0, 1] probability to integer risk score [0, 100]."""
    return np.clip(np.round(proba * 100).astype(int), 0, 100)


# OUTPUT

OUTPUT_FIELDS = [
    "event_id", "user_id", "department", "resource_id", "sensitivity",
    "network_zone", "device_trust", "failure_count_24h", "bytes_transferred",
    "ground_truth_label", "anomaly_type",
    "split", "anomaly_proba", "risk_score",
]


def write_scores(events: list[dict], probas: np.ndarray,
                 splits: list[str], path: str) -> None:
    scores = scale_to_risk(probas)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for e, prob, score, sp in zip(events, probas, scores, splits):
            row = {k: e.get(k, "") for k in OUTPUT_FIELDS}
            row["anomaly_proba"] = f"{prob:.4f}"
            row["risk_score"]    = int(score)
            row["split"]         = sp
            writer.writerow(row)
    print(f"  [CSV]  {len(events):,} risk scores  →  {path}")


# EVALUATION

def evaluate(events, probas, scores, split_name):
    y_true = build_labels(events)
    y_pred = (scores >= 50).astype(int)  

    if y_true.sum() == 0:
        print(f"  [{split_name}]  No positives — skipping metrics.")
        return

    pr  = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, probas)
        pr_auc = average_precision_score(y_true, probas)
    except Exception:
        roc = pr_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    print(f"\n  ── {split_name} Metrics ──────────────────────────────────────")
    print(f"  Events       : {len(events):,}  "
          f"(anomalous: {int(y_true.sum()):,}, normal: {int((1-y_true).sum()):,})")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision    : {pr:.3f}")
    print(f"  Recall       : {rec:.3f}")
    print(f"  F1-score     : {f1:.3f}")
    print(f"  ROC-AUC      : {roc:.3f}")
    print(f"  PR-AUC       : {pr_auc:.3f}")

    # Per-category breakdown
    atype_stats: dict[str, dict] = {}
    for e, s in zip(events, scores):
        at = e.get("anomaly_type", "")
        if not at:
            continue
        if at not in atype_stats:
            atype_stats[at] = {"total": 0, "detected": 0}
        atype_stats[at]["total"] += 1
        if s >= 50:
            atype_stats[at]["detected"] += 1

    if atype_stats:
        print(f"\n  ── Per-Category ({split_name}) ───────────────────────────")
        header = f"  {'Anomaly Type':<35} {'Total':>5} {'Det.':>5}  Recall"
        print(header)
        print("  " + "-" * 55)
        for at, s in sorted(atype_stats.items()):
            r = s["detected"] / s["total"] if s["total"] > 0 else 0.0
            print(f"  {at:<35} {s['total']:>5} {s['detected']:>5}  {r:.1%}")


def feature_importances(model, top_n=10):
    try:
        base = model.calibrated_classifiers_[0].estimator
        imp  = base.feature_importances_
        pairs = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
        print(f"\n  ── Top {top_n} Feature Importances ──────────────────────")
        for name, score in pairs[:top_n]:
            bar = "█" * int(score * 200)
            print(f"  {name:<30}  {score:.4f}  {bar}")
    except Exception:
        pass


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="TechnoMed Solutions — AI Risk-Scoring Model (Task 4)"
    )
    parser.add_argument("--input",  default="logs_with_anomalies.jsonl")
    parser.add_argument("--output", default="risk_scores.csv")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    print(f"\n  TechnoMed Solutions — AI Risk-Scoring Model")
    print(f"  Input : {args.input}  |  Seed : {args.seed}\n")

    # Load and sort 
    with open(args.input, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    events.sort(key=lambda e: e["timestamp"])
    print(f"  Loaded {len(events):,} events.")

    # Temporal split
    train_events, test_events = split_events(events)
    splits = ["train"] * len(train_events) + ["test"] * len(test_events)
    print(f"  Train : {len(train_events):,} events (days 1–{TRAIN_DAYS})")
    print(f"  Test  : {len(test_events):,} events (days {TRAIN_DAYS+1}–14)\n")

    # Fit transformer on training data ONLY 
    transformer = FeatureTransformer().fit(train_events)
    print("  Feature transformer fitted on training data.")

    X_train = transformer.transform(train_events)
    X_test  = transformer.transform(test_events)
    X_all   = np.vstack([X_train, X_test])

    y_train = build_labels(train_events)
    y_test  = build_labels(test_events)

    print(f"  Features : {X_train.shape[1]}  ({', '.join(FEATURE_NAMES[:5])}, …)\n")

    # Train model
    print("  Training calibrated Random Forest …")
    model = train_model(X_train, y_train, seed=args.seed)
    print("  Training complete.\n")

    # Score all events 
    all_events = train_events + test_events
    probas_all = model.predict_proba(X_all)[:, 1]
    scores_all = scale_to_risk(probas_all)

    # Write output 
    write_scores(all_events, probas_all, splits, args.output)

    # Save model and transformer for evaluation.py 
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("transformer.pkl", "wb") as f:
        pickle.dump(transformer, f)
    print("  [PKL]  model.pkl and transformer.pkl saved.")

    # Evaluation 
    n_tr = len(train_events)
    print("\n" + "═" * 60)
    print("  AI RISK-SCORING MODEL — EVALUATION REPORT")
    print("═" * 60)

    evaluate(train_events, probas_all[:n_tr], scores_all[:n_tr], "Train (in-sample)")
    evaluate(test_events,  probas_all[n_tr:], scores_all[n_tr:], "Test  (held-out)")
    feature_importances(model)

    print("\n" + "═" * 60)
    print(f"\n  Done.  Reproduce with:  python risk_model.py --seed {args.seed}\n")


if __name__ == "__main__":
    main()
