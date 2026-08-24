#!/usr/bin/env python3
"""
drift_simulation.py

Simulates a remote-work week drift scenario: all employees are instructed
to work from home for a 4-day period (the same dates as the test partition).
This causes the model's learned association between external network access
and anomalous behaviour to break down, because external access is now the
expected norm rather than the exception.

Outputs:
  drifted_decisions.csv   — scored drifted test events
  Prints full pre/post comparison to stdout.
"""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_score, recall_score)

# CONSTANTS  

DEPT_HOURS = {
    "Engineering": (7, 20), "Finance": (8, 18), "HR": (8, 17),
    "Clinical": (7, 19), "IT Security": (0, 24),
    "Legal": (9, 18), "Executive": (7, 21),
}
AUTHORIZED_DEPTS = {
    "RES-001": {"Clinical","IT Security"}, "RES-002": {"Clinical","IT Security"},
    "RES-003": {"Clinical","IT Security"}, "RES-004": {"HR","IT Security"},
    "RES-005": {"Finance","HR","IT Security"}, "RES-006": {"IT Security"},
    "RES-007": {"IT Security","Legal"}, "RES-008": {"Executive","IT Security"},
    "RES-009": {"Finance","Executive","IT Security"}, "RES-010": {"Engineering","IT Security"},
    "RES-011": {"Engineering","Finance","IT Security"}, "RES-012": {"Engineering","Finance","IT Security"},
    "RES-013": {"IT Security"}, "RES-014": {"IT Security"},
    "RES-015": {"Legal","IT Security"}, "RES-016": {"Legal","IT Security"},
    "RES-017": {"Engineering","IT Security"}, "RES-018": {"Engineering","IT Security"},
    "RES-019": {"Engineering","IT Security"}, "RES-020": {"IT Security"},
    "RES-021": None, "RES-022": None, "RES-023": None,
    "RES-024": {"Finance","Legal","IT Security"},
    "RES-025": None, "RES-026": None, "RES-027": None,
}
SENSITIVITY_LEVEL = {"Public": 0, "Internal": 1, "Confidential": 2, "Privileged": 3}
SENSITIVE_TIERS   = {"Privileged", "Confidential"}
SIM_START  = datetime(2024, 1, 15, tzinfo=timezone.utc)
TRAIN_DAYS = 10

FEATURE_NAMES = [
    "sensitivity_level","is_off_hours","is_external","is_vpn",
    "device_trust_unknown","device_trust_noncompliant","is_first_seen_device",
    "failure_count_24h","mfa_not_used","action_is_export",
    "action_write_or_delete","hour_of_day_norm","is_weekend",
    "sensitivity_mismatch","external_privileged","off_hours_privileged",
    "peer_resource_rarity","user_resource_novelty",
    "user_hour_deviation_norm","bytes_log_zscore",
]

# HELPERS

def parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def is_off_hours(event):
    start, end = DEPT_HOURS.get(event.get("department",""), (8, 18))
    return not (start <= parse_ts(event["timestamp"]).hour < end)

def is_role_mismatch(event):
    allowed = AUTHORIZED_DEPTS.get(event.get("resource_id",""))
    if allowed is None: return False
    return event.get("department","") not in allowed

def build_labels(events):
    return np.array([1 if e.get("ground_truth_label")=="anomalous" else 0
                     for e in events], dtype=np.int32)

def scale_to_risk(proba):
    return np.clip(np.round(proba * 100).astype(int), 0, 100)

def split_events(events):
    cutoff = SIM_START.replace(day=SIM_START.day + TRAIN_DAYS)
    train = [e for e in events if parse_ts(e["timestamp"]) < cutoff]
    test  = [e for e in events if parse_ts(e["timestamp"]) >= cutoff]
    return train, test

# FEATURE TRANSFORMER  

class FeatureTransformer:
    def fit(self, events):
        drc = defaultdict(lambda: defaultdict(int))
        dc  = defaultdict(int)
        for e in events:
            drc[e["department"]][e["resource_id"]] += 1
            dc[e["department"]] += 1
        self._drc, self._dc = drc, dc

        ur = defaultdict(set)
        for e in events:
            ur[e["user_id"]].add(e["resource_id"])
        self._ur = ur

        uh, dh = defaultdict(list), defaultdict(list)
        for e in events:
            h = parse_ts(e["timestamp"]).hour
            uh[e["user_id"]].append(h); dh[e["department"]].append(h)
        self._umh = {u: float(np.mean(v)) for u,v in uh.items()}
        self._dmh = {d: float(np.mean(v)) for d,v in dh.items()}

        ulb = defaultdict(list)
        for e in events:
            ulb[e["user_id"]].append(math.log1p(e.get("bytes_transferred",0)))
        self._umb  = {u: float(np.mean(v)) for u,v in ulb.items()}
        self._ustd = {u: float(np.std(v))+1e-6 for u,v in ulb.items()}
        all_lb = [math.log1p(e.get("bytes_transferred",0)) for e in events]
        self._gmb = float(np.mean(all_lb)); self._gstd = float(np.std(all_lb))+1e-6
        return self

    def transform_one(self, event):
        uid=event["user_id"]; dept=event["department"]; res=event["resource_id"]
        sens=event.get("sensitivity","Public"); ts=parse_ts(event["timestamp"])
        hour=ts.hour; nz=event.get("network_zone","corporate")
        dt=event.get("device_trust","compliant"); action=event.get("action","read")
        bytes_=event.get("bytes_transferred",0)
        d_tot=self._dc.get(dept,1); r_cnt=self._drc.get(dept,{}).get(res,0)
        pr=1.0 if r_cnt==0 else 1.0-(r_cnt/d_tot)
        nov=0.0 if res in self._ur.get(uid,set()) else 1.0
        mh=self._umh.get(uid,self._dmh.get(dept,12.0))
        hd=abs(hour-mh)/12.0
        lb=math.log1p(bytes_)
        bz=max((lb-self._umb.get(uid,self._gmb))/self._ustd.get(uid,self._gstd),-4.0)
        oh=is_off_hours(event); mm=is_role_mismatch(event) and sens in SENSITIVE_TIERS
        nis=dept!="IT Security"
        return [
            float(SENSITIVITY_LEVEL.get(sens,0)), float(oh),
            float(nz=="external"), float(nz=="vpn"),
            float(dt=="unknown"), float(dt=="non_compliant"),
            float(event.get("is_first_seen_device",False)),
            float(min(event.get("failure_count_24h",0),10)),
            float(not event.get("mfa_used",True)),
            float(action=="export"), float(action in ("write","delete")),
            float(hour)/23.0, float(ts.weekday()>=5),
            float(mm), float(nz=="external" and sens=="Privileged"),
            float(oh and sens=="Privileged" and nis),
            pr, nov, hd, bz,
        ]

    def transform(self, events):
        return np.array([self.transform_one(e) for e in events], dtype=np.float32)

# DRIFT SIMULATION

def apply_remote_work_drift(events):
    """
    Simulate a remote-work week: convert corporate-zone normal events to
    external-zone access, representing legitimate employees working from home.
    Anomalous events are left unchanged — attackers don't switch to WFH.
    VPN events are also left unchanged — already remote.
    """
    drifted = []
    n_converted = 0
    for e in events:
        e2 = dict(e)
        if (e.get("ground_truth_label") == "normal" and
                e.get("network_zone") == "corporate"):
            e2["network_zone"] = "external"
            n_converted += 1
        drifted.append(e2)
    print(f"  Drift applied: {n_converted:,} normal corporate events "
          f"→ external (remote-work simulation)")
    return drifted, n_converted

# METRICS

def compute_metrics(events, scores, label=""):
    y_true = build_labels(events)
    y_pred = (scores >= 50).astype(int)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Per-category
    atype_stats = defaultdict(lambda: {"total":0,"detected":0})
    for e, s in zip(events, scores):
        at = e.get("anomaly_type","")
        if not at: continue
        atype_stats[at]["total"] += 1
        if s >= 50: atype_stats[at]["detected"] += 1

    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(pr), recall=float(rec), f1=float(f1),
                per_type=dict(atype_stats))

# REPORT

def print_comparison(pre, post, n_converted, total_normal):
    print("\n" + "═"*66)
    print("  DRIFT SIMULATION — PRE vs POST COMPARISON")
    print("═"*66)
    print(f"\n  Drift scenario : Remote-work week")
    print(f"  Conversion     : {n_converted:,} of {total_normal:,} normal corporate "
          f"events → external ({100*n_converted/total_normal:.1f}%)")

    print(f"\n  {'Metric':<18}  {'Pre-Drift':>10}  {'Post-Drift':>10}  {'Change':>8}")
    print("  " + "-"*52)
    for lbl, pk, dk in [("TP", "tp","tp"),("FP","fp","fp"),
                         ("FN","fn","fn"),("TN","tn","tn")]:
        d = post[dk] - pre[pk]; s = "+" if d>=0 else ""
        print(f"  {lbl:<18}  {pre[pk]:>10,}  {post[dk]:>10,}  {s}{d:>7,}")
    for lbl, pk, dk in [("Precision","precision","precision"),
                          ("Recall","recall","recall"),
                          ("F1","f1","f1")]:
        d = post[dk] - pre[pk]; s = "+" if d>=0 else ""
        print(f"  {lbl:<18}  {pre[pk]:>10.3f}  {post[dk]:>10.3f}  {s}{d:>7.3f}")

    print(f"\n  ── Per-Category Recall ─────────────────────────────────────")
    all_types = sorted(set(list(pre["per_type"].keys()) + list(post["per_type"].keys())))
    print(f"  {'Anomaly Type':<35}  {'Pre':>6}  {'Post':>6}  {'Change':>8}")
    print("  " + "-"*60)
    for at in all_types:
        ps = pre["per_type"].get(at, {"total":0,"detected":0})
        ds = post["per_type"].get(at, {"total":0,"detected":0})
        pr_ = ps["detected"]/ps["total"] if ps["total"]>0 else 0.0
        dr_ = ds["detected"]/ds["total"] if ds["total"]>0 else 0.0
        chg = dr_ - pr_
        s = "+" if chg>=0 else ""
        print(f"  {at:<35}  {pr_:>5.1%}  {dr_:>5.1%}  {s}{chg:>+6.1%}")

    print("═"*66 + "\n")

# MAIN

def main():
    print("\n  TechnoMed Solutions — Drift Simulation (Task 7)\n")

    # Load and split
    with open("logs_with_anomalies.jsonl", encoding="utf-8") as f:
        all_events = [json.loads(l) for l in f if l.strip()]
    all_events.sort(key=lambda e: e["timestamp"])
    train_events, test_events = split_events(all_events)
    print(f"  Train: {len(train_events):,}  |  Test: {len(test_events):,}")

    # Fit transformer and train model on ORIGINAL training data
    transformer = FeatureTransformer().fit(train_events)
    X_train = transformer.transform(train_events)
    y_train = build_labels(train_events)

    print("  Training model on original (pre-drift) data …")
    base_rf = RandomForestClassifier(
        n_estimators=50, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model = CalibratedClassifierCV(base_rf, method="isotonic", cv=2)
    model.fit(X_train, y_train)
    print("  Done.\n")

    # Score original test set (pre-drift)
    X_test_orig  = transformer.transform(test_events)
    probas_orig  = model.predict_proba(X_test_orig)[:, 1]
    scores_orig  = scale_to_risk(probas_orig)
    pre_metrics  = compute_metrics(test_events, scores_orig, "Pre-drift")

    # Apply drift and score
    drifted_events, n_converted = apply_remote_work_drift(test_events)
    X_test_drift = transformer.transform(drifted_events)
    probas_drift = model.predict_proba(X_test_drift)[:, 1]
    scores_drift = scale_to_risk(probas_drift)
    post_metrics = compute_metrics(drifted_events, scores_drift, "Post-drift")

    # Save drifted decisions
    total_normal = sum(1 for e in test_events if e.get("ground_truth_label")=="normal")
    with open("drifted_decisions.csv","w",newline="",encoding="utf-8") as f:
        fieldnames = ["event_id","user_id","department","resource_id","sensitivity",
                      "network_zone","ground_truth_label","anomaly_type",
                      "risk_score_pre","risk_score_post","decision_pre","decision_post"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e, sp, sd in zip(test_events, scores_orig, scores_drift):
            w.writerow({
                "event_id": e.get("event_id"),
                "user_id": e.get("user_id"),
                "department": e.get("department"),
                "resource_id": e.get("resource_id"),
                "sensitivity": e.get("sensitivity"),
                "network_zone": e.get("network_zone"),
                "ground_truth_label": e.get("ground_truth_label"),
                "anomaly_type": e.get("anomaly_type",""),
                "risk_score_pre":  int(sp),
                "risk_score_post": int(sd),
                "decision_pre":  "flagged" if sp>=50 else "allowed",
                "decision_post": "flagged" if sd>=50 else "allowed",
            })
    print(f"  [CSV]  drifted_decisions.csv saved.")

    print_comparison(pre_metrics, post_metrics, n_converted, total_normal)
    print(f"  Done.  Reproduce with:  python drift_simulation.py\n")

if __name__ == "__main__":
    main()
