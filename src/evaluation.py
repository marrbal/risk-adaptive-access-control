#!/usr/bin/env python3
"""
evaluation.py

Outputs:
  explanations.json   — machine-readable SHAP explanation records
  Prints full report to stdout.
"""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

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

# 6.1  QUANTITATIVE METRICS  (from risk_scores.csv — no retraining)

def section_61():
    print("\n  ══ 6.1  QUANTITATIVE EVALUATION (TEST SET) ══════════════════")

    with open("risk_scores.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"]=="test"]

    y_true  = np.array([1 if r["ground_truth_label"]=="anomalous" else 0 for r in rows])
    probas  = np.array([float(r["anomaly_proba"]) for r in rows])
    scores  = np.array([int(r["risk_score"]) for r in rows])
    y_pred  = (scores >= 50).astype(int)

    pr     = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    roc    = roc_auc_score(y_true, probas)
    pr_auc = average_precision_score(y_true, probas)
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()

    baseline = dict(tp=445, fp=620, fn=166, tn=10942,
                    precision=0.418, recall=0.728, f1=0.531)

    print(f"  Test events : {len(rows):,}  "
          f"(anomalous: {int(y_true.sum())}, normal: {int((1-y_true).sum())})\n")
    print(f"  {'Metric':<18}  {'AI Engine':>10}  {'Baseline':>10}  {'Change':>8}")
    print("  " + "-"*52)
    for lbl, ai, b in [("TP", int(tp), baseline["tp"]),
                        ("FP", int(fp), baseline["fp"]),
                        ("FN", int(fn), baseline["fn"]),
                        ("TN", int(tn), baseline["tn"])]:
        d=ai-b; s="+" if d>=0 else ""
        print(f"  {lbl:<18}  {ai:>10,}  {b:>10,}  {s}{d:>7,}")
    for lbl, ai, b in [("Precision", pr, baseline["precision"]),
                        ("Recall",    rec, baseline["recall"]),
                        ("F1-score",  f1,  baseline["f1"])]:
        d=ai-b; s="+" if d>=0 else ""
        print(f"  {lbl:<18}  {ai:>10.3f}  {b:>10.3f}  {s}{d:>7.3f}")
    print(f"  {'ROC-AUC':<18}  {roc:>10.3f}  {'—':>10}  {'—':>8}")
    print(f"  {'PR-AUC':<18}  {pr_auc:>10.3f}  {'—':>10}  {'—':>8}")

    return dict(rows=rows, y_true=y_true, y_pred=y_pred, probas=probas, scores=scores,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(pr), recall=float(rec), f1=float(f1),
                roc=float(roc), pr_auc=float(pr_auc))

# 6.2  SHAP EVENT-LEVEL EXPLANATIONS

def section_62(m61, log_events, train_events, test_events, X_test):
    print("\n  ══ 6.2  EVENT-LEVEL EXPLANATIONS (SHAP) ════════════════════")

    # Fast model for SHAP (50 trees, same seed/pipeline as risk_model.py)
    print("  Training 50-tree RF for SHAP (fast) …")
    transformer = FeatureTransformer().fit(train_events)
    X_train = transformer.transform(train_events)
    y_train = build_labels(train_events)
    base_rf = RandomForestClassifier(
        n_estimators=50, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    base_rf.fit(X_train, y_train)
    print("  Done. Building SHAP explainer …")

    explainer = shap.TreeExplainer(base_rf)

    y_true = m61["y_true"]; y_pred = m61["y_pred"]; scores = m61["scores"]

    def pick(cond):
        idxs = [i for i in range(len(test_events)) if cond(i)]
        return max(idxs, key=lambda i: scores[i]) if idxs else None

    candidates = {
        "E1 – TP (bulk_data_exfiltration)":  pick(lambda i: y_true[i]==1 and y_pred[i]==1 and
                                                   test_events[i].get("anomaly_type")=="bulk_data_exfiltration"),
        "E2 – TP (impossible_travel)":        pick(lambda i: y_true[i]==1 and y_pred[i]==1 and
                                                   test_events[i].get("anomaly_type")=="impossible_travel"),
        "E3 – TP (first_seen_device)":        pick(lambda i: y_true[i]==1 and y_pred[i]==1 and
                                                   test_events[i].get("anomaly_type")=="first_seen_device_sensitive"),
        "E4 – FP (normal, wrongly flagged)":  pick(lambda i: y_true[i]==0 and y_pred[i]==1),
        "E5 – FN (anomaly, narrowly missed)": pick(lambda i: y_true[i]==1 and y_pred[i]==0),
    }

    explanations = []
    used = set()
    for label, idx in candidates.items():
        if idx is None or idx in used:
            continue
        used.add(idx)
        e     = test_events[idx]
        score = int(scores[idx])
        prob  = float(m61["probas"][idx])
        gt    = "anomalous" if y_true[idx]==1 else "normal"
        pred  = "flagged" if y_pred[idx]==1 else "allowed"
        # Compute SHAP for this one event only (fast)
        sv_raw = explainer.shap_values(X_test[idx:idx+1])
        sv_row = sv_raw[1][0] if isinstance(sv_raw, list) else (
                 sv_raw[0,:,1] if sv_raw.ndim==3 else sv_raw[0])
        top5  = sorted(enumerate(sv_row), key=lambda x: -abs(x[1]))[:5]
        top5f = [{"feature": FEATURE_NAMES[fi],
                  "shap_value": round(float(sv_row[fi]), 4),
                  "feature_val": round(float(X_test[idx, fi]), 4),
                  "direction": "↑ risk" if sv_row[fi]>0 else "↓ risk"}
                 for fi, _ in top5]
        rec = {"event_label": label,
               "event_id": e.get("event_id"), "user_id": e.get("user_id"),
               "department": e.get("department"), "resource_id": e.get("resource_id"),
               "sensitivity": e.get("sensitivity"), "network_zone": e.get("network_zone"),
               "action": e.get("action"), "bytes_transferred": e.get("bytes_transferred"),
               "anomaly_type": e.get("anomaly_type","—"),
               "ground_truth": gt, "prediction": pred,
               "risk_score": score, "anomaly_proba": round(prob,4),
               "top_features": top5f}
        explanations.append(rec)

        print(f"\n  ── {label}")
        print(f"     {e.get('event_id')}  |  {e.get('user_id')}  |  {e.get('department')}  |  "
              f"{e.get('resource_id')} ({e.get('sensitivity')})  |  {e.get('network_zone')}")
        print(f"     Action: {e.get('action')}  |  Bytes: {e.get('bytes_transferred'):,}  |  "
              f"Anomaly: {e.get('anomaly_type','—')}")
        print(f"     Ground truth: {gt}  |  Risk score: {score}  |  Decision: {pred}")
        ev = explainer.expected_value
        base_val = ev[1] if hasattr(ev, '__len__') else float(ev)
        print(f"     Top 5 SHAP contributions (base value: {base_val:.4f}):")
        for f_ in top5f:
            bar = ("▲" if f_["shap_value"]>0 else "▼") * min(int(abs(f_["shap_value"])*40), 20)
            print(f"       {f_['feature']:<30} val={f_['feature_val']:>6.3f}  "
                  f"SHAP={f_['shap_value']:>+7.4f}  {f_['direction']}  {bar}")

    with open("explanations.json","w",encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)
    print(f"\n  [JSON]  Saved → explanations.json")
    return explanations

# 6.3  COST OF ERRORS

def section_63(m61, test_events):
    y_true=m61["y_true"]; y_pred=m61["y_pred"]
    fp_ev=[test_events[i] for i in range(len(test_events)) if y_true[i]==0 and y_pred[i]==1]
    fn_ev=[test_events[i] for i in range(len(test_events)) if y_true[i]==1 and y_pred[i]==0]

    print("\n  ══ 6.3  COST OF ERRORS ══════════════════════════════════════")
    print(f"\n  False Positives: {len(fp_ev)}  |  False Negatives: {len(fn_ev)}")

    fp_by_sens = defaultdict(int)
    for e in fp_ev: fp_by_sens[e.get("sensitivity","?")] += 1
    print("\n  FP breakdown by sensitivity:")
    for s,c in sorted(fp_by_sens.items()): print(f"    {s:<15}: {c}")

    fn_by_type = defaultdict(int)
    for e in fn_ev: fn_by_type[e.get("anomaly_type","?")] += 1
    print("\n  FN breakdown by anomaly type:")
    for at,c in sorted(fn_by_type.items()): print(f"    {at:<35}: {c}")

    return {"fp_events": fp_ev, "fn_events": fn_ev,
            "fp_by_sensitivity": dict(fp_by_sens), "fn_by_type": dict(fn_by_type)}

# 6.4  SUBGROUP PERFORMANCE

def section_64(test_events, m61):
    y_true=m61["y_true"]; y_pred=m61["y_pred"]

    def show_subgroup(title, key_fn, keys):
        print(f"\n  {title}")
        print(f"  {'Group':<25}  {'N':>6}  {'Anom':>5}  {'TP':>4}  {'FP':>4}  "
              f"{'FN':>4}  {'TN':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
        print("  " + "-"*85)
        for k in keys:
            idx=[i for i in range(len(test_events)) if key_fn(test_events[i])==k]
            if not idx: continue
            yt=y_true[idx]; yp=y_pred[idx]
            if yt.sum()==0: continue
            tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum())
            fn=int(((yt==1)&(yp==0)).sum()); tn=int(((yt==0)&(yp==0)).sum())
            pr=tp/(tp+fp) if (tp+fp)>0 else 0.0
            rc=tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1=2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.0
            print(f"  {k:<25}  {len(idx):>6,}  {int(yt.sum()):>5}  {tp:>4}  "
                  f"{fp:>4}  {fn:>4}  {tn:>6,}  {pr:>6.3f}  {rc:>6.3f}  {f1:>6.3f}")

    print("\n  ══ 6.4  SUBGROUP PERFORMANCE (TEST SET) ═════════════════════")
    show_subgroup("By Department:", lambda e: e.get("department",""),
                  sorted(set(e.get("department","") for e in test_events)))
    show_subgroup("By Sensitivity Tier:", lambda e: e.get("sensitivity",""),
                  ["Public","Internal","Confidential","Privileged"])
    show_subgroup("By Network Zone:", lambda e: e.get("network_zone",""),
                  sorted(set(e.get("network_zone","") for e in test_events)))

# MAIN

def main():
    print("\n  TechnoMed Solutions — Task 6 Evaluation & Explainability\n")

    # Load log events (needed for SHAP and subgroup analysis)
    with open("logs_with_anomalies.jsonl", encoding="utf-8") as f:
        all_events = [json.loads(l) for l in f if l.strip()]
    all_events.sort(key=lambda e: e["timestamp"])
    train_events, test_events = split_events(all_events)
    print(f"  Loaded {len(all_events):,} events  "
          f"(train: {len(train_events):,}, test: {len(test_events):,})")

    # Pre-compute test features (needed for SHAP)
    transformer = FeatureTransformer().fit(train_events)
    X_test = transformer.transform(test_events)

    m61 = section_61()
    explanations = section_62(m61, all_events, train_events, test_events, X_test)
    cost_data = section_63(m61, test_events)
    section_64(test_events, m61)

    print("\n  ═══════════════════════════════════════════════════════════")
    print("  Done.  Reproduce with:  python evaluation.py")
    print("  ═══════════════════════════════════════════════════════════\n")

    return {"metrics": m61, "explanations": explanations, "costs": cost_data}

if __name__ == "__main__":
    main()
