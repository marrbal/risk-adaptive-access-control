#!/usr/bin/env python3
"""
anomaly_injector.py
Injects six categories of security-relevant anomalies into the TechnoMed
Solutions synthetic access log produced by data_generator.py.

Anomaly categories
------------------
1. impossible_travel            — access from an external/foreign IP within
                                  30 min of a corporate-zone event for the
                                  same user (requires sequence context)
2. off_hours_privileged         — non-IT user accesses a Privileged resource
                                  outside their normal working hours
3. first_seen_device_sensitive  — unknown/unregistered device used to access
                                  a Confidential or Privileged resource
4. unusual_resource_for_role    — user accesses a resource completely outside
                                  their department's normal scope
5. bulk_data_exfiltration       — abnormally high bytes_transferred across a
                                  cluster of export/read events in one session
6. privilege_escalation         — repeated failures on a Privileged resource
                                  followed by a successful access

Target anomaly rate: ~5 % of total events.

"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

# CONFIGURATION

ANOMALY_COUNTS = {
    "impossible_travel":            100,
    "off_hours_privileged":         150,
    "first_seen_device_sensitive":  100,
    "unusual_resource_for_role":    130,
    "bulk_data_exfiltration":        80,
    "privilege_escalation":          50,
}

# CROSS-DEPARTMENT RESOURCE MAPPINGS
# Resources that are not-normal for each department.
# Used by inject_unusual_resource_for_role.

CROSS_DEPT_RESOURCES = {
    "Engineering": [
        ("RES-004", "HR Personnel Database",    "Privileged",   ["read"]),
        ("RES-001", "Patient Records System",   "Privileged",   ["read"]),
        ("RES-005", "Payroll System",           "Privileged",   ["read"]),
    ],
    "Finance": [
        ("RES-001", "Patient Records System",   "Privileged",   ["read"]),
        ("RES-017", "Source Code Repository",   "Internal",     ["read"]),
        ("RES-002", "Clinical Trial Database",  "Privileged",   ["read"]),
    ],
    "HR": [
        ("RES-010", "AWS Management Console",   "Confidential", ["read"]),
        ("RES-017", "Source Code Repository",   "Internal",     ["read"]),
        ("RES-001", "Patient Records System",   "Privileged",   ["read"]),
    ],
    "Clinical": [
        ("RES-005", "Payroll System",           "Privileged",   ["read"]),
        ("RES-009", "Financial Reports Portal", "Confidential", ["read"]),
        ("RES-017", "Source Code Repository",   "Internal",     ["read"]),
    ],
    "Legal": [
        ("RES-001", "Patient Records System",   "Privileged",   ["read"]),
        ("RES-010", "AWS Management Console",   "Confidential", ["read"]),
        ("RES-005", "Payroll System",           "Privileged",   ["read"]),
    ],
    "Executive": [
        ("RES-017", "Source Code Repository",   "Internal",     ["write"]),
        ("RES-001", "Patient Records System",   "Privileged",   ["write"]),
        ("RES-005", "Payroll System",           "Privileged",   ["write"]),
    ],
}

# Normal working-hour windows per department 
DEPT_HOURS = {
    "Engineering":  (7,  20),
    "Finance":      (8,  18),
    "HR":           (8,  17),
    "Clinical":     (7,  19),
    "IT Security":  (0,  24),
    "Legal":        (9,  18),
    "Executive":    (7,  21),
}


# HELPERS

def parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def foreign_ip(rng) -> str:
    """Synthetic IP suggesting a non-corporate, geographically distant origin."""
    prefixes = [203, 185, 197, 45, 91, 103]
    p = int(rng.choice(prefixes))
    return f"{p}.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"


def mark(event: dict, anomaly_type: str) -> None:
    event["ground_truth_label"] = "anomalous"
    event["anomaly_type"]       = anomaly_type


# INJECTORS

def inject_impossible_travel(events, rng, target, used):
    user_idx = defaultdict(list)
    for i, e in enumerate(events):
        if e["department"] != "IT Security" and i not in used:
            user_idx[e["user_id"]].append(i)

    candidates = []
    for uid, idxs in user_idx.items():
        idxs_sorted = sorted(idxs, key=lambda i: events[i]["timestamp"])
        for j in range(len(idxs_sorted) - 1):
            i1, i2 = idxs_sorted[j], idxs_sorted[j + 1]
            if i2 in used:
                continue
            t1 = parse_ts(events[i1]["timestamp"])
            t2 = parse_ts(events[i2]["timestamp"])
            delta = (t2 - t1).total_seconds()
            if 60 <= delta <= 1800 and events[i1]["network_zone"] in ("corporate", "vpn"):
                candidates.append(i2)

    n = min(target, len(candidates))
    if n == 0:
        return 0
    selected = rng.choice(candidates, size=n, replace=False)
    for idx in selected:
        events[idx]["network_zone"] = "external"
        events[idx]["location"]     = "External"
        events[idx]["ip_address"]   = foreign_ip(rng)
        mark(events[idx], "impossible_travel")
        used.add(int(idx))
    return n


def inject_off_hours_privileged(events, rng, target, used):
    candidates = [
        i for i, e in enumerate(events)
        if e["sensitivity"] == "Privileged"
        and e["department"] != "IT Security"
        and i not in used
    ]
    n = min(target, len(candidates))
    if n == 0:
        return 0
    selected = rng.choice(candidates, size=n, replace=False)
    for idx in selected:
        orig_ts    = parse_ts(events[idx]["timestamp"])
        night_hour = float(rng.uniform(1.0, 4.5))
        new_ts     = orig_ts.replace(
            hour   = int(night_hour),
            minute = int((night_hour % 1) * 60),
            second = int(rng.integers(0, 60)),
        )
        events[idx]["timestamp"] = fmt_ts(new_ts)
        mark(events[idx], "off_hours_privileged")
        used.add(int(idx))
    return n


def inject_first_seen_device(events, rng, target, used):
    candidates = [
        i for i, e in enumerate(events)
        if e["sensitivity"] in ("Privileged", "Confidential")
        and i not in used
    ]
    n = min(target, len(candidates))
    if n == 0:
        return 0
    selected = rng.choice(candidates, size=n, replace=False)
    os_options = ["Windows 10", "Android 13", "iOS 17", "ChromeOS"]
    for counter, idx in enumerate(selected):
        events[idx]["device_id"]            = f"DEV-UNKNOWN-{counter + 1:04d}"
        events[idx]["device_os"]            = str(rng.choice(os_options))
        events[idx]["device_trust"]         = "unknown"
        events[idx]["is_first_seen_device"] = True
        mark(events[idx], "first_seen_device_sensitive")
        used.add(int(idx))
    return n


def inject_unusual_resource(events, rng, target, used):
    candidates = [
        i for i, e in enumerate(events)
        if e["department"] in CROSS_DEPT_RESOURCES
        and e["department"] != "IT Security"
        and i not in used
    ]
    n = min(target, len(candidates))
    if n == 0:
        return 0
    selected = rng.choice(candidates, size=n, replace=False)
    for idx in selected:
        dept    = events[idx]["department"]
        options = CROSS_DEPT_RESOURCES[dept]
        pick    = options[int(rng.integers(0, len(options)))]
        res_id, res_name, sensitivity, actions = pick
        events[idx]["resource_id"]   = res_id
        events[idx]["resource_name"] = res_name
        events[idx]["sensitivity"]   = sensitivity
        events[idx]["action"]        = str(rng.choice(actions))
        mark(events[idx], "unusual_resource_for_role")
        used.add(int(idx))
    return n


def inject_bulk_exfiltration(events, rng, target, used):
    session_map = defaultdict(list)
    for i, e in enumerate(events):
        if i not in used and e["sensitivity"] in ("Confidential", "Privileged"):
            session_map[(e["user_id"], e["session_id"])].append(i)

    rich_sessions = [(k, v) for k, v in session_map.items() if len(v) >= 5]
    perm = rng.permutation(len(rich_sessions))
    rich_sessions = [rich_sessions[i] for i in perm]

    total = 0
    for (uid, sid), idxs in rich_sessions:
        if total >= target:
            break
        cluster_size = min(8, len(idxs), target - total)
        cluster      = [i for i in idxs[:cluster_size] if i not in used]
        for idx in cluster:
            orig = events[idx]["bytes_transferred"]
            mult = int(rng.integers(100, 501))
            events[idx]["bytes_transferred"] = max(orig * mult, 10_000_000)
            events[idx]["duration_seconds"]  = max(2, events[idx]["duration_seconds"] // 10)
            if events[idx]["action"] not in ("connect", "execute", "delete"):
                events[idx]["action"] = "export"
            mark(events[idx], "bulk_data_exfiltration")
            used.add(idx)
            total += 1
    return total


def inject_privilege_escalation(events, rng, target, used):
    PRIV_RESOURCES = [
        ("RES-006", "Identity Management System", "Privileged"),
        ("RES-007", "Audit Log Archive",          "Privileged"),
        ("RES-004", "HR Personnel Database",      "Privileged"),
        ("RES-005", "Payroll System",             "Privileged"),
    ]

    session_map = defaultdict(list)
    for i, e in enumerate(events):
        if e["department"] != "IT Security" and i not in used:
            session_map[(e["user_id"], e["session_id"])].append(i)

    rich_sessions = [(k, v) for k, v in session_map.items() if len(v) >= 3]
    perm = rng.permutation(len(rich_sessions))
    rich_sessions = [rich_sessions[i] for i in perm]

    total = 0
    for (uid, sid), idxs in rich_sessions:
        if total >= target:
            break
        group = [i for i in idxs[:5] if i not in used][:3]
        if len(group) < 3:
            continue
        res_id, res_name, sensitivity = PRIV_RESOURCES[
            int(rng.integers(0, len(PRIV_RESOURCES)))
        ]
        for k, idx in enumerate(group):
            events[idx]["resource_id"]   = res_id
            events[idx]["resource_name"] = res_name
            events[idx]["sensitivity"]   = sensitivity
            events[idx]["action"]        = "read"
            events[idx]["mfa_used"]      = False
            if k < len(group) - 1:
                events[idx]["outcome"]           = "failure"
                events[idx]["failure_count_24h"] = k
            else:
                events[idx]["outcome"]           = "success"
                events[idx]["failure_count_24h"] = len(group) - 1
            mark(events[idx], "privilege_escalation")
            used.add(idx)
            total += 1
    return total

# OUTPUT

def write_jsonl(events, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, default=str) + "\n")
    print(f"  [JSONL] {len(events):,} events  →  {path}")


def write_csv(events, path):
    if not events:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)
    print(f"  [CSV]   {len(events):,} events  →  {path}")


def main():
    parser = argparse.ArgumentParser(
        description="TechnoMed Solutions — Anomaly injector (Task 2)"
    )
    parser.add_argument("--input",  default="logs.jsonl",
                        help="Input JSONL log from data_generator.py (default: logs.jsonl)")
    parser.add_argument("--output", default="logs_with_anomalies",
                        help="Output filename stem (default: logs_with_anomalies)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--format", default="both",
                        choices=["jsonl", "csv", "both"])
    args = parser.parse_args()

    print(f"\n  TechnoMed Solutions — Anomaly Injector")
    print(f"  Input : {args.input}  |  Seed : {args.seed}")

    with open(args.input, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(events):,} events.\n")

    rng  = np.random.default_rng(args.seed)
    used = set()

    injectors = [
        ("impossible_travel",           inject_impossible_travel),
        ("off_hours_privileged",        inject_off_hours_privileged),
        ("first_seen_device_sensitive", inject_first_seen_device),
        ("unusual_resource_for_role",   inject_unusual_resource),
        ("bulk_data_exfiltration",      inject_bulk_exfiltration),
        ("privilege_escalation",        inject_privilege_escalation),
    ]

    results = {}
    for name, fn in injectors:
        count = fn(events, rng, ANOMALY_COUNTS[name], used)
        results[name] = count
        print(f"  [{name:<35}]  {count:>3} events injected")

    total_anomalous = sum(results.values())
    total_events    = len(events)
    pct             = 100 * total_anomalous / total_events

    print(f"\n{'─'*52}")
    print(f"  Total anomalous : {total_anomalous:,}  ({pct:.1f}%)")
    print(f"  Total normal    : {total_events - total_anomalous:,}")
    print(f"  Total events    : {total_events:,}")
    print(f"{'─'*52}\n")

    if args.format in ("jsonl", "both"):
        write_jsonl(events, f"{args.output}.jsonl")
    if args.format in ("csv", "both"):
        write_csv(events, f"{args.output}.csv")

    print(f"\n  Done. Reproduce with:  python anomaly_injector.py --seed {args.seed}\n")


if __name__ == "__main__":
    main()
