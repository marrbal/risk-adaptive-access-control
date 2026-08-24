#!/usr/bin/env python3
"""
data_generator.py
Generates 14 days of realistic enterprise access events (≥12,000 events).
All data is entirely synthetic. No real individuals or organisations are represented.

Usage
-----
    python data_generator.py                      # outputs logs.jsonl + logs.csv + schema.json
    python data_generator.py --seed 42            # fixed seed for reproducibility (default)
    python data_generator.py --output myfile      # custom output filename stem
    python data_generator.py --format jsonl       # JSONL only
    python data_generator.py --format csv         # CSV only
    python data_generator.py --format both        # both (default)

Requirements
------------
    pip install numpy
"""

import argparse
import csv
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np


SIM_START = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)   # Day 1 start
SIM_DAYS  = 14                                                       # 14-day window


# USER REGISTRY  (52 users, 6 departments, 9 distinct roles)
# No real names — all users identified by anonymous ID only.

USERS = [
    # Engineering (14)
    {"user_id": "ENG-001", "role": "Software Engineer",    "department": "Engineering", "seniority": "Senior",    "devices": 2},
    {"user_id": "ENG-002", "role": "Software Engineer",    "department": "Engineering", "seniority": "Mid",       "devices": 1},
    {"user_id": "ENG-003", "role": "Software Engineer",    "department": "Engineering", "seniority": "Mid",       "devices": 2},
    {"user_id": "ENG-004", "role": "Software Engineer",    "department": "Engineering", "seniority": "Junior",    "devices": 1},
    {"user_id": "ENG-005", "role": "Software Engineer",    "department": "Engineering", "seniority": "Junior",    "devices": 1},
    {"user_id": "ENG-006", "role": "Senior Engineer",      "department": "Engineering", "seniority": "Senior",    "devices": 2},
    {"user_id": "ENG-007", "role": "Senior Engineer",      "department": "Engineering", "seniority": "Senior",    "devices": 2},
    {"user_id": "ENG-008", "role": "Senior Engineer",      "department": "Engineering", "seniority": "Senior",    "devices": 2},
    {"user_id": "ENG-009", "role": "Solutions Architect",  "department": "Engineering", "seniority": "Lead",      "devices": 2},
    {"user_id": "ENG-010", "role": "Solutions Architect",  "department": "Engineering", "seniority": "Lead",      "devices": 2},
    {"user_id": "ENG-011", "role": "DevOps Engineer",      "department": "Engineering", "seniority": "Senior",    "devices": 2},
    {"user_id": "ENG-012", "role": "DevOps Engineer",      "department": "Engineering", "seniority": "Mid",       "devices": 2},
    {"user_id": "ENG-013", "role": "QA Engineer",          "department": "Engineering", "seniority": "Mid",       "devices": 1},
    {"user_id": "ENG-014", "role": "QA Engineer",          "department": "Engineering", "seniority": "Junior",    "devices": 1},
    # Finance (8) 
    {"user_id": "FIN-001", "role": "Finance Analyst",           "department": "Finance", "seniority": "Mid",       "devices": 1},
    {"user_id": "FIN-002", "role": "Finance Analyst",           "department": "Finance", "seniority": "Junior",    "devices": 1},
    {"user_id": "FIN-003", "role": "Senior Finance Analyst",    "department": "Finance", "seniority": "Senior",    "devices": 1},
    {"user_id": "FIN-004", "role": "Senior Finance Analyst",    "department": "Finance", "seniority": "Senior",    "devices": 2},
    {"user_id": "FIN-005", "role": "Finance Manager",           "department": "Finance", "seniority": "Lead",      "devices": 2},
    {"user_id": "FIN-006", "role": "Payroll Specialist",        "department": "Finance", "seniority": "Mid",       "devices": 1},
    {"user_id": "FIN-007", "role": "Payroll Specialist",        "department": "Finance", "seniority": "Junior",    "devices": 1},
    {"user_id": "FIN-008", "role": "CFO",                       "department": "Finance", "seniority": "Executive", "devices": 2},
    # HR (7) 
    {"user_id": "HR-001",  "role": "HR Specialist",             "department": "HR", "seniority": "Mid",       "devices": 1},
    {"user_id": "HR-002",  "role": "HR Specialist",             "department": "HR", "seniority": "Junior",    "devices": 1},
    {"user_id": "HR-003",  "role": "HR Specialist",             "department": "HR", "seniority": "Mid",       "devices": 1},
    {"user_id": "HR-004",  "role": "Senior HR Specialist",      "department": "HR", "seniority": "Senior",    "devices": 1},
    {"user_id": "HR-005",  "role": "HR Manager",                "department": "HR", "seniority": "Lead",      "devices": 2},
    {"user_id": "HR-006",  "role": "Recruiter",                 "department": "HR", "seniority": "Mid",       "devices": 1},
    {"user_id": "HR-007",  "role": "CHRO",                      "department": "HR", "seniority": "Executive", "devices": 2},
    # Clinical Affairs (8)
    {"user_id": "CA-001",  "role": "Clinical Specialist",       "department": "Clinical", "seniority": "Senior",    "devices": 2},
    {"user_id": "CA-002",  "role": "Clinical Specialist",       "department": "Clinical", "seniority": "Mid",       "devices": 1},
    {"user_id": "CA-003",  "role": "Clinical Specialist",       "department": "Clinical", "seniority": "Junior",    "devices": 1},
    {"user_id": "CA-004",  "role": "Medical Reviewer",          "department": "Clinical", "seniority": "Senior",    "devices": 2},
    {"user_id": "CA-005",  "role": "Medical Reviewer",          "department": "Clinical", "seniority": "Senior",    "devices": 2},
    {"user_id": "CA-006",  "role": "Clinical Data Analyst",     "department": "Clinical", "seniority": "Mid",       "devices": 1},
    {"user_id": "CA-007",  "role": "VP Clinical Affairs",       "department": "Clinical", "seniority": "Lead",      "devices": 2},
    {"user_id": "CA-008",  "role": "Chief Medical Officer",     "department": "Clinical", "seniority": "Executive", "devices": 2},
    # IT Security (8) 
    {"user_id": "IT-001",  "role": "IT Admin",                  "department": "IT Security", "seniority": "Senior",    "devices": 2},
    {"user_id": "IT-002",  "role": "IT Admin",                  "department": "IT Security", "seniority": "Mid",       "devices": 2},
    {"user_id": "IT-003",  "role": "Security Analyst",          "department": "IT Security", "seniority": "Senior",    "devices": 2},
    {"user_id": "IT-004",  "role": "Security Analyst",          "department": "IT Security", "seniority": "Mid",       "devices": 2},
    {"user_id": "IT-005",  "role": "SOC Analyst",               "department": "IT Security", "seniority": "Senior",    "devices": 2},
    {"user_id": "IT-006",  "role": "SOC Analyst",               "department": "IT Security", "seniority": "Mid",       "devices": 2},
    {"user_id": "IT-007",  "role": "IT Security Manager",       "department": "IT Security", "seniority": "Lead",      "devices": 2},
    {"user_id": "IT-008",  "role": "CISO",                      "department": "IT Security", "seniority": "Executive", "devices": 2},
    # Legal & Executive (7)
    {"user_id": "LEG-001", "role": "Compliance Officer",        "department": "Legal", "seniority": "Senior",    "devices": 1},
    {"user_id": "LEG-002", "role": "Compliance Officer",        "department": "Legal", "seniority": "Mid",       "devices": 1},
    {"user_id": "LEG-003", "role": "Legal Counsel",             "department": "Legal", "seniority": "Senior",    "devices": 2},
    {"user_id": "LEG-004", "role": "Legal Counsel",             "department": "Legal", "seniority": "Mid",       "devices": 1},
    {"user_id": "LEG-005", "role": "Chief Legal Officer",       "department": "Legal", "seniority": "Executive", "devices": 2},
    {"user_id": "EX-001",  "role": "CEO",                       "department": "Executive", "seniority": "Executive", "devices": 2},
    {"user_id": "EX-002",  "role": "COO",                       "department": "Executive", "seniority": "Executive", "devices": 2},
]

# RESOURCE REGISTRY  (27 resources, 4 sensitivity tiers)

RESOURCES = [
    # Privileged 
    {"resource_id": "RES-001", "resource_name": "Patient Records System",      "sensitivity": "Privileged",   "actions": ["read", "write"]},
    {"resource_id": "RES-002", "resource_name": "Clinical Trial Database",     "sensitivity": "Privileged",   "actions": ["read", "write"]},
    {"resource_id": "RES-003", "resource_name": "Drug Safety Repository",      "sensitivity": "Privileged",   "actions": ["read"]},
    {"resource_id": "RES-004", "resource_name": "HR Personnel Database",       "sensitivity": "Privileged",   "actions": ["read", "write"]},
    {"resource_id": "RES-005", "resource_name": "Payroll System",              "sensitivity": "Privileged",   "actions": ["read", "write"]},
    {"resource_id": "RES-006", "resource_name": "Identity Management System",  "sensitivity": "Privileged",   "actions": ["read", "write"]},
    {"resource_id": "RES-007", "resource_name": "Audit Log Archive",           "sensitivity": "Privileged",   "actions": ["read"]},
    {"resource_id": "RES-008", "resource_name": "Executive Dashboard",         "sensitivity": "Privileged",   "actions": ["read"]},
    # Confidential
    {"resource_id": "RES-009", "resource_name": "Financial Reports Portal",    "sensitivity": "Confidential", "actions": ["read", "export"]},
    {"resource_id": "RES-010", "resource_name": "AWS Management Console",      "sensitivity": "Confidential", "actions": ["read", "write"]},
    {"resource_id": "RES-011", "resource_name": "Data Warehouse",              "sensitivity": "Confidential", "actions": ["read", "export"]},
    {"resource_id": "RES-012", "resource_name": "Customer Database",           "sensitivity": "Confidential", "actions": ["read", "write"]},
    {"resource_id": "RES-013", "resource_name": "Security Alerts System",      "sensitivity": "Confidential", "actions": ["read", "write"]},
    {"resource_id": "RES-014", "resource_name": "Network Monitoring Dashboard","sensitivity": "Confidential", "actions": ["read"]},
    {"resource_id": "RES-015", "resource_name": "Legal Document Repository",   "sensitivity": "Confidential", "actions": ["read", "write"]},
    {"resource_id": "RES-016", "resource_name": "Compliance Reports",          "sensitivity": "Confidential", "actions": ["read", "write"]},
    # Internal 
    {"resource_id": "RES-017", "resource_name": "Source Code Repository",      "sensitivity": "Internal",     "actions": ["read", "write", "delete"]},
    {"resource_id": "RES-018", "resource_name": "CI/CD Pipeline",              "sensitivity": "Internal",     "actions": ["read", "execute"]},
    {"resource_id": "RES-019", "resource_name": "API Gateway Console",         "sensitivity": "Internal",     "actions": ["read", "write"]},
    {"resource_id": "RES-020", "resource_name": "Backup Storage",              "sensitivity": "Internal",     "actions": ["read", "write"]},
    {"resource_id": "RES-021", "resource_name": "VPN Gateway",                 "sensitivity": "Internal",     "actions": ["connect"]},
    {"resource_id": "RES-022", "resource_name": "JIRA Project Tracker",        "sensitivity": "Internal",     "actions": ["read", "write"]},
    {"resource_id": "RES-023", "resource_name": "Confluence Wiki",             "sensitivity": "Internal",     "actions": ["read", "write"]},
    {"resource_id": "RES-024", "resource_name": "Vendor Portal",               "sensitivity": "Internal",     "actions": ["read", "write"]},
    # Public
    {"resource_id": "RES-025", "resource_name": "SharePoint Intranet",         "sensitivity": "Public",       "actions": ["read"]},
    {"resource_id": "RES-026", "resource_name": "Email (Exchange)",            "sensitivity": "Public",       "actions": ["read", "write"]},
    {"resource_id": "RES-027", "resource_name": "Marketing Portal",            "sensitivity": "Public",       "actions": ["read"]},
]

RESOURCE_MAP = {r["resource_id"]: r for r in RESOURCES}

# DEVICE OS POOLS  (per department)

OS_POOL = {
    "Engineering":  ["Windows 11", "macOS 14", "Ubuntu 22.04"],
    "Finance":      ["Windows 11", "Windows 10"],
    "HR":           ["Windows 11", "Windows 10"],
    "Clinical":     ["Windows 11", "iPadOS 17"],
    "IT Security":  ["Windows 11", "macOS 14", "Ubuntu 22.04", "Kali Linux"],
    "Legal":        ["Windows 11", "macOS 14"],
    "Executive":    ["macOS 14",   "iPadOS 17", "Windows 11"],
}


# DEPARTMENT BEHAVIOURAL PROFILES
# resource_weights : list of (resource_id, relative_weight)
# work_hours       : (start_hour, end_hour) in UTC (proxy for local time)
# weekend_prob     : probability user is active on a given weekend day
# mean_events_*    : Poisson mean for events per day
# vpn_prob         : probability network_zone == "vpn"
# external_prob    : probability network_zone == "external"
# mfa_prob         : probability MFA is used/required for an event

DEPT_PROFILES = {
    "Engineering": {
        "work_hours":           (7, 20),
        "mean_events_weekday":  24,
        "mean_events_weekend":  3,
        "weekend_prob":         0.30,
        "vpn_prob":             0.30,
        "external_prob":        0.05,
        "mfa_prob":             0.40,
        "resource_weights": [
            ("RES-017", 20), ("RES-018", 12), ("RES-022", 15), ("RES-023", 10),
            ("RES-010",  8), ("RES-019",  6), ("RES-026", 12), ("RES-025",  5),
            ("RES-011",  4), ("RES-012",  3), ("RES-021",  5),
        ],
    },
    "Finance": {
        "work_hours":           (8, 18),
        "mean_events_weekday":  20,
        "mean_events_weekend":  2,
        "weekend_prob":         0.10,
        "vpn_prob":             0.15,
        "external_prob":        0.02,
        "mfa_prob":             0.50,
        "resource_weights": [
            ("RES-009", 18), ("RES-011", 14), ("RES-022", 10), ("RES-026", 15),
            ("RES-024",  8), ("RES-025",  8), ("RES-023",  7), ("RES-012",  6),
            ("RES-021",  4),
        ],
    },
    "HR": {
        "work_hours":           (8, 17),
        "mean_events_weekday":  18,
        "mean_events_weekend":  1,
        "weekend_prob":         0.05,
        "vpn_prob":             0.10,
        "external_prob":        0.01,
        "mfa_prob":             0.60,
        "resource_weights": [
            ("RES-004", 22), ("RES-026", 20), ("RES-023", 15), ("RES-025", 12),
            ("RES-022", 10), ("RES-005",  5), ("RES-021",  4), ("RES-027",  3),
        ],
    },
    "Clinical": {
        "work_hours":           (7, 19),
        "mean_events_weekday":  22,
        "mean_events_weekend":  3,
        "weekend_prob":         0.20,
        "vpn_prob":             0.25,
        "external_prob":        0.03,
        "mfa_prob":             0.70,
        "resource_weights": [
            ("RES-001", 20), ("RES-002", 15), ("RES-003", 10), ("RES-022", 12),
            ("RES-023", 10), ("RES-026", 12), ("RES-025",  6), ("RES-021",  5),
        ],
    },
    "IT Security": {
        "work_hours":           (0, 24),   # 24/7 SOC operation
        "mean_events_weekday":  30,
        "mean_events_weekend":  25,
        "weekend_prob":         1.00,
        "vpn_prob":             0.35,
        "external_prob":        0.10,
        "mfa_prob":             0.85,
        "resource_weights": [
            ("RES-013", 18), ("RES-014", 15), ("RES-007", 12), ("RES-006", 10),
            ("RES-020",  8), ("RES-021",  8), ("RES-026",  8), ("RES-023",  5),
            ("RES-017",  4), ("RES-010",  4), ("RES-001",  2), ("RES-004",  2),
        ],
    },
    "Legal": {
        "work_hours":           (9, 18),
        "mean_events_weekday":  16,
        "mean_events_weekend":  2,
        "weekend_prob":         0.10,
        "vpn_prob":             0.20,
        "external_prob":        0.02,
        "mfa_prob":             0.55,
        "resource_weights": [
            ("RES-015", 22), ("RES-016", 18), ("RES-023", 14), ("RES-026", 16),
            ("RES-025", 10), ("RES-024",  8), ("RES-022",  6), ("RES-021",  3),
        ],
    },
    "Executive": {
        "work_hours":           (7, 21),
        "mean_events_weekday":  12,
        "mean_events_weekend":  4,
        "weekend_prob":         0.40,
        "vpn_prob":             0.40,
        "external_prob":        0.15,
        "mfa_prob":             0.75,
        "resource_weights": [
            ("RES-008", 20), ("RES-026", 25), ("RES-025", 15), ("RES-023", 10),
            ("RES-009",  8), ("RES-015",  5), ("RES-021",  8), ("RES-016",  4),
        ],
    },
}


# HELPER FUNCTIONS

def sample_work_hour(rng, work_start, work_end):
    """Sample a fractional hour from a Gaussian centred on the morning peak."""
    span = work_end - work_start
    if span <= 3:
        return float(rng.uniform(work_start, work_end))

    mu    = work_start + 2.0          # morning peak
    sigma = max(0.8, span / 7)        # spread grows with window length

    if float(rng.random()) < 0.10:
        h = rng.uniform(work_start, work_end)
    else:
        h = rng.normal(mu, sigma)
    return float(np.clip(h, work_start, work_end - 0.017))   # 0.017 h ≈ 1 minute


def sample_timestamp(rng, day_offset, work_start, work_end, continuous_24h=False):
    """Return a UTC datetime for the event."""
    base = SIM_START + timedelta(days=day_offset)
    if continuous_24h:
        frac_hour = float(rng.uniform(0, 24))
    else:
        frac_hour = sample_work_hour(rng, work_start, work_end)
    total_seconds = int(frac_hour * 3600)
    jitter = int(rng.integers(-30, 31))          # ±30 s jitter
    return base + timedelta(seconds=total_seconds + jitter)


def sample_network_zone(rng, profile, is_weekend):
    """Return network_zone string, with VPN more likely at weekends."""
    vpn_boost = 0.12 if is_weekend else 0.0
    ext  = profile["external_prob"]
    vpn  = min(0.90, profile["vpn_prob"] + vpn_boost)
    corp = max(0.0, 1.0 - vpn - ext)
    return str(rng.choice(["corporate", "vpn", "external"], p=[corp, vpn, ext]))


def sample_ip(rng, zone):
    """Generate a plausible synthetic IP based on network zone."""
    if zone == "corporate":
        return f"10.{rng.integers(0, 5)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
    elif zone == "vpn":
        return f"172.16.{rng.integers(0, 16)}.{rng.integers(1, 255)}"
    else:
        return (f"{rng.integers(1, 224)}.{rng.integers(0, 256)}"
                f".{rng.integers(0, 256)}.{rng.integers(1, 255)}")


def sample_bytes(rng, action, sensitivity):
    """Sample bytes_transferred based on action and resource sensitivity."""
    if action in ("connect", "delete"):
        return 0
    mu_map = {
        ("read",    "Privileged"):   10.0,
        ("read",    "Confidential"): 9.0,
        ("read",    "Internal"):     8.5,
        ("read",    "Public"):       8.0,
        ("export",  "Privileged"):   11.0,
        ("export",  "Confidential"): 10.5,
        ("write",   "Privileged"):   9.0,
        ("write",   "Confidential"): 8.5,
        ("write",   "Internal"):     8.0,
        ("execute", "Internal"):     7.5,
    }
    mu = mu_map.get((action, sensitivity), 8.0)
    return max(0, int(rng.lognormal(mean=mu, sigma=1.3)))


def sample_duration(rng, action, sensitivity):
    """Sample duration_seconds (log-normal)."""
    if action == "connect":
        return max(30, int(rng.lognormal(mean=7.5, sigma=1.0)))   # VPN session
    if sensitivity in ("Privileged", "Confidential"):
        return max(5, int(rng.lognormal(mean=5.5, sigma=1.2)))
    return max(5, int(rng.lognormal(mean=4.5, sigma=1.2)))


def sample_outcome(rng, mfa_prob):
    """Success ~95%, failure ~3%, challenge ~ f(mfa_prob)."""
    r = float(rng.random())
    if r < 0.03:
        return "failure"
    if r < 0.03 + mfa_prob * 0.04:
        return "challenge"
    return "success"


def build_device_pool(rng, user):
    """Build the list of registered devices for a user."""
    dept      = user["department"]
    os_list   = OS_POOL.get(dept, ["Windows 11"])
    pool      = []
    for i, letter in enumerate(["A", "B", "C"][: user["devices"]]):
        dev_id = f"DEV-{user['user_id']}-{letter}"
        os_name = os_list[0] if i == 0 else str(rng.choice(os_list))
        r = float(rng.random())
        trust = "compliant" if r < 0.88 else ("non_compliant" if r < 0.96 else "unknown")
        pool.append({"device_id": dev_id, "device_os": os_name, "device_trust": trust})
    return pool


# CORE GENERATOR

def generate_logs(seed=42):
    """
    Generate all synthetic access events.

    Returns
    -------
    list[dict]
        Sorted list of event dicts (by timestamp ascending).
    """
    rng = np.random.default_rng(seed)

    # Pre-build device pools (deterministic for given seed)
    user_devices = {u["user_id"]: build_device_pool(rng, u) for u in USERS}

    # Pre-compute normalised resource sampling arrays per department
    dept_res_ids   = {}
    dept_res_probs = {}
    for dept, profile in DEPT_PROFILES.items():
        pairs   = profile["resource_weights"]
        ids     = [r for r, _ in pairs]
        weights = np.array([w for _, w in pairs], dtype=float)
        weights /= weights.sum()
        dept_res_ids[dept]   = ids
        dept_res_probs[dept] = weights

    # Per-user history: list of (datetime, outcome, device_id)
    user_history: dict[str, list] = defaultdict(list)

    events        = []
    event_counter = 0

    for day_offset in range(SIM_DAYS):
        day_dt     = SIM_START + timedelta(days=day_offset)
        is_weekend = day_dt.weekday() >= 5          # Saturday = 5, Sunday = 6

        for user in USERS:
            uid    = user["user_id"]
            dept   = user["department"]
            profile = DEPT_PROFILES[dept]

            # Determine activity level 
            if is_weekend:
                if float(rng.random()) > profile["weekend_prob"]:
                    continue
                n_events = max(0, int(rng.poisson(profile["mean_events_weekend"])))
            else:
                n_events = max(0, int(rng.poisson(profile["mean_events_weekday"])))

            if n_events == 0:
                continue

            # Session structure 
            n_sessions = max(1, int(rng.integers(1, 4)))   
            session_ids = [f"sess-{uuid.uuid4().hex[:8]}" for _ in range(n_sessions)]
            session_assign = rng.integers(0, n_sessions, size=n_events)

            work_start, work_end = profile["work_hours"]
            is_it = (dept == "IT Security")

            # Device and network zone are consistent per session 
            devices         = user_devices[uid]
            primary_device  = devices[int(rng.integers(0, len(devices)))]
            session_zones   = [sample_network_zone(rng, profile, is_weekend)
                               for _ in range(n_sessions)]

            # Generate individual events 
            for i in range(n_events):
                event_counter += 1
                sess_idx    = int(session_assign[i])
                session_id  = session_ids[sess_idx]
                zone        = session_zones[sess_idx]

                ts          = sample_timestamp(rng, day_offset, work_start, work_end, is_it)
                res_id      = str(rng.choice(dept_res_ids[dept], p=dept_res_probs[dept]))
                resource    = RESOURCE_MAP[res_id]
                action      = str(rng.choice(resource["actions"]))
                mfa_used    = bool(float(rng.random()) < profile["mfa_prob"])
                outcome     = sample_outcome(rng, profile["mfa_prob"])
                bytes_xfer  = sample_bytes(rng, action, resource["sensitivity"])
                duration_s  = sample_duration(rng, action, resource["sensitivity"])
                ip_addr     = sample_ip(rng, zone)

                # Occasional device switch within a day (10% chance if >1 device)
                if len(devices) > 1 and float(rng.random()) < 0.10:
                    device_info = devices[int(rng.integers(0, len(devices)))]
                else:
                    device_info = primary_device

                # Derived fields
                cutoff_24h    = ts - timedelta(hours=24)
                fail_24h      = sum(
                    1 for (ht, ho, _) in user_history[uid]
                    if ho == "failure" and cutoff_24h <= ht < ts
                )
                seen_devices  = {hd for (_, _, hd) in user_history[uid]}
                first_seen    = device_info["device_id"] not in seen_devices

                # Location derived from network zone
                location = {"corporate": "Office-HQ", "vpn": "Remote"}.get(zone, "External")

                # Assemble event record 
                event = {
                    "timestamp":            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event_id":             f"evt-{event_counter:06d}",
                    "user_id":              uid,
                    "username":             f"User_{uid.replace('-', '_')}",
                    "role":                 user["role"],
                    "department":           dept,
                    "seniority":            user["seniority"],
                    "location":             location,
                    "network_zone":         zone,
                    "ip_address":           ip_addr,
                    "device_id":            device_info["device_id"],
                    "device_os":            device_info["device_os"],
                    "device_trust":         device_info["device_trust"],
                    "resource_id":          res_id,
                    "resource_name":        resource["resource_name"],
                    "sensitivity":          resource["sensitivity"],
                    "action":               action,
                    "outcome":              outcome,
                    "mfa_used":             mfa_used,
                    "session_id":           session_id,
                    "bytes_transferred":    bytes_xfer,
                    "duration_seconds":     duration_s,
                    "failure_count_24h":    fail_24h,
                    "is_first_seen_device": first_seen,
                    "ground_truth_label":   "normal",
                    "anomaly_type":         None,
                }
                events.append(event)
                user_history[uid].append((ts, outcome, device_info["device_id"]))

    # Sort chronologically
    events.sort(key=lambda e: e["timestamp"])
    return events

# OUTPUT WRITERS

SCHEMA = [
    {"field": "timestamp",            "type": "datetime", "description": "ISO 8601 UTC timestamp of the access event",                             "example": "2024-01-15T08:32:11Z"},
    {"field": "event_id",             "type": "string",   "description": "Unique sequential event identifier",                                     "example": "evt-000001"},
    {"field": "user_id",              "type": "string",   "description": "Anonymous user identifier (no real names stored)",                        "example": "ENG-001"},
    {"field": "username",             "type": "string",   "description": "Anonymous display label derived from user_id",                            "example": "User_ENG_001"},
    {"field": "role",                 "type": "string",   "description": "Job role / job title",                                                    "example": "Software Engineer"},
    {"field": "department",           "type": "string",   "description": "Organisational department",                                               "example": "Engineering"},
    {"field": "seniority",            "type": "string",   "description": "Career level: Junior / Mid / Senior / Lead / Executive",                  "example": "Senior"},
    {"field": "location",             "type": "string",   "description": "Physical or logical location: Office-HQ / Remote / External",             "example": "Office-HQ"},
    {"field": "network_zone",         "type": "string",   "description": "Network segment: corporate / vpn / external",                             "example": "corporate"},
    {"field": "ip_address",           "type": "string",   "description": "Source IP address (synthetic, not real)",                                 "example": "10.0.1.45"},
    {"field": "device_id",            "type": "string",   "description": "Unique device identifier registered to the user",                         "example": "DEV-ENG-001-A"},
    {"field": "device_os",            "type": "string",   "description": "Operating system of the device",                                          "example": "Windows 11"},
    {"field": "device_trust",         "type": "string",   "description": "Endpoint compliance state: compliant / non_compliant / unknown",          "example": "compliant"},
    {"field": "resource_id",          "type": "string",   "description": "Target resource identifier",                                              "example": "RES-017"},
    {"field": "resource_name",        "type": "string",   "description": "Human-readable resource name",                                            "example": "Source Code Repository"},
    {"field": "sensitivity",          "type": "string",   "description": "Resource sensitivity label: Public / Internal / Confidential / Privileged","example": "Internal"},
    {"field": "action",               "type": "string",   "description": "Access action: read / write / execute / delete / connect / export",        "example": "read"},
    {"field": "outcome",              "type": "string",   "description": "Result: success / failure / challenge",                                   "example": "success"},
    {"field": "mfa_used",             "type": "boolean",  "description": "Whether MFA was completed for this event",                                "example": "true"},
    {"field": "session_id",           "type": "string",   "description": "Session identifier grouping related events",                              "example": "sess-abc123ef"},
    {"field": "bytes_transferred",    "type": "integer",  "description": "Data volume in bytes (0 for connect and delete actions)",                  "example": "4096"},
    {"field": "duration_seconds",     "type": "integer",  "description": "Time spent on the access in seconds",                                     "example": "42"},
    {"field": "failure_count_24h",    "type": "integer",  "description": "Count of failed attempts by this user in the prior 24 hours (no leakage)","example": "0"},
    {"field": "is_first_seen_device", "type": "boolean",  "description": "True if this device_id has never appeared for this user in prior events", "example": "false"},
    {"field": "ground_truth_label",   "type": "string",   "description": "Ground truth annotation: normal / anomalous",                             "example": "normal"},
    {"field": "anomaly_type",         "type": "string",   "description": "Anomaly category if anomalous, else null (set by anomaly_injector.py)",   "example": "null"},
]


def write_jsonl(events, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, default=str) + "\n")
    print(f"  [JSONL]   {len(events):,} events  →  {path}")


def write_csv(events, path):
    if not events:
        return
    fieldnames = list(events[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
    print(f"  [CSV]     {len(events):,} events  →  {path}")


def write_schema(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SCHEMA, f, indent=2)
    print(f"  [SCHEMA]  {len(SCHEMA)} fields    →  {path}")


def print_summary(events):
    print(f"\n{'─'*55}")
    print(f"  Total events  : {len(events):,}")
    if events:
        print(f"  Date range    : {events[0]['timestamp']}  →  {events[-1]['timestamp']}")
        print(f"  Normal events : {sum(1 for e in events if e['ground_truth_label']=='normal'):,}")
        print(f"\n  Events by department:")
        for dept, count in sorted(Counter(e["department"] for e in events).items()):
            print(f"    {dept:<22} {count:>5,}")
        print(f"\n  Outcome breakdown:")
        for outcome, count in sorted(Counter(e["outcome"] for e in events).items()):
            pct = 100 * count / len(events)
            print(f"    {outcome:<12} {count:>5,}  ({pct:.1f}%)")
    print(f"{'─'*55}\n")


# ENTRY POINT

def main():
    parser = argparse.ArgumentParser(
        description="TechnoMed Solutions — Synthetic access log generator (Task 1)"
    )
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for full reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="logs",
                        help="Output filename stem (default: logs → logs.jsonl / logs.csv)")
    parser.add_argument("--format", type=str, default="both",
                        choices=["jsonl", "csv", "both"],
                        help="Output format (default: both)")
    args = parser.parse_args()

    print(f"\n  TechnoMed Solutions — Access Log Generator")
    print(f"  Seed : {args.seed}  |  Days : {SIM_DAYS}  |  "
          f"Users : {len(USERS)}  |  Resources : {len(RESOURCES)}")

    print("\nGenerating events ...")
    events = generate_logs(seed=args.seed)
    print_summary(events)

    print("Writing output files ...")
    if args.format in ("jsonl", "both"):
        write_jsonl(events, f"{args.output}.jsonl")
    if args.format in ("csv", "both"):
        write_csv(events, f"{args.output}.csv")
    write_schema("schema.json")

    print(f"\n  Done. Reproduce exactly with:  python data_generator.py --seed {args.seed}\n")


if __name__ == "__main__":
    main()
