# Risk-Adaptive Access Control (RAAC)

A full pipeline that replaces static, rule-based access control with a
machine-learning risk-scoring engine, built for a synthetic healthcare
enterprise ("TechnoMed Solutions"). Every access event gets a continuous
risk score (0–100) that is mapped to **allow / challenge / block**, with
sensitivity-aware thresholds and step-up authentication.

This started as a postgraduate assignment (MSc Cybersecurity & AI
Technologies, University of Piraeus) — the full write-up, with methodology,
results and discussion, is in [`docs/RAAC_Assignment_Report.docx`](docs/RAAC_Assignment_Report.docx).

All data is synthetic. No real individuals or organisations are represented.

## Results at a glance

| | Precision | Recall | F1 |
|---|---|---|---|
| Static rule-based baseline | 0.418 | 0.728 | 0.531 |
| RAAC risk-scoring engine | 0.935 | 0.969 | 0.952 |

13,440 synthetic access events over 14 days, 52 users across 6 departments,
27 resources across 4 sensitivity tiers, with 6 categories of injected
anomalies (impossible travel, off-hours privileged access, first-seen
device on a sensitive resource, unusual resource for role, bulk data
exfiltration, privilege escalation).

## Pipeline

Each script reads the previous script's output. Run them in order from the
repo root:

| Step | Script | Input | Output |
|---|---|---|---|
| 1. Generate synthetic logs | `src/data_generator.py` | — | `logs.jsonl` |
| 2. Inject anomalies | `src/anomaly_injector.py` | `logs.jsonl` | `data/logs_with_anomalies.jsonl` |
| 3. Static baseline policy | `src/baseline_policy.py` | `logs_with_anomalies.jsonl` | `data/baseline_decisions.csv` |
| 4. Train risk model | `src/risk_model.py --seed 42` | `logs_with_anomalies.jsonl` | `data/risk_scores.csv`, `model.pkl`, `transformer.pkl` |
| 5. Adaptive decisions | `src/decision_engine.py` | `risk_scores.csv`, `baseline_decisions.csv` | `data/final_decisions.csv` |
| 6. Evaluate & explain (SHAP) | `src/evaluation.py` | `logs_with_anomalies.jsonl`, `risk_scores.csv`, `model.pkl`, `transformer.pkl` | `data/explanations.json` |
| 7. Simulate concept drift | `src/drift_simulation.py` | `logs_with_anomalies.jsonl` | `data/drifted_decisions.csv` |

```bash
pip install -r requirements.txt

cd src
python data_generator.py
python anomaly_injector.py
python baseline_policy.py
python risk_model.py --seed 42
python decision_engine.py
python evaluation.py
python drift_simulation.py
```

The scripts as written read/write in the working directory they're run
from, so either run them from inside `src/` as above, or pass explicit
`--input` / `--output` paths (each script accepts them via `argparse`) to
point at `../data/`.

## Repository structure

```
.
├── src/            pipeline scripts (steps 1–7 above)
├── data/           output artifacts already committed, so results can be
│                   inspected without re-running the pipeline
├── docs/           the full assignment report
├── requirements.txt
└── README.md
```

`data/` does **not** include `logs.jsonl` (the pre-anomaly-injection log) or
the trained `model.pkl` / `transformer.pkl` — these are large, fully
regenerable from the seeded scripts, and pickle files aren't something you
should distribute/trust blindly anyway. See `.gitignore`.

## Reproducibility

Every script with randomness is seeded (`--seed 42` by default). The
train/test split is time-based — first 10 days train, last 4 days test —
with no shuffling, so there's no temporal leakage. Features that need
historical context (peer-resource rarity, user-resource novelty, byte
volume z-scores) are fit on training data only.

## Requirements

Python 3.10+.

```bash
pip install -r requirements.txt
```

## License

MIT — see [LICENSE](LICENSE). Swap this out if your university's academic
integrity policy prefers something more restrictive (e.g. all-rights-reserved)
for coursework you might resubmit or extend later.
