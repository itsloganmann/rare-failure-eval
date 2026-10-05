# Rare Failure Eval

**An agent evaluator can make more wrong picks and still lose less utility.**

A reproducible Python experiment on rare failures, fixed evaluation budgets,
and the gap between selection accuracy and the cost of a mistake.

[Results](FINDINGS.md) · [Full comparison](results/REPORT.md) · [Protocol](PROTOCOL.md) · [Raw data](results/results.jsonl.gz)

<img src="figures/headline.png" alt="Uniform versus variance allocation. Concentrated risk: 455 versus 539 wrong picks, regret 0.0268 versus 0.0193. Diffuse risk: 541 versus 602 wrong picks, regret 0.0208 versus 0.0229. Each condition has 1,024 worlds and 25,600 draws per policy per world." width="540">

## Results

Same budget. Same tasks. Same 1,024 simulated worlds.

| At 25,600 draws per policy per world | Uniform, full budget | Variance allocation |
| :--- | ---: | ---: |
| Concentrated risk: wrong picks | 455 / 1,024 | 539 / 1,024 |
| Concentrated risk: mean utility lost | 0.0268 | 0.0193 |
| Diffuse risk: wrong picks | 541 / 1,024 | 602 / 1,024 |
| Diffuse risk: mean utility lost | 0.0208 | 0.0229 |

**Selection accuracy and decision cost can rank evaluators differently.** In the
concentrated setting, variance allocation increased the wrong-selection rate
from 44.4% to 52.6%, while reducing mean utility lost by 27.8% (0.0268 to 0.0193).
It picked the true best agent less often, but its mistakes were less costly on
average. Under diffuse risk, both metrics were worse.

For an evaluation pipeline, the practical lesson is to measure both how often
it selects a suboptimal agent and how much expected utility that choice loses.
Accuracy counts a near tie and a costly mistake equally. Regret captures the
utility gap, but does not enforce a catastrophic-risk limit. These synthetic
results motivate reporting both; they do not establish deployment safety or a
general advantage for adaptive evaluation.

## Try it

Python 3.13. CPU only. No model API, credentials or private data.

```sh
git clone https://github.com/itsloganmann/rare-failure-eval.git
cd rare-failure-eval
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_experiment.py --mode smoke --out outputs/smoke
```

Read `outputs/smoke/REPORT.md` to inspect the smoke run. Use a new output directory
for each run. The published findings come from the full run, not smoke mode.

<details>
<summary>Tests, full replication and independent verification</summary>

```sh
.venv/bin/python -m pytest -q
.venv/bin/python run_experiment.py --mode primary --approved-primary --out outputs/replication
.venv/bin/python run_experiment.py --verify --out outputs/replication
.venv/bin/python independent_audit.py outputs/replication
.venv/bin/python make_figures.py outputs/replication/summary.json --out outputs/figures
```

The primary flag is an execution gate, not evidence of peer review. To verify the
distributed run without resampling:

```sh
mkdir -p outputs/published
cp results/*.json results/REPORT.md outputs/published/
gzip -dc results/results.jsonl.gz > outputs/published/results.jsonl
.venv/bin/python run_experiment.py --verify --out outputs/published
.venv/bin/python independent_audit.py outputs/published
```

The manifest hashes the frozen study code and outputs. The separate arithmetic
audit reconstructs weighted estimates, rankings, ties, selection error and regret
from saved sufficient statistics.

</details>

## What is inside

Four synthetic agents, eight tasks, three risk settings, two budgets and four
sampling policies. Utility is `reward - 800 * failure`, using fixed task weights.
Regret is the true best agent's expected utility minus the selected agent's.

| Policy | Allocation |
| :--- | :--- |
| Uniform, full budget | Every draw goes into balanced estimation |
| Uniform, pilot matched | A pilot followed by balanced independent estimation |
| Variance allocation | A four-draw-per-cell pilot sets a frozen Neyman allocation, with a uniform floor |
| Follow pilot failures | Allocate toward observed failures; otherwise use uniform sampling |

The full run contains 24,576 records and 393,216,000 simulated draws.
[All conditions](figures/all_conditions.png) · [Paired intervals](FINDINGS.md#uncertainty) · [Audit](results/audit.json)

## Scope

- **Synthetic, exploratory replication.** Fresh seeds after a 128-world pilot;
  not publicly preregistered and not a real-agent benchmark.
- **No universal improvement claim.** Results depend on setting and budget.
  Risk settings do not match total event mass, so their difference is not causal.
- **Rare failures are easy to miss.** Four draws detect a 0.001-probability event
  only 0.4% of the time. No observed failures does not establish safety.
- **Expected regret is not a safety constraint.** Paired bootstrap intervals are
  descriptive and unadjusted. One-look risk bounds are not anytime guarantees.

## Background

Inspired by [Active Evaluation of General Agents](https://arxiv.org/abs/2601.07651v2),
Lanctot et al. (2026). This independent experiment studies rare-loss utility and
sampling allocation; it does not reproduce that paper's Elo or Soft Condorcet
algorithms. No employer data or institutional endorsement is involved.
