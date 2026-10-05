# Fresh-seed results

The metric disagreement seen in an earlier 128-world pilot remained in a fresh
1,024-world run. These worlds use new random seeds with the same fixed generator;
they are not new task distributions or real agents.

At 25,600 simulated draws per world and policy:

| Risk setting | Policy | Wrong selections / 1,024 | Mean regret |
| --- | --- | ---: | ---: |
| Concentrated | Uniform, full budget | 455 | 0.0267578 |
| Concentrated | Variance allocation | 539 | 0.0193262 |
| Diffuse | Uniform, full budget | 541 | 0.0207695 |
| Diffuse | Variance allocation | 602 | 0.0228867 |

Regret is the expected utility of the true best agent minus that of the selected
agent. Selection error only asks whether they differ. In the concentrated setting,
variance allocation made more mistakes but its mistakes cost less on average.
This relationship depends on the setting and budget. At 6,400 draws, variance
allocation had slightly fewer observed wrong selections than full-budget uniform
in both risk settings; see the complete report rather than extrapolating one row.

## Uncertainty

Paired descriptive bootstrap intervals for variance allocation minus full-budget
uniform, at 25,600 draws:

| Setting | Metric | Mean difference | 2.5/97.5 percentile endpoints |
| --- | --- | ---: | --- |
| Concentrated | Regret | -0.0074316 | [-0.0100195, -0.0049512] |
| Concentrated | Selection error | +8.2031 percentage points | [+5.4663, +11.2305] |
| Diffuse | Regret | +0.0021172 | [+0.0008359, +0.0033438] |
| Diffuse | Selection error | +5.9570 percentage points | [+2.9297, +8.9844] |

These are 2,000 paired-seed resamples, unadjusted for multiple comparisons.
They describe Monte Carlo variation under the chosen generator, not uncertainty
about transfer to real agents. No significance or general superiority claim is made.
The full summary retains all six policy contrasts for every metric and condition.

## A pilot can miss almost all rare failures

Four draws in a cell with failure probability 0.001 have probability
`1 - (1 - 0.001)^4 = 0.003994` of seeing at least one failure.

The concentrated-risk pilot observed an event in only 4 of 1,024 worlds. This
cannot support a claim that the method reliably learned where rare risk lived.
The variance rule can also allocate according to ordinary reward variation and
known task weights. No event discovered is not the same as no risk present.

## Scope and checks

All 24 conditions completed: 24,576 records and 393,216,000 simulated draws.
Budgets include pilot cost, each agent has equal cost, task weights remain fixed,
and final means use independent second-stage samples. The full-budget uniform
baseline spends no samples on a pilot. A second uniform baseline matches pilot
cost, and event-following allocation is also retained in the full report.

The manifest preceded collection. The source/protocol hashes remained unchanged,
all record and summary hashes verified, and an independent arithmetic audit
reconstructed the primary metrics with maximum absolute difference 0.0.
No paid inference, private datasets or employer experiments were used.

The run provides a concrete reason to report both selection frequency and loss
severity. It does not show that minimizing expected regret enforces a catastrophic
risk constraint. Adding such a constraint and testing public real-agent traces
are future work, not completed results.
