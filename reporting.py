"""Complete-grid descriptive summaries and paired-world bootstrap intervals."""

from collections.abc import Sequence
from itertools import product
from typing import cast

import numpy as np

from experiment import METRICS, POLICIES, SCENARIOS, validate_grid
from record_types import (
    BootstrapResult,
    ContrastRecord,
    IntArray,
    NullableMetric,
    PolicyMean,
    RunRecord,
    SummaryRecord,
)

CONTRASTS = (
    ("uniform", "uniform_all"),
    ("neyman", "uniform_all"),
    ("event_following", "uniform_all"),
    ("neyman", "uniform"),
    ("event_following", "uniform"),
    ("neyman", "event_following"),
)


def bootstrap_contrast(
    left: Sequence[NullableMetric],
    right: Sequence[NullableMetric],
    resampled_indices: IntArray | Sequence[Sequence[int]],
) -> BootstrapResult:
    """Use identical seed indices on both sides; nulls are never coerced to zero."""
    if all(v is None for v in left) or all(v is None for v in right):
        return {"mean_difference": None, "low": None, "high": None, "paired_seeds": 0}
    if any(v is None for v in left) or any(v is None for v in right):
        raise ValueError("partially missing comparison; no silent seed dropping")
    delta = np.asarray(left) - np.asarray(right)
    replicates = delta[np.asarray(resampled_indices)].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975])
    return {
        "mean_difference": float(delta.mean()),
        "low": float(low),
        "high": float(high),
        "paired_seeds": len(delta),
    }


def summarize(
    rows: Sequence[RunRecord],
    seeds: Sequence[int],
    budgets: Sequence[int],
) -> SummaryRecord:
    """Validate first, then retain every prespecified metric and contrast."""
    validate_grid(rows, seeds, budgets)
    lookup = {(r["seed"], r["scenario"], r["policy"], r["budget"]): r for r in rows}
    rng = np.random.Generator(np.random.PCG64(950001))
    indices = rng.integers(len(seeds), size=(2000, len(seeds)))
    means: list[PolicyMean] = []
    contrasts: list[ContrastRecord] = []
    for scenario, budget in product(SCENARIOS, budgets):
        values = {
            policy: {
                metric: [
                    lookup[seed, scenario, policy, budget][metric] for seed in seeds
                ]
                for metric in METRICS
            }
            for policy in POLICIES
        }
        for policy in POLICIES:
            record: dict[str, str | int | NullableMetric] = {
                "scenario": scenario,
                "budget": budget,
                "policy": policy,
                "seeds": len(seeds),
            }
            for metric in METRICS:
                present = [v for v in values[policy][metric] if v is not None]
                if present and len(present) != len(seeds):
                    raise ValueError(
                        "partially missing summary; no silent seed dropping"
                    )
                record[metric] = float(np.mean(present)) if present else None
            means.append(cast(PolicyMean, record))
        for (left, right), metric in product(CONTRASTS, METRICS):
            contrasts.append(
                {
                    "scenario": scenario,
                    "budget": budget,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    **bootstrap_contrast(
                        values[left][metric], values[right][metric], indices
                    ),
                }
            )
    return {
        "scope": "synthetic descriptive results; no p-values or significance claims",
        "interval_method": "2000 paired-seed percentile resamples, seed 950001, unadjusted",
        "metric_definitions": {
            "utility_rmse": "mean per-world agent utility RMSE, not pooled RMSE",
            "risk_rmse": "mean per-world agent risk RMSE, not pooled RMSE",
        },
        "policy_means": means,
        "paired_contrasts": contrasts,
    }


def markdown_report(summary: SummaryRecord, mode: str) -> str:
    """Compact primary-endpoint table; exhaustive diagnostics stay in summary.json."""
    lines = [
        f"# {mode.upper()} results: bounded synthetic risk utility",
        "",
        "SMOKE is a software check only; do not interpret smoke policy differences."
        if mode != "primary"
        else "Exploratory synthetic results, not real agent evaluation.",
        "",
        "All policies, scenarios and budgets are reported below. Every scalar metric",
        "and all six paired contrasts are retained in `summary.json`.",
        "",
        "RMSE summaries are mean per-world RMSE, not globally pooled RMSE.",
        "",
        "| Scenario | Budget | Policy | Simple regret | Best-agent error | Kendall error |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary["policy_means"]:
        lines.append(
            f"| {row['scenario']} | {row['budget']} | {row['policy']} | "
            f"{row['simple_regret']:.6f} | {row['best_agent_error']:.6f} | "
            f"{row['kendall_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "Four pilot draws detect p=.001 risk only about 0.4% of the time.",
            "Failure intervals are one-look Clopper-Pearson with Bonferroni over",
            "32 cells per run, not anytime bounds. Zero events do not establish safety.",
            "Only independent estimation samples contribute to point estimates.",
            "Task weights and per-agent budgets remain fixed. Selected maxima may be optimistic.",
            "Bootstrap intervals are descriptive, unadjusted and not significance tests.",
            "Sequential Halving, real-agent transfer and frontier claims were not tested.",
            "",
        ]
    )
    return "\n".join(lines)
