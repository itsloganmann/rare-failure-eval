"""Recompute selection metrics from saved sufficient statistics, without the engine."""

import argparse
import json
from collections.abc import Iterable, Sequence
from itertools import product
from pathlib import Path

import numpy as np

from record_types import RunRecord

POLICIES = ("uniform_all", "uniform", "neyman", "event_following")
SCENARIOS = ("no_rare_loss", "concentrated", "diffuse")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_rows(
    rows: Iterable[RunRecord], seeds: Sequence[int], budgets: Sequence[int]
) -> dict[str, object]:
    """Check the whole grid and primary metrics; retain no raw rows in memory."""
    expected = set(product(seeds, SCENARIOS, POLICIES, budgets))
    seen = set()
    max_difference = 0.0
    total_cost = 0
    groups = {}
    for row in rows:
        key = (row["seed"], row["scenario"], row["policy"], row["budget"])
        require(key in expected and key not in seen, "unexpected or duplicate grid key")
        seen.add(key)
        counts = np.asarray(row["stage2_counts"], dtype=float)
        sums = np.asarray(row["stage2_utility_sums"], dtype=float)
        failures = np.asarray(row["stage2_failures"], dtype=float)
        weights = np.asarray(row["task_weights"], dtype=float)
        truth = np.asarray(row["true_utility"], dtype=float)
        require(counts.shape == sums.shape == failures.shape == (4, 8), "cell shape")
        require(weights.shape == (8,) and truth.shape == (4,), "target shape")
        require(
            all(np.isfinite(x).all() for x in (counts, sums, failures, weights, truth)),
            "nonfinite input",
        )
        require(
            bool(np.all(counts > 0) and np.all(counts == np.floor(counts))), "counts"
        )
        require(
            bool(
                np.all(failures >= 0)
                and np.all(failures <= counts)
                and np.all(failures == np.floor(failures))
            ),
            "failure counts",
        )
        require(
            bool(np.all(weights > 0) and np.isclose(weights.sum(), 1)), "task weights"
        )
        pilot_cost = 0 if row["policy"] == "uniform_all" else 128
        require(row["pilot_cost"] == pilot_cost, "pilot cost")
        require(counts.sum() + pilot_cost == row["cost"] == row["budget"], "total cost")
        require(
            bool(np.all(counts.sum(axis=1) + pilot_cost / 4 == row["budget"] / 4)),
            "per-agent cost",
        )
        estimated = (sums / counts) @ weights
        risk_estimated = (failures / counts) @ weights
        np.testing.assert_allclose(
            estimated, row["estimated_utility"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            risk_estimated, row["estimated_agent_risk"], atol=1e-12, rtol=0
        )
        selected = row["selected_agent"]
        require(type(selected) is int and 0 <= selected < 4, "selected agent")
        require(estimated[selected] == estimated.max(), "selected utility maximum")
        for order_name, values in (
            ("estimated_order", estimated),
            ("true_order", truth),
        ):
            order = row[order_name]
            require(sorted(order) == list(range(4)), "invalid ranking permutation")
            require(
                all(values[a] >= values[b] for a, b in zip(order, order[1:])),
                "invalid descending ranking",
            )
        require(selected == row["estimated_order"][0], "selection and ranking disagree")
        best = row["true_best_agent"]
        require(
            type(best) is int and 0 <= best < 4 and truth[best] == truth.max(),
            "true best",
        )
        require(best == row["true_order"][0], "true selection and ranking disagree")
        regret = float(truth.max() - truth[selected])
        error = int(truth[selected] < truth.max())
        gap = abs(regret - row["simple_regret"])
        require(
            gap < 1e-12 and error == row["best_agent_error"], "primary metric mismatch"
        )
        tie_rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([row["seed"], 0]))
        )
        tie_rng.permutation(4)
        tie_rng.permutation(8)
        tie_rng.integers(8)
        tie_order = list(tie_rng.permutation(4))
        for order_name, values in (
            ("estimated_order", estimated),
            ("true_order", truth),
        ):
            expected_order = sorted(
                range(4), key=lambda a: (-values[a], tie_order.index(a))
            )
            require(row[order_name] == expected_order, "prescribed tie order mismatch")
        max_difference = max(max_difference, gap)
        total_cost += row["cost"]
        group_key = (row["scenario"], row["budget"], row["policy"])
        group = groups.setdefault(group_key, {"n": 0, "regret": 0.0, "errors": 0})
        group["n"] += 1
        group["regret"] += regret
        group["errors"] += error
    require(seen == expected, "incomplete grid")
    return {
        "rows": len(seen),
        "simulated_draws": total_cost,
        "max_primary_metric_difference": max_difference,
        "conditions": [
            {
                "scenario": k[0],
                "budget": k[1],
                "policy": k[2],
                "worlds": v["n"],
                "mean_regret": v["regret"] / v["n"],
                "wrong_selections": v["errors"],
            }
            for k, v in sorted(groups.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.out / "manifest.json").read_text())
    expected_seeds = (
        list(range(970000, 971024))
        if manifest["mode"] == "primary"
        else [940000, 940001, 940002]
    )
    require(manifest["mode"] in ("primary", "smoke"), "unsupported audit mode")
    require(
        manifest["seeds"] == expected_seeds and manifest["budgets"] == [6400, 25600],
        "manifest does not match frozen grid",
    )
    require(
        manifest["policies"] == list(POLICIES)
        and manifest["scenarios"] == list(SCENARIOS),
        "manifest does not match policies/scenarios",
    )
    with (args.out / "results.jsonl").open() as stream:
        audit = audit_rows(
            (json.loads(line) for line in stream),
            manifest["seeds"],
            manifest["budgets"],
        )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
