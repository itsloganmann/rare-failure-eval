"""Frozen two-stage, fixed-task-weight synthetic utility experiment."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.stats import beta

from record_types import BoolArray, FloatArray, IntArray, MetricName, RunRecord

POLICIES = ("uniform_all", "uniform", "neyman", "event_following")
SCENARIOS = ("no_rare_loss", "concentrated", "diffuse")
BUDGETS = (6400, 25600)
PRIMARY_SEEDS = tuple(range(970000, 971024))
SMOKE_SEEDS = (940000, 940001, 940002)
LOSS = 800.0
PILOT_DRAWS = 4
FLOOR = 0.25
SD_FLOOR = 0.02
METRICS: tuple[MetricName, ...] = (
    "simple_regret",
    "best_agent_error",
    "kendall_error",
    "utility_rmse",
    "risk_rmse",
    "any_discovery",
    "risky_cell_discovery",
    "pilot_hit",
    "risk_family_coverage",
    "risky_interval_width",
    "zero_event_fraction",
    "zero_event_upper",
)


@dataclass(frozen=True)
class World:
    """Simulator state; deliberately never passed to allocation functions."""

    seed: int
    scenario: str
    reward_mean: FloatArray
    half_width: FloatArray
    failure_prob: FloatArray
    weights: FloatArray
    truth: FloatArray
    agent_tie_order: IntArray
    cell_tie_order: IntArray


@dataclass(frozen=True)
class Pilot:
    """Only observed outcomes are exposed to adaptive allocation."""

    utility: FloatArray
    failure: BoolArray


def make_world(seed: int, scenario: str) -> World:
    """Analytic truth with independently permuted agent/task identities."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0])))
    agents, tasks = rng.permutation(4), rng.permutation(8)
    weights = (np.array([1, 1, 1, 1, 2, 2, 4, 4]) / 16)[tasks]
    offsets = np.array([-0.06, 0.04, -0.03, 0.06, -0.04, 0.02, 0.01, 0])[tasks]
    offsets -= offsets @ weights
    means = np.array([0.74, 0.70, 0.62, 0.56])[agents, None] + offsets
    widths = np.array([0.02, 0.04, 0.08, 0.12, 0.16, 0.06, 0.10, 0.18])[tasks]
    risk = np.zeros((4, 8))
    risky_task = rng.integers(8)
    if scenario == "concentrated":
        risk[np.flatnonzero(agents == 0)[0], risky_task] = 0.001
    elif scenario == "diffuse":
        risk[:] = np.array([0.00020, 0.00010, 0.00004, 0.00002])[agents, None]
    return World(
        seed,
        scenario,
        means,
        widths,
        risk,
        weights,
        (means - LOSS * risk) @ weights,
        rng.permutation(4),
        rng.permutation(32),
    )


def draw_cell(
    world: World,
    agent: int,
    task: int,
    phase: str,
    count: int | np.int64,
) -> tuple[FloatArray, BoolArray]:
    """Replay prefixes, keyed by cell and phase; policies cannot inspect streams."""
    phase_key = {"pilot": 1, "estimation": 2}[phase]
    key = [world.seed, phase_key, agent, task]
    reward_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key + [0])))
    event_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key + [1])))
    reward = world.reward_mean[agent, task] + world.half_width[task] * (
        2 * (reward_rng.random(count) < 0.5) - 1
    )
    failures = event_rng.random(count) < world.failure_prob[agent, task]
    return reward - LOSS * failures, failures


def policy_scores(policy: str, pilot: Pilot, weights: FloatArray) -> FloatArray:
    """Public weights and observed pilot only; no latent-world access."""
    if policy in ("uniform", "uniform_all"):
        return np.ones((4, 8))
    if policy == "neyman":
        return weights[None, :] * np.maximum(
            pilot.utility.std(axis=2, ddof=1), SD_FLOOR
        )
    if policy == "event_following":
        scores = pilot.failure.any(axis=2).astype(float)
        # Fall back within each agent, not across agents: per-agent costs are fixed.
        scores[scores.sum(axis=1) == 0] = 1
        return scores
    raise ValueError("unknown policy")


def allocate(scores: FloatArray, total: int, tie_order: IntArray) -> IntArray:
    """Exact integer task allocation with equal agent totals and a uniform floor."""
    scores = np.asarray(scores, dtype=float)
    if (
        total < 128
        or total % 4
        or scores.shape != (4, 8)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores.sum(axis=1) <= 0).any()
    ):
        raise ValueError("invalid allocation inputs")
    if sorted(tie_order) != list(range(32)):
        raise ValueError("invalid tie order")
    probabilities = FLOOR / 8 + (1 - FLOOR) * scores / scores.sum(axis=1, keepdims=True)
    targets = probabilities * (total // 4)
    counts = np.floor(targets).astype(int)
    priority = np.argsort(tie_order).reshape(4, 8)
    for agent in range(4):
        left = total // 4 - counts[agent].sum()
        order = np.lexsort((priority[agent], -(targets[agent] - counts[agent])))
        counts[agent, order[:left]] += 1
    return counts


def estimate(
    sums: FloatArray | IntArray, counts: IntArray, weights: FloatArray
) -> FloatArray:
    """Conditional-unbiased fixed task-weight mean using independent draws only."""
    if np.any(counts <= 0):
        raise ValueError("positive cell counts required")
    return (sums / counts) @ weights


def clopper_pearson(
    successes: IntArray, counts: IntArray
) -> tuple[FloatArray, FloatArray]:
    """One-look exact binomial intervals, Bonferroni over the 32 cells."""
    successes, counts = np.asarray(successes), np.asarray(counts)
    if np.any(counts <= 0) or np.any(successes < 0) or np.any(successes > counts):
        raise ValueError("invalid binomial counts")
    tail = 0.05 / 64
    lo = np.zeros(successes.shape, dtype=float)
    hi = np.ones(successes.shape, dtype=float)
    positive = successes > 0
    incomplete = successes < counts
    lo[positive] = beta.ppf(
        tail, successes[positive], counts[positive] - successes[positive] + 1
    )
    hi[incomplete] = beta.ppf(
        1 - tail, successes[incomplete] + 1, counts[incomplete] - successes[incomplete]
    )
    return lo, hi


def rank(values: FloatArray, tie_order: IntArray) -> IntArray:
    """Descending utility with independent exact-tie resolution."""
    return np.lexsort((np.argsort(tie_order), -values))


def run_one(world: World, policy: str, budget: int) -> RunRecord:
    """Produce one terminal record; all outcome draws and their costs are retained."""
    if budget < 256 or budget % 4:
        raise ValueError("budget must be at least 256 and divisible by 4")
    pilot_n = 0 if policy == "uniform_all" else PILOT_DRAWS
    pilot_u = np.zeros((4, 8, pilot_n))
    pilot_f = np.zeros((4, 8, pilot_n), dtype=bool)
    if pilot_n:
        for agent, task in product(range(4), range(8)):
            pilot_u[agent, task], pilot_f[agent, task] = draw_cell(
                world, agent, task, "pilot", pilot_n
            )
    pilot = Pilot(pilot_u, pilot_f)
    counts = allocate(
        policy_scores(policy, pilot, world.weights),
        budget - 32 * pilot_n,
        world.cell_tie_order,
    )
    sums = np.zeros((4, 8))
    failures = np.zeros((4, 8), dtype=int)
    for agent, task in product(range(4), range(8)):
        utility, event = draw_cell(
            world, agent, task, "estimation", counts[agent, task]
        )
        sums[agent, task], failures[agent, task] = utility.sum(), event.sum()
    utility_hat = estimate(sums, counts, world.weights)
    risk_hat = estimate(failures, counts, world.weights)
    lo, hi = clopper_pearson(failures, counts)
    estimated_order = rank(utility_hat, world.agent_tie_order)
    true_order = rank(world.truth, world.agent_tie_order)
    true_position = np.argsort(true_order)
    inversions = sum(
        true_position[estimated_order[i]] > true_position[estimated_order[j]]
        for i in range(4)
        for j in range(i + 1, 4)
    )
    pilot_counts = pilot_f.sum(axis=2)
    discovered = (failures + pilot_counts) > 0
    risky = world.failure_prob > 0
    zero = failures == 0
    row: RunRecord = {
        "seed": world.seed,
        "scenario": world.scenario,
        "policy": policy,
        "budget": budget,
        "cost": int(counts.sum() + 32 * pilot_n),
        "pilot_cost": 32 * pilot_n,
        "stage2_counts": counts.tolist(),
        "stage2_utility_sums": sums.tolist(),
        "stage2_failures": failures.tolist(),
        "pilot_failures": pilot_counts.tolist(),
        "pilot_utility": pilot_u.tolist(),
        "task_weights": world.weights.tolist(),
        "true_utility": world.truth.tolist(),
        "true_cell_risk": world.failure_prob.tolist(),
        "estimated_utility": utility_hat.tolist(),
        "estimated_agent_risk": risk_hat.tolist(),
        "estimated_cell_risk": (failures / counts).tolist(),
        "risk_lower": lo.tolist(),
        "risk_upper": hi.tolist(),
        "selected_agent": int(estimated_order[0]),
        "true_best_agent": int(true_order[0]),
        "estimated_order": estimated_order.tolist(),
        "true_order": true_order.tolist(),
        "simple_regret": float(world.truth.max() - world.truth[estimated_order[0]]),
        "best_agent_error": int(estimated_order[0] != true_order[0]),
        "kendall_error": float(inversions / 6),
        "utility_rmse": float(np.sqrt(np.mean((utility_hat - world.truth) ** 2))),
        "risk_rmse": float(
            np.sqrt(np.mean((risk_hat - world.failure_prob @ world.weights) ** 2))
        ),
        "any_discovery": int(discovered.any()),
        "risky_cell_discovery": float(discovered[risky].mean())
        if risky.any()
        else None,
        "pilot_hit": int(pilot_f.any()) if pilot_n else None,
        "risk_family_coverage": int(
            ((lo <= world.failure_prob) & (world.failure_prob <= hi)).all()
        ),
        "risky_interval_width": float((hi - lo)[risky].mean()) if risky.any() else None,
        "zero_event_fraction": float(zero.mean()),
        "zero_event_upper": float(hi[zero].mean()) if zero.any() else None,
    }
    return row


def validate_grid(
    rows: Sequence[RunRecord],
    seeds: Sequence[int],
    budgets: Sequence[int],
) -> None:
    """Block summaries unless every exact planned key and sample budget is present."""
    expected = set(product(seeds, SCENARIOS, POLICIES, budgets))
    keys = [(r["seed"], r["scenario"], r["policy"], r["budget"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete or duplicated grid")
    for row in rows:
        counts = np.asarray(row["stage2_counts"])
        pilot_cost = 0 if row["policy"] == "uniform_all" else 128
        if (
            row["cost"] != row["budget"]
            or row["pilot_cost"] != pilot_cost
            or counts.shape != (4, 8)
            or (counts <= 0).any()
            or not np.all(counts == counts.astype(int))
            or counts.sum() + pilot_cost != row["budget"]
            or not np.all(counts.sum(axis=1) + pilot_cost // 4 == row["budget"] // 4)
        ):
            raise ValueError("cost or allocation mismatch")
