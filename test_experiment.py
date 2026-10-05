


import inspect
from dataclasses import fields

import numpy as np
import pytest

from experiment import (
    BUDGETS,
    POLICIES,
    SCENARIOS,
    Pilot,
    allocate,
    clopper_pearson,
    draw_cell,
    estimate,
    make_world,
    policy_scores,
    run_one,
    validate_grid,
)


def test_large_loss_reverses_nominal_best_and_truth_is_analytic() -> None:
    for seed in range(940010, 940042):
        world = make_world(seed, "concentrated")
        assert np.argmax(world.reward_mean @ world.weights) != np.argmax(world.truth)
        np.testing.assert_allclose(
            world.truth, (world.reward_mean - 800 * world.failure_prob) @ world.weights
        )
        assert len(np.unique(world.truth)) == 4
        assert np.all(world.reward_mean - world.half_width >= 0)
        assert np.all(world.reward_mean + world.half_width <= 1)


def test_world_permutations_vary_risk_indices_without_changing_task_target() -> None:
    indices = {
        tuple(np.argwhere(make_world(s, "concentrated").failure_prob)[0])
        for s in range(940010, 940042)
    }
    assert len(indices) > 8
    for scenario in SCENARIOS:
        world = make_world(940011, scenario)
        assert world.weights.sum() == 1
        if scenario == "diffuse":
            np.testing.assert_allclose(np.sort(world.truth), [0.544, 0.58, 0.588, 0.62])


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("policy", POLICIES)
def test_equal_exact_total_cost_and_strict_positive_estimation_counts(
    budget: int, policy: str
) -> None:
    world = make_world(940000, "concentrated")
    row = run_one(world, policy, budget)
    counts = np.array(row["stage2_counts"])
    assert counts.sum() + row["pilot_cost"] == budget == row["cost"]
    np.testing.assert_array_equal(
        counts.sum(axis=1) + row["pilot_cost"] // 4, budget // 4
    )
    assert counts.min() >= int(0.25 * (budget - 128) / 32)
    if policy == "uniform":
        assert np.all(counts == (budget - 128) // 32)
    if policy == "uniform_all":
        assert np.all(counts == budget // 32) and row["pilot_cost"] == 0


def test_fixed_task_weighting_is_not_sample_frequency_weighting() -> None:
    counts = np.tile(np.arange(1, 9), (4, 1))
    cell_means = np.tile(np.arange(8) / 8, (4, 1))
    weights = np.array([4, 4, 2, 2, 1, 1, 1, 1]) / 16
    estimated = estimate(cell_means * counts, counts, weights)
    np.testing.assert_allclose(estimated, cell_means @ weights)
    assert not np.isclose(estimated[0], np.average(cell_means[0], weights=counts[0]))


def test_stage_two_only_estimator_and_independent_phase_prefixes() -> None:
    world = make_world(940001, "concentrated")
    row = run_one(world, "neyman", 6400)
    np.testing.assert_allclose(
        row["estimated_utility"],
        estimate(
            np.array(row["stage2_utility_sums"]),
            np.array(row["stage2_counts"]),
            world.weights,
        ),
    )
    np.testing.assert_array_equal(
        row["stage2_counts"],
        allocate(
            policy_scores(
                "neyman",
                Pilot(np.array(row["pilot_utility"]), np.zeros((4, 8, 4))),
                world.weights,
            ),
            6400 - 128,
            world.cell_tie_order,
        ),
    )
    pilot_u, _ = draw_cell(world, 0, 0, "pilot", 4)
    stage_u, _ = draw_cell(world, 0, 0, "estimation", 32)
    replay_u, _ = draw_cell(world, 0, 0, "estimation", 8)
    np.testing.assert_array_equal(stage_u[:8], replay_u)
    assert not np.array_equal(pilot_u, stage_u[:4])
    assert "pilot" not in inspect.signature(estimate).parameters


def test_zero_event_bounds_are_positive_and_formula_matches() -> None:
    lo, hi = clopper_pearson(np.array([0, 10, 3]), np.array([200, 10, 20]))
    assert lo[0] == 0 < hi[0]
    assert hi[0] == pytest.approx(1 - (0.05 / 64) ** (1 / 200))
    assert hi[1] == 1 > lo[1]
    assert 0 < lo[2] < hi[2] < 1
    with pytest.raises(ValueError):
        clopper_pearson(np.array([0]), np.array([0]))


def test_policy_only_observes_pilot_and_public_prior_inputs() -> None:
    assert {f.name for f in fields(Pilot)} == {"utility", "failure"}
    assert set(inspect.signature(policy_scores).parameters) == {
        "policy",
        "pilot",
        "weights",
    }
    pilot = Pilot(np.zeros((4, 8, 4)), np.zeros((4, 8, 4), dtype=bool))
    np.testing.assert_array_equal(
        policy_scores("event_following", pilot, np.ones(8) / 8), 1
    )
    pilot.failure[2, 5, 0] = True
    scores = policy_scores("event_following", pilot, np.ones(8) / 8)
    assert scores[2, 5] == 1 and scores[2].sum() == 1
    np.testing.assert_array_equal(scores[[0, 1, 3]].sum(axis=1), 8)


def test_reproducible_rows_and_policy_common_pilot() -> None:
    world = make_world(940002, "diffuse")
    one = run_one(world, "uniform", 6400)
    assert one == run_one(world, "uniform", 6400)
    other = run_one(world, "event_following", 6400)
    assert one["pilot_failures"] == other["pilot_failures"]
    if not one["pilot_hit"]:
        assert one["stage2_counts"] == other["stage2_counts"]
        assert one["estimated_utility"] == other["estimated_utility"]


def test_complete_grid_gate_rejects_missing_duplicate_or_cost_mismatch() -> None:
    rows = [
        run_one(make_world(940003, scenario), policy, 6400)
        for scenario in SCENARIOS
        for policy in POLICIES
    ]
    validate_grid(rows, (940003,), (6400,))
    with pytest.raises(ValueError, match="grid"):
        validate_grid(rows[:-1], (940003,), (6400,))
    with pytest.raises(ValueError, match="grid"):
        validate_grid(rows + rows[:1], (940003,), (6400,))
    rows[0]["cost"] -= 1
    with pytest.raises(ValueError, match="cost"):
        validate_grid(rows, (940003,), (6400,))


def test_allocation_remainder_and_invalid_budget() -> None:
    counts = allocate(np.ones((4, 8)), 132, np.arange(32)[::-1])
    assert counts.sum() == 132 and counts.min() > 0
    np.testing.assert_array_equal(counts.sum(axis=1), 33)
    assert counts.ravel()[-1] == 5
    with pytest.raises(ValueError):
        allocate(np.ones((4, 8)), 10, np.arange(32))
    with pytest.raises(ValueError):
        policy_scores(
            "oracle", Pilot(np.zeros((4, 8, 4)), np.zeros((4, 8, 4))), np.ones(8) / 8
        )


def test_invalid_world_estimation_budget_and_tie_order_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_world(941000, "unknown")
    with pytest.raises(ValueError):
        estimate(np.ones((4, 8)), np.zeros((4, 8)), np.ones(8) / 8)
    with pytest.raises(ValueError):
        run_one(make_world(941000, "diffuse"), "uniform", 255)
    with pytest.raises(ValueError):
        allocate(np.ones((4, 8)), 128, np.zeros(32))
