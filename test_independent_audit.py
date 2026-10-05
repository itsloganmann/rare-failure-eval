from copy import deepcopy

import pytest

from experiment import POLICIES, SCENARIOS, make_world, run_one
from independent_audit import audit_rows
from record_types import RunRecord


@pytest.fixture(scope="module")
def records() -> list[RunRecord]:
    return [
        run_one(make_world(941200, scenario), policy, 6400)
        for scenario in SCENARIOS
        for policy in POLICIES
    ]


def test_independent_reconstruction(records: list[RunRecord]) -> None:
    audit = audit_rows(iter(records), (941200,), (6400,))
    assert audit["rows"] == 12
    assert audit["simulated_draws"] == 12 * 6400
    assert audit["max_primary_metric_difference"] < 1e-12


@pytest.mark.parametrize(
    "field,value",
    [
        ("simple_regret", 123),
        ("best_agent_error", 7),
        ("cost", 7),
        ("selected_agent", -1),
        ("true_best_agent", -1),
        ("pilot_cost", 9),
    ],
)
def test_corrupted_metrics_are_rejected(
    records: list[RunRecord], field: str, value: object
) -> None:
    altered = deepcopy(records)
    altered[0][field] = value
    with pytest.raises(ValueError):
        audit_rows(altered, (941200,), (6400,))


def test_pooled_instead_of_fixed_weight_estimate_rejected(
    records: list[RunRecord],
) -> None:
    altered = deepcopy(records)
    row = altered[2]
    row["estimated_utility"] = [
        sum(s) / sum(n)
        for s, n in zip(row["stage2_utility_sums"], row["stage2_counts"], strict=True)
    ]
    with pytest.raises(AssertionError):
        audit_rows(altered, (941200,), (6400,))


def test_missing_or_duplicate_world_rejected(records: list[RunRecord]) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        audit_rows(records[:-1], (941200,), (6400,))
    with pytest.raises(ValueError, match="duplicate"):
        audit_rows([*records, records[0]], (941200,), (6400,))


def test_nonfinite_cell_rejected(records: list[RunRecord]) -> None:
    altered = deepcopy(records)
    altered[0]["stage2_utility_sums"][0][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit_rows(altered, (941200,), (6400,))


def test_inconsistent_tied_selection_rejected(records: list[RunRecord]) -> None:
    altered = deepcopy(records)
    row = altered[0]
    row["stage2_utility_sums"] = [[n * 0.5 for n in ns] for ns in row["stage2_counts"]]
    row["estimated_utility"] = [0.5] * 4
    row["selected_agent"] = (row["estimated_order"][0] + 1) % 4
    with pytest.raises(ValueError, match="selection and ranking"):
        audit_rows(altered, (941200,), (6400,))


def test_invalid_ranking_permutation_rejected(records: list[RunRecord]) -> None:
    altered = deepcopy(records)
    altered[0]["estimated_order"] = [0, 0, 1, 2]
    with pytest.raises(ValueError, match="ranking permutation"):
        audit_rows(altered, (941200,), (6400,))


def test_consistent_but_wrong_tie_order_rejected(records: list[RunRecord]) -> None:
    altered = deepcopy(records)
    row = altered[0]
    row["stage2_utility_sums"] = [[n * 0.5 for n in ns] for ns in row["stage2_counts"]]
    row["estimated_utility"] = [0.5] * 4
    row["estimated_order"] = [0, 1, 2, 3]
    row["selected_agent"] = 0
    row["simple_regret"] = max(row["true_utility"]) - row["true_utility"][0]
    row["best_agent_error"] = int(0 != row["true_best_agent"])
    with pytest.raises(ValueError, match="tie order"):
        audit_rows(altered, (941200,), (6400,))
