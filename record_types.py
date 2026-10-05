"""Concrete numerical aliases and JSON record schemas for the frozen experiment."""

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type SourceHashes = dict[str, str]
type NullableMetric = float | None
type MetricName = Literal[
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
]


class MetricRecord(TypedDict):
    """Per-world metrics or their world means; undefined diagnostics remain null."""

    simple_regret: float
    best_agent_error: float
    kendall_error: float
    utility_rmse: float
    risk_rmse: float
    any_discovery: float
    risky_cell_discovery: NullableMetric
    pilot_hit: NullableMetric
    risk_family_coverage: float
    risky_interval_width: NullableMetric
    zero_event_fraction: float
    zero_event_upper: NullableMetric


class RunRecord(MetricRecord):
    """One world/scenario/policy/budget observation with auditable cell arrays."""

    seed: int
    scenario: str
    policy: str
    budget: int
    cost: int
    pilot_cost: int
    stage2_counts: list[list[int]]
    stage2_utility_sums: list[list[float]]
    stage2_failures: list[list[int]]
    pilot_failures: list[list[int]]
    pilot_utility: list[list[list[float]]]
    task_weights: list[float]
    true_utility: list[float]
    true_cell_risk: list[list[float]]
    estimated_utility: list[float]
    estimated_agent_risk: list[float]
    estimated_cell_risk: list[list[float]]
    risk_lower: list[list[float]]
    risk_upper: list[list[float]]
    selected_agent: int
    true_best_agent: int
    estimated_order: list[int]
    true_order: list[int]


class PolicyMean(MetricRecord):
    """Mean per-world metrics within a fixed scenario, budget and policy."""

    scenario: str
    budget: int
    policy: str
    seeds: int


class BootstrapResult(TypedDict):
    mean_difference: NullableMetric
    low: NullableMetric
    high: NullableMetric
    paired_seeds: int


class ContrastRecord(BootstrapResult):
    scenario: str
    budget: int
    left: str
    right: str
    metric: MetricName


class SummaryRecord(TypedDict):
    scope: str
    interval_method: str
    metric_definitions: dict[str, str]
    policy_means: list[PolicyMean]
    paired_contrasts: list[ContrastRecord]


class ManifestRecord(TypedDict):
    mode: str
    seeds: list[int]
    budgets: list[int]
    policies: list[str]
    scenarios: list[str]
    expected_rows: int
    expected_draws: int
    source_hashes: SourceHashes
    started_utc: str
    python: str
    numpy: str
    scipy: str
    platform: str
    rng: str
    command: list[str]
    primary_approval_flag: bool


class CompletionRecord(TypedDict):
    complete: bool
    rows: int
    draws: int
    finished_utc: str
    source_hashes: SourceHashes
    manifest_sha256: str
    results_sha256: str
    summary_sha256: str
    report_sha256: str
