import json
from pathlib import Path

import pytest

from experiment import BUDGETS, METRICS, POLICIES, SCENARIOS, make_world, run_one
from record_types import RunRecord
from reporting import CONTRASTS, bootstrap_contrast, summarize
from run_experiment import (
    file_hash,
    main,
    run_grid,
    source_hashes,
    validate_mode_plan,
    verify_output,
)


@pytest.fixture(scope="module")
def rows() -> list[RunRecord]:
    return [
        run_one(make_world(seed, scenario), policy, budget)
        for seed in (941010, 941011)
        for scenario in SCENARIOS
        for policy in POLICIES
        for budget in BUDGETS
    ]


def test_all_comparisons_are_preserved_with_explicit_nulls(
    rows: list[RunRecord],
) -> None:
    summary = summarize(rows, (941010, 941011), BUDGETS)
    assert len(summary["policy_means"]) == 3 * 2 * 4
    assert len(summary["paired_contrasts"]) == 3 * 2 * len(CONTRASTS) * len(METRICS)
    absent = [
        r
        for r in summary["paired_contrasts"]
        if r["metric"] == "pilot_hit" and "uniform_all" in (r["left"], r["right"])
    ]
    assert all(r["mean_difference"] is None for r in absent)
    assert summarize(rows, (941010, 941011), BUDGETS) == summary


def test_pairing_preserved_and_bootstrap_descriptive() -> None:
    result = bootstrap_contrast([1, 3], [0, 2], [[0, 0], [0, 1], [1, 1]])
    assert result == {
        "mean_difference": 1.0,
        "low": 1.0,
        "high": 1.0,
        "paired_seeds": 2,
    }
    assert bootstrap_contrast([None], [1], [[0]])["mean_difference"] is None
    with pytest.raises(ValueError):
        bootstrap_contrast([1, None], [1, 2], [[0, 1]])


def test_no_summary_for_incomplete_grid(rows: list[RunRecord]) -> None:
    with pytest.raises(ValueError, match="grid"):
        summarize(rows[:-1], (941010, 941011), BUDGETS)


def test_runner_immutable_outputs_manifest_and_completion(tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    run_grid(out, (941001,), (6400,), "test")
    manifest = json.loads((out / "manifest.json").read_text())
    completion = json.loads((out / "completion.json").read_text())
    assert manifest["expected_rows"] == 12
    assert completion["rows"] == 12 and completion["complete"]
    assert completion["source_hashes"] == source_hashes()
    assert completion["results_sha256"] == file_hash(out / "results.jsonl")
    assert verify_output(out)["complete"]
    with pytest.raises(FileExistsError):
        run_grid(out, (941001,), (6400,), "test")


def test_output_verifier_rejects_artifact_tampering(tmp_path: Path) -> None:
    out = tmp_path / "test"
    run_grid(out, (941002,), (6400,), "test")
    with (out / "results.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        verify_output(out)


def test_source_changes_during_collection_block_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = source_hashes()
    snapshots = iter([actual, {**actual, "PROTOCOL.md": "changed"}])
    monkeypatch.setattr("run_experiment.source_hashes", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="changed"):
        run_grid(tmp_path / "changed", (941003,), (6400,), "test")
    assert not (tmp_path / "changed" / "summary.json").exists()


def test_cli_requires_primary_approval_and_smoke_has_separate_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        main(["--mode", "primary", "--out", str(tmp_path / "primary")])
    called = []
    monkeypatch.setattr(
        "run_experiment.run_grid", lambda *args, **kwargs: called.append((args, kwargs))
    )
    main(["--mode", "smoke", "--out", str(tmp_path / "smoke")])
    assert called[0][0][1] == (940000, 940001, 940002)
    assert called[0][1] == {"approved_primary": False}
    main(
        ["--mode", "primary", "--approved-primary", "--out", str(tmp_path / "primary")]
    )
    assert called[1][0][1] == tuple(range(970000, 971024))
    assert called[1][1] == {"approved_primary": True}


@pytest.mark.parametrize(
    "mode,seeds,budgets,approval",
    [
        ("unknown", (941000,), BUDGETS, False),
        ("primary", (940000, 940001, 940002), BUDGETS, True),
        ("primary", tuple(range(970000, 971024)), (6400,), True),
        ("primary", tuple(range(970000, 971024)), BUDGETS, False),
        ("smoke", (940000,), BUDGETS, False),
        ("smoke", (940000, 940001, 940002), (6400,), False),
        ("test", (970000,), (6400,), False),
    ],
)
def test_runner_rejects_invalid_mode_plan_before_creating_output(
    tmp_path: Path,
    mode: str,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    approval: bool,
) -> None:
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_grid(out, seeds, budgets, mode, approved_primary=approval)
    assert not out.exists()


def test_full_primary_plan_validation_does_not_collect_primary_data() -> None:
    assert validate_mode_plan(
        "primary",
        tuple(range(970000, 971024)),
        BUDGETS,
        POLICIES,
        SCENARIOS,
        True,
    ) == (24576, 393216000)


def test_verifier_rejects_smoke_subset_relabelled_primary(tmp_path: Path) -> None:
    out = tmp_path / "forged_primary"
    run_grid(out, (941002,), (6400,), "test")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["mode"] = "primary"
    manifest["primary_approval_flag"] = True
    (out / "manifest.json").write_text(json.dumps(manifest))
    completion = json.loads((out / "completion.json").read_text())
    completion["manifest_sha256"] = file_hash(out / "manifest.json")
    (out / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="plan"):
        verify_output(out)


@pytest.mark.parametrize(
    "field,value",
    [
        ("policies", ["uniform"]),
        ("scenarios", ["diffuse"]),
        ("expected_rows", 999),
        ("expected_draws", 999),
        ("mode", "unknown"),
    ],
)
def test_verifier_rejects_invalid_manifest_plan(
    tmp_path: Path, field: str, value: object
) -> None:
    out = tmp_path / "invalid_manifest"
    run_grid(out, (941004,), (6400,), "test")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest[field] = value
    (out / "manifest.json").write_text(json.dumps(manifest))
    completion = json.loads((out / "completion.json").read_text())
    completion["manifest_sha256"] = file_hash(out / "manifest.json")
    (out / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="plan"):
        verify_output(out)


@pytest.mark.parametrize("update_hash", [False, True])
def test_verifier_rejects_changed_report_even_with_updated_hash(
    tmp_path: Path, update_hash: bool
) -> None:
    out = tmp_path / "tampered_report"
    run_grid(out, (941005,), (6400,), "test")
    (out / "REPORT.md").write_text("Misleading replacement report\n")
    if update_hash:
        completion = json.loads((out / "completion.json").read_text())
        completion["report_sha256"] = file_hash(out / "REPORT.md")
        (out / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="report"):
        verify_output(out)
