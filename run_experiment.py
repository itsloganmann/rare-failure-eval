"""Run the frozen grid in a new directory; primary execution is operator gated."""

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import cast

import numpy as np
import scipy

from experiment import (
    BUDGETS,
    POLICIES,
    PRIMARY_SEEDS,
    SCENARIOS,
    SMOKE_SEEDS,
    make_world,
    run_one,
    validate_grid,
)
from record_types import (
    CompletionRecord,
    ManifestRecord,
    RunRecord,
    SourceHashes,
    SummaryRecord,
)
from reporting import markdown_report, summarize

ROOT = Path(__file__).resolve().parent
FROZEN_FILES = (
    "PROTOCOL.md",
    "experiment.py",
    "reporting.py",
    "run_experiment.py",
    "record_types.py",
    "test_experiment.py",
    "test_reporting.py",
    "independent_audit.py",
    "test_independent_audit.py",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes() -> SourceHashes:
    return {name: file_hash(ROOT / name) for name in FROZEN_FILES}


def write_json(
    path: Path, value: ManifestRecord | SummaryRecord | CompletionRecord
) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_mode_plan(
    mode: str,
    seeds: Sequence[int],
    budgets: Sequence[int],
    policies: Sequence[str],
    scenarios: Sequence[str],
    approved_primary: bool,
) -> tuple[int, int]:
    """Bind the mode to its frozen plan, not to a self-reported manifest subset."""
    if (
        mode not in ("primary", "smoke", "test")
        or type(approved_primary) is not bool
        or tuple(policies) != POLICIES
        or tuple(scenarios) != SCENARIOS
    ):
        raise ValueError("invalid mode, approval or policy/scenario plan")
    if mode in ("primary", "smoke"):
        expected_seeds = PRIMARY_SEEDS if mode == "primary" else SMOKE_SEEDS
        if tuple(seeds) != expected_seeds or tuple(budgets) != BUDGETS:
            raise ValueError(f"{mode} plan must exactly match frozen seeds and budgets")
    elif (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or set(seeds).intersection(PRIMARY_SEEDS)
        or not budgets
        or len(set(budgets)) != len(budgets)
        or any(
            type(budget) is not int or budget < 256 or budget % 4 for budget in budgets
        )
    ):
        raise ValueError("test plan needs unique nonprimary seeds and valid budgets")
    if approved_primary != (mode == "primary"):
        raise ValueError(
            "primary plan requires explicit operator approval; other modes must not claim it"
        )
    return len(seeds) * len(budgets) * len(POLICIES) * len(SCENARIOS), len(seeds) * sum(
        budgets
    ) * len(POLICIES) * len(SCENARIOS)


def run_grid(
    out: Path,
    seeds: Sequence[int],
    budgets: Sequence[int],
    mode: str,
    *,
    approved_primary: bool = False,
) -> Path:
    """Archive manifest before collection, then only summarize a complete stable grid."""
    expected_rows, expected_draws = validate_mode_plan(
        mode, seeds, budgets, POLICIES, SCENARIOS, approved_primary
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    manifest: ManifestRecord = {
        "mode": mode,
        "seeds": list(seeds),
        "budgets": list(budgets),
        "policies": list(POLICIES),
        "scenarios": list(SCENARIOS),
        "expected_rows": expected_rows,
        "expected_draws": expected_draws,
        "source_hashes": hashes,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.system() + " " + platform.machine(),
        "rng": "NumPy PCG64 with keyed SeedSequence",
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "primary_approval_flag": approved_primary,
    }
    write_json(out / "manifest.json", manifest)
    rows: list[RunRecord] = []
    with (out / "results.jsonl").open("x", encoding="utf-8") as stream:
        for seed, scenario in product(seeds, SCENARIOS):
            world = make_world(seed, scenario)
            for policy, budget in product(POLICIES, budgets):
                row = run_one(world, policy, budget)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                rows.append(row)
            stream.flush()
    validate_grid(rows, seeds, budgets)
    if source_hashes() != hashes:
        raise RuntimeError("source/protocol changed during collection; summary blocked")
    summary = summarize(rows, seeds, budgets)
    write_json(out / "summary.json", summary)
    with (out / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown_report(summary, mode))
    write_json(
        out / "completion.json",
        {
            "complete": True,
            "rows": len(rows),
            "draws": sum(r["cost"] for r in rows),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "source_hashes": hashes,
            "manifest_sha256": file_hash(out / "manifest.json"),
            "results_sha256": file_hash(out / "results.jsonl"),
            "summary_sha256": file_hash(out / "summary.json"),
            "report_sha256": file_hash(out / "REPORT.md"),
        },
    )
    return out


def verify_output(out: Path) -> CompletionRecord:
    """Read-only on-disk completion, frozen-source, artifact-hash and grid audit."""
    out = Path(out)
    manifest = cast(ManifestRecord, json.loads((out / "manifest.json").read_text()))
    complete = cast(CompletionRecord, json.loads((out / "completion.json").read_text()))
    expected_rows, expected_draws = validate_mode_plan(
        manifest["mode"],
        manifest["seeds"],
        manifest["budgets"],
        manifest["policies"],
        manifest["scenarios"],
        manifest["primary_approval_flag"],
    )
    if (
        manifest["expected_rows"] != expected_rows
        or manifest["expected_draws"] != expected_draws
    ):
        raise ValueError("manifest totals do not match the validated mode plan")
    if (
        not complete["complete"]
        or manifest["source_hashes"] != source_hashes()
        or complete["source_hashes"] != manifest["source_hashes"]
    ):
        raise ValueError("incomplete run or frozen source hash mismatch")
    for name, filename in (
        ("manifest", "manifest.json"),
        ("results", "results.jsonl"),
        ("summary", "summary.json"),
        ("report", "REPORT.md"),
    ):
        if file_hash(out / filename) != complete[f"{name}_sha256"]:
            raise ValueError(f"{name} artifact hash mismatch")
    rows = [
        cast(RunRecord, json.loads(line))
        for line in (out / "results.jsonl").read_text().splitlines()
    ]
    validate_grid(rows, manifest["seeds"], manifest["budgets"])
    if (
        len(rows) != complete["rows"]
        or len(rows) != manifest["expected_rows"]
        or sum(r["cost"] for r in rows) != complete["draws"]
        or complete["draws"] != manifest["expected_draws"]
    ):
        raise ValueError("completion totals mismatch")
    replayed_summary = summarize(rows, manifest["seeds"], manifest["budgets"])
    if replayed_summary != json.loads((out / "summary.json").read_text()):
        raise ValueError("summary does not replay from raw records")
    if (out / "REPORT.md").read_text() != markdown_report(
        replayed_summary, manifest["mode"]
    ):
        raise ValueError("report does not replay from verified summary and mode")
    return complete


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "primary"), default="smoke")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--approved-primary",
        action="store_true",
        help="explicit operator authorization, not evidence of scientific review",
    )
    parser.add_argument(
        "--verify", action="store_true", help="read-only audit of completed output"
    )
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_output(args.out), indent=2, sort_keys=True))
        return
    if args.mode == "primary" and not args.approved_primary:
        parser.error("primary execution needs completed review and --approved-primary")
    seeds = PRIMARY_SEEDS if args.mode == "primary" else SMOKE_SEEDS
    run_grid(
        args.out, seeds, BUDGETS, args.mode, approved_primary=args.approved_primary
    )


if __name__ == "__main__":
    main()
