import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "uniform_all": "#34475E",
    "uniform": "#8B9BAC",
    "neyman": "#008A81",
    "event_following": "#BF9270",
}
LABELS = {
    "uniform_all": "Uniform, full budget",
    "uniform": "Uniform, pilot matched",
    "neyman": "Variance allocation",
    "event_following": "Follow pilot failures",
}


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD2D5")
    ax.tick_params(axis="both", which="both", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E3E7E8", linewidth=0.8)


def build(summary_path: Path, out: Path) -> None:
    summary = json.loads(summary_path.read_text())
    rows = summary["policy_means"]
    lookup = {(r["scenario"], r["budget"], r["policy"]): r for r in rows}
    n = rows[0]["seeds"]
    budgets = sorted({r["budget"] for r in rows})
    budget = max(budgets)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": "#243342",
            "text.color": "#243342",
            "xtick.color": "#576675",
            "ytick.color": "#243342",
            "figure.facecolor": "#FAFAF7",
            "axes.facecolor": "#FAFAF7",
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
    fig.subplots_adjust(
        left=0.24, right=0.94, top=0.68, bottom=0.19, hspace=0.9, wspace=0.32
    )
    fig.text(0.055, 0.955, "RARE FAILURE EVAL", size=12, weight="bold", color="#008A81")
    fig.text(
        0.055,
        0.90,
        "Picking the wrong agent is\nonly half the story.",
        size=29,
        weight="bold",
        linespacing=1.15,
        va="top",
    )
    fig.text(
        0.055,
        0.755,
        f"{n:,} fresh simulated worlds  /  {budget:,} draws per policy per world",
        size=13,
        color="#576675",
    )
    policies = ["uniform_all", "neyman"]
    scenarios = [
        ("concentrated", "Concentrated rare loss"),
        ("diffuse", "Diffuse rare loss"),
    ]
    for i, (scenario, label) in enumerate(scenarios):
        for j, metric in enumerate(["best_agent_error", "simple_regret"]):
            ax = axs[i, j]
            raw = [lookup[scenario, budget, p][metric] for p in policies]
            values = [v * n for v in raw] if j == 0 else raw
            ax.barh([1, 0], values, color=[COLORS[p] for p in policies], height=0.48)
            ax.set_yticks([1, 0], [LABELS[p] for p in policies] if j == 0 else ["", ""])
            ax.set_xlim(0, max(values) * 1.3 if max(values) else 1)
            for y, value in zip([1, 0], values, strict=True):
                text = f"{round(value):,} / {n:,}" if j == 0 else f"{value:.4f}"
                ax.text(
                    value + ax.get_xlim()[1] * 0.025,
                    y,
                    text,
                    va="center",
                    size=12,
                    weight="bold",
                )
            ax.set_xlabel(
                "Wrong-agent selections" if j == 0 else "Mean regret (utility lost)",
                size=11,
            )
            if j == 0:
                ax.set_title(label, loc="left", pad=18, weight="bold", size=14)
            style_axis(ax)
    fig.text(
        0.055,
        0.05,
        "Lower is better in both columns. Point estimates from a fixed synthetic generator.\n"
        "All 4 policies, both budgets, paired intervals and limitations are in the report.",
        size=11,
        color="#576675",
        linespacing=1.5,
    )
    fig.savefig(out / "headline.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "headline.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    all_policies = list(LABELS)
    for i, (scenario, title) in enumerate(
        [("no_rare_loss", "No rare loss"), *scenarios]
    ):
        for j, metric in enumerate(["simple_regret", "best_agent_error"]):
            ax = axs[i, j]
            x = np.arange(len(budgets))
            for k, policy in enumerate(all_policies):
                vals = [lookup[scenario, b, policy][metric] for b in budgets]
                if j:
                    vals = [v * 100 for v in vals]
                ax.bar(
                    x + (k - 1.5) * 0.19,
                    vals,
                    width=0.18,
                    color=COLORS[policy],
                    label=LABELS[policy],
                )
            ax.set_xticks(x, [f"{b:,}" for b in budgets])
            ax.set_xlabel("Simulated draws per policy per world")
            ax.set_ylabel(
                "Mean regret (utility lost)" if j == 0 else "Wrong-agent selections (%)"
            )
            ax.set_title(title, loc="left", weight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E3E7E8")
            if all(
                lookup[scenario, b, p][metric] == 0
                for b in budgets
                for p in all_policies
            ):
                ax.set_ylim(0, 0.01 if j == 0 else 1)
                ax.text(
                    0.5,
                    0.5,
                    "All observed values are zero",
                    transform=ax.transAxes,
                    ha="center",
                    color="#576675",
                )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Complete synthetic comparison: {n:,} worlds per condition\n"
        "Every policy, scenario and budget; lower is better",
        size=19,
        weight="bold",
    )
    fig.savefig(out / "all_conditions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.summary, args.out)


if __name__ == "__main__":
    main()
