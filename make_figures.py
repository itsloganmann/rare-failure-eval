import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "uniform_all": "#929DA6",
    "uniform": "#8B9BAC",
    "neyman": "#12675F",
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    policies = ["uniform_all", "neyman"]
    scenarios = [
        ("concentrated", "Concentrated rare loss"),
        ("diffuse", "Diffuse rare loss"),
    ]
    fig = plt.figure(figsize=(10, 11.25))
    fig.text(
        0.065,
        0.947,
        "Selection accuracy can miss\nthe cost of a wrong choice",
        size=27,
        weight="semibold",
        va="top",
        linespacing=1.12,
    )
    fig.text(
        0.065,
        0.835,
        f"{n:,} simulated worlds · {budget:,} draws per policy per world",
        size=13,
        color="#576675",
    )
    for j, metric in enumerate(["best_agent_error", "simple_regret"]):
        ax = fig.add_axes((0.26, 0.485 if j == 0 else 0.17, 0.66, 0.20))
        values = [
            lookup[scenario, budget, p][metric] * (n if j == 0 else 1)
            for scenario, _ in scenarios
            for p in policies
        ]
        limit = max(max(values) * 1.24, 1 if j == 0 else 0.001)
        for i, (scenario, _) in enumerate(scenarios):
            for k, policy in enumerate(policies):
                y = 2.9 - i * 2 - k * 0.55
                value = lookup[scenario, budget, policy][metric] * (n if j == 0 else 1)
                ax.barh(
                    y,
                    value,
                    color=COLORS[policy],
                    height=0.27,
                    label=LABELS[policy] if i == 0 else None,
                )
                ax.text(
                    value + limit * 0.025,
                    y,
                    f"{round(value):,}" if j == 0 else f"{value:.4f}",
                    size=16,
                    va="center",
                    weight="medium",
                )
        ax.set_xlim(0, limit)
        ax.set_ylim(-0.15, 3.5)
        ax.set_yticks([2.625, 0.625], ["Concentrated\nrisk", "Diffuse\nrisk"], size=14)
        ax.tick_params(axis="x", labelsize=12)
        ax.locator_params(axis="x", nbins=4)
        ax.set_title(
            f"Wrong selections out of {n:,}"
            if j == 0
            else "Mean utility lost per selection",
            loc="left",
            pad=19,
            size=17,
            weight="semibold",
        )
        style_axis(ax)
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.056, 0.806),
                frameon=False,
                ncol=2,
                fontsize=13,
                handlelength=1.2,
            )
    fig.text(
        0.065,
        0.048,
        "Lower is better. Utility lost = best agent's expected utility minus the selected agent's.\n"
        "Synthetic results, not a real-agent benchmark. Full data and uncertainty in the repo.",
        size=10.5,
        color="#576675",
        linespacing=1.5,
    )
    fig.savefig(out / "headline.png", dpi=160)
    fig.savefig(out / "headline.svg")
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
