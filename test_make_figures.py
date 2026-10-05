import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from make_figures import build


def test_nonzero_control_is_visible_and_headline_text_is_separated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "scenario": scenario,
            "budget": budget,
            "policy": policy,
            "seeds": 100,
            "simple_regret": 0.04,
            "best_agent_error": 0.1,
        }
        for scenario in ("no_rare_loss", "concentrated", "diffuse")
        for budget in (6400, 25600)
        for policy in ("uniform_all", "uniform", "neyman", "event_following")
    ]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"policy_means": rows}))
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: figures.append(figure))
    try:
        build(summary, tmp_path / "figures")
        headline, complete = [f for f in figures if hasattr(f, "axes")]
        headline.canvas.draw()
        renderer = headline.canvas.get_renderer()
        brand, title = [t.get_window_extent(renderer) for t in headline.texts[:2]]
        assert not brand.overlaps(title)
        footer = headline.texts[-1].get_window_extent(renderer)
        assert all(
            not footer.overlaps(ax.xaxis.label.get_window_extent(renderer))
            for ax in headline.axes
        )
        assert complete.axes[0].get_ylim()[1] > 0.04
        assert complete.axes[1].get_ylim()[1] > 10
        assert not any(
            "All observed values are zero" in t.get_text()
            for ax in complete.axes
            for t in ax.texts
        )
    finally:
        for fig in figures:
            close(fig)
