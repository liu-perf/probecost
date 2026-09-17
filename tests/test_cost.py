"""Probe cost, and why the wall clock cannot police it."""

from __future__ import annotations

import json

import pytest

from probecost import cost
from probecost.model import DATA_DIR, CostTable


def test_the_same_probe_spans_two_orders_of_magnitude():
    t = cost.cost_table()
    lo, hi = t["same_probe_cost_range"]
    assert lo < 1.02 and hi > 100.0
    assert t["cost_spread"] > 150.0


def test_the_expensive_run_used_fewer_passes_than_the_cheap_ones():
    """Rules out "the metric set was heavier" as the explanation."""
    t = CostTable.load()
    heavy = t.by_id("ncu_no_filter")
    light = t.by_id("ncu_filter_magma")
    assert heavy.replay_passes < light.replay_passes
    assert heavy.launches_profiled > light.launches_profiled * 50


def test_filtered_runs_are_within_a_few_percent_of_no_probe_at_all():
    rows = {r["id"]: r for r in cost.cost_table()["rows"]}
    for rid in ("ncu_filter_magma", "ncu_filter_cutlass_full_name"):
        assert abs(rows[rid]["cost_vs_baseline"] - 1.0) < 0.025, rid


def test_exactly_one_run_collected_nothing_and_it_reported_a_number():
    fails = cost.silent_failures()
    assert len(fails) == 1
    f = fails[0]
    assert f["distance_from_baseline_pct"] < 0.2
    assert f["report_written"] is True
    assert "No kernels were profiled" in f["only_signal"]


def test_the_failed_run_sits_between_the_two_that_worked():
    """Why a threshold cannot work: it is not an outlier, it is in the middle."""
    rows = {r["id"]: r["mean_ms_per_image"] for r in cost.cost_table()["rows"]}
    lo = rows["ncu_filter_magma"]
    hi = rows["ncu_filter_cutlass_full_name"]
    assert lo < rows["ncu_filter_cutlass_short_name"] < hi


@pytest.mark.parametrize("tol", [1.0, 2.0, 5.0, 10.0])
def test_no_tolerance_separates_collected_from_not_collected(tol):
    """Parametrised deliberately.

    This is not true of one threshold, it is true of every plausible one --
    which is what makes it a detection impossibility rather than a badly
    chosen constant.
    """
    r = cost.indistinguishable_from_baseline(tolerance_pct=tol)
    assert r["both_outcomes_inside"] is True, (tol, r["inside"])


def test_every_probed_cell_is_reported_as_underpowered():
    under = {u["id"] for u in cost.underpowered()}
    assert "none" not in under
    assert len(under) == 4


def test_the_verdict_tracks_the_data_and_is_not_a_constant(tmp_path):
    """Fault injection on the conclusion itself.

    Move the empty run far from the baseline and the verdict has to change --
    otherwise this check asserts a constant instead of measuring the data.
    """
    raw = json.loads((DATA_DIR / "probe_cost.json").read_text(encoding="utf-8"))
    for r in raw["runs"]:
        if r["id"] == "ncu_filter_cutlass_short_name":
            r["mean_ms_per_image"] = 9999.0
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    t = CostTable.load(p)
    r = cost.indistinguishable_from_baseline(t, tolerance_pct=5.0)
    assert r["both_outcomes_inside"] is False
    assert "property of this tolerance" in r["verdict"]
