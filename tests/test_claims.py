"""Every number this repository publishes, recomputed from ``data/``.

The convention is inherited from the sibling projects: a claim that lives only
in prose cannot fail, so it will eventually be wrong. Each assertion here
corresponds to a sentence in the README or ``docs/``, and the negative ones --
the claims this repository declines to make -- are asserted too, because a
boundary that is not tested gets quietly crossed in an edit.
"""

from __future__ import annotations

import pytest

from probecost import analysis


def test_the_probe_cost_spans_two_orders_of_magnitude_with_one_probe():
    p = analysis.probe_cost()
    assert p["baseline_ms"] == pytest.approx(83.41)
    assert p["baseline_n"] == 760
    lo, hi = p["filtered_range"]
    assert lo == pytest.approx(0.977, abs=0.001)
    assert hi == pytest.approx(1.009, abs=0.001)
    assert p["unfiltered"] == pytest.approx(179.5, abs=0.1)
    assert p["cost_spread"] > 150.0


def test_the_metric_set_is_not_the_explanation():
    """The expensive run used fewer replay passes per launch than the cheap ones."""
    d = analysis.probe_cost()["passes_do_not_explain_it"]
    assert d["unfiltered_passes"] == 9
    assert d["filtered_passes"] == 21
    assert d["unfiltered_passes"] < d["filtered_passes"]


def test_the_run_that_collected_nothing_is_undetectable_from_the_timings():
    s = analysis.silent_failure()
    assert s["count"] == 1
    assert s["closest_to_baseline_pct"] < 0.1
    assert s["wall_clock_cannot_detect_it"] is True
    assert s["and_it_still_wrote_a_report"] is True
    assert "No kernels were profiled" in s["what_the_only_signal_was"]


def test_the_instrument_itself_is_precise_and_the_repository_says_so():
    """The control. This is the claim that keeps the rest honest."""
    r = analysis.instrument_is_repeatable()
    assert r["median_cv_pct"] < 0.5
    assert r["groups_under_1pct_cv"] >= 7
    assert "precise instrument" in r["reading"]


def test_the_repository_does_not_claim_the_profiler_is_wrong():
    """The negative claim, tested.

    Both instruments report exactly what they measured. If a future edit turns
    this into "Nsight Compute's durations are unreliable", the control above
    contradicts it and this fails.
    """
    r = analysis.instrument_is_repeatable()
    assert r["median_cv_pct"] < 1.0
    k = analysis.the_name_is_the_wrong_key()
    assert "the key, not the probe" in r["reading"] or "wrong key" in k["reading"]


def test_the_name_is_the_wrong_key_and_the_geometry_is_not_enough_either():
    k = analysis.the_name_is_the_wrong_key()
    assert k["worst"]["kernel"] == "magma_sgemmEx_kernel"
    assert k["worst"]["spread"] == pytest.approx(5.47, abs=0.01)
    assert k["worst"]["distinct_shapes"] == 4
    same = k["same_geometry_still_varies"]
    assert len(same) == 1
    assert same[0]["spread"] == pytest.approx(3.47, abs=0.01)


def test_seventeen_cutlass_kernels_share_one_short_name():
    n = analysis.the_name_matches_two_things()
    w = n["worst_collision"]
    assert w["short_name"] == "Kernel2"
    assert w["rows"] == 17
    assert w["median_ratio"] == pytest.approx(127.1, abs=0.2)
    # and the other direction, in the same dataset
    assert n["and_the_other_direction"]["matched"] == 0
    assert n["lookup_refusal"] and "matches 2 rows" in n["lookup_refusal"]


def test_the_collision_counts_in_both_workloads():
    b = analysis.the_name_matches_two_things()["by_workload"]
    assert b["depth_anything_v2"]["colliding_short_names"] == 5
    assert b["depth_anything_v2"]["distinct_short_names"] == 22
    assert b["flownet2"]["colliding_short_names"] == 11
    assert b["flownet2"]["distinct_short_names"] == 58


def test_the_whole_run_spread_replicates_across_two_workloads():
    w = analysis.whole_run_spread()
    da, of = w["depth_anything_v2"], w["flownet2"]
    assert (da["names_spanning_factor"], da["names_considered"]) == (29, 41)
    assert (of["names_spanning_factor"], of["names_considered"]) == (54, 94)
    assert da["median_cv_pct"] == pytest.approx(35.8, abs=0.1)
    assert of["median_cv_pct"] == pytest.approx(33.6, abs=0.1)
    assert da["total_launches"] == 14789
    assert of["total_launches"] == 54380
    # The two workloads share a node and a harness and nothing else, so a
    # median CV that lands within a couple of points in both is the point.
    assert abs(da["median_cv_pct"] - of["median_cv_pct"]) < 3.0


def test_every_cross_instrument_comparison_is_refused():
    c = analysis.the_comparison_that_gets_refused()
    assert len(c["pairs"]) == 4
    assert c["all_refused"] is True
    lo, hi = c["ratio_range_if_you_ignored_the_refusal"]
    assert lo == pytest.approx(0.775, abs=0.002)
    assert hi == pytest.approx(1.250, abs=0.002)
    assert c["changes_sign"] is True


def test_the_apparent_disagreement_is_not_claimed_to_be_a_calibration_error():
    """The negative claim that matters most here.

    Ignoring the refusal makes the two profilers look like they disagree by
    -22% to +29%. That reading is available and wrong, and the repository says
    so rather than reporting the ratios as a finding.
    """
    c = analysis.the_comparison_that_gets_refused()
    assert "is not one" in c["reading"]
    assert "different sets" in c["reading"]


def test_the_sample_size_limit_is_attached_not_footnoted():
    s = analysis.sample_size_limits()
    assert len(s["cells"]) == 4
    assert all(c["n"] <= 6 for c in s["cells"])
    assert "not a difference" in s["reading"]
    assert "does not depend on this" in s["reading"]


def test_all_findings_is_complete_and_serialisable():
    import json

    a = analysis.all_findings()
    assert set(a) == {
        "probe_cost", "silent_failure", "instrument_is_repeatable",
        "the_name_is_the_wrong_key", "whole_run_spread",
        "the_name_matches_two_things", "the_comparison_that_gets_refused",
        "sample_size_limits",
    }
    json.dumps(a, default=str)
