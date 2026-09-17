"""The findings this repository claims, recomputed from ``data/``.

Nothing here needs a GPU, a profiler, or a network. Producing the data needed
all three; re-checking what was concluded from it needs none.
"""

from __future__ import annotations

from . import cost, population
from .instrument import InstrumentError
from .model import CostTable, LaunchSet, SummaryTable


def probe_cost() -> dict:
    """The same probe, on the same workload, costing 1.00x and 179x."""
    t = cost.cost_table()
    rows = {r["id"]: r for r in t["rows"]}
    return {
        "baseline_ms": t["baseline_ms"],
        "baseline_n": t["baseline_n"],
        "rows": t["rows"],
        "filtered_range": [
            min(r["cost_vs_baseline"] for r in t["rows"]
                if r["probe"] and r["filtered"]),
            max(r["cost_vs_baseline"] for r in t["rows"]
                if r["probe"] and r["filtered"]),
        ],
        "unfiltered": rows["ncu_no_filter"]["cost_vs_baseline"],
        "cost_spread": t["cost_spread"],
        "passes_do_not_explain_it": {
            "unfiltered_passes": rows["ncu_no_filter"]["replay_passes"],
            "filtered_passes": rows["ncu_filter_magma"]["replay_passes"],
            "reading": "the expensive run used fewer replay passes per launch than "
                       "the cheap ones; the cost follows the number of launches "
                       "replayed, not the weight of the metric set",
        },
    }


def silent_failure() -> dict:
    """A run that wrote a report, printed a throughput, and measured nothing."""
    fails = cost.silent_failures()
    ind = cost.indistinguishable_from_baseline()
    return {
        "runs": fails,
        "count": len(fails),
        "closest_to_baseline_pct": min(f["distance_from_baseline_pct"] for f in fails),
        "wall_clock_cannot_detect_it": ind["both_outcomes_inside"],
        "detection_verdict": ind["verdict"],
        "inside_tolerance": ind["inside"],
        "what_the_only_signal_was": fails[0]["only_signal"],
        "and_it_still_wrote_a_report": fails[0]["report_written"],
    }


def instrument_is_repeatable() -> dict:
    """The control: group by launch geometry and the instrument is tight.

    Without this the finding below reads as "Nsight Compute's durations are
    unreliable", which the data contradicts.
    """
    w = population.within_shape()
    return {
        "groups_with_repeats": w["groups_with_repeats"],
        "median_cv_pct": w["median_cv_pct"],
        "groups_under_1pct_cv": w["tight_under_1pct"],
        "worst_group": w["worst"],
        "reading": "grouped by (name, grid, block) the same instrument reproduces "
                   "itself to well under a percent in most groups; it is a precise "
                   "instrument and the imprecision is in the key, not the probe",
    }


def the_name_is_the_wrong_key() -> dict:
    """The same eight launches, grouped by name instead of by shape."""
    a = population.across_shape()
    same = population.same_shape_still_varies()
    return {
        "kernels": a["kernels"],
        "worst": a["worst"],
        "kernels_with_more_than_one_shape": len(a["multi_shape"]),
        "same_geometry_still_varies": same,
        "reading": "eight launches of one kernel name span several times in duration "
                   "because they are several different shapes; and for the GEMM even "
                   "identical grid and block still vary, because K is not in the "
                   "launch geometry -- so there is no key in a profiler summary that "
                   "would make the row a quantity",
    }


def whole_run_spread() -> dict:
    """The other instrument, whole runs, two workloads."""
    return population.name_row_spread()


def the_comparison_that_gets_refused() -> dict:
    """Assemble the comparison a person would make, and record the refusal."""
    # cunn_SoftMaxForwardReg is deliberately absent: its short name matches two
    # different rows in the summary table, so it never reaches this comparison.
    # It is refused one step earlier, and that earlier refusal is its own
    # finding -- see `the_name_matches_two_things`.
    pairs = [
        ("magma_sgemmEx_kernel", "magma_sgemmEx_kernel"),
        ("cutlass_80_simt_sgemm_64x64_8x5_nn_align1", "cutlass_80_simt_sgemm_64x64_8x5"),
        ("softmax_warp_forward", "softmax_warp_forward"),
        ("upsample_bicubic2d_out_frame", "upsample_bicubic2d_out_frame"),
    ]
    out = []
    for ncu_name, nsys_sub in pairs:
        try:
            population.compare_across_instruments(ncu_name, nsys_sub)
        except InstrumentError as exc:
            d = getattr(exc, "detail", None)
            if d is None:
                raise
            out.append({"refused": True, "message": str(exc), **d})
        else:  # pragma: no cover - the function exists to raise
            raise AssertionError(f"{ncu_name}: the comparison was not refused")
    ratios = [r["ratio_if_you_did_it_anyway"] for r in out]
    return {
        "pairs": out,
        "all_refused": all(r["refused"] for r in out),
        "ratio_range_if_you_ignored_the_refusal": [min(ratios), max(ratios)],
        "ratio_spread": max(ratios) / min(ratios),
        "changes_sign": min(ratios) < 1.0 < max(ratios),
        "reading": "if the refusal is ignored the two instruments appear to disagree "
                   "by -22% to +29%, in both directions, which looks like a "
                   "calibration problem and is not one -- it is two different sets "
                   "of launches being averaged",
    }


def the_name_matches_two_things() -> dict:
    """The same naming scheme under-matches and over-matches, in one dataset.

    One short name matched **zero** kernels and silently cost a profiling run.
    Another matches **two** rows whose medians are nearly nine times apart. The
    fix for the first (use the fully mangled name) is the same fix as for the
    second, which is why they belong in one finding.
    """
    amb = population.ambiguous_short_names()
    da = amb["depth_anything_v2"]
    worst = da["collisions"][0] if da["collisions"] else None
    from .model import SummaryTable
    st = SummaryTable.load()
    try:
        st.row("depth_anything_v2", "cunn_SoftMaxForwardReg")
    except Exception as exc:
        refused = str(exc)
    else:  # pragma: no cover
        refused = None
    return {
        "by_workload": {k: {kk: vv for kk, vv in v.items() if kk != "collisions"}
                        for k, v in amb.items()},
        "worst_collision": worst,
        "lookup_refusal": refused,
        "and_the_other_direction": {
            "filter": "cutlass_80_simt_sgemm",
            "matched": 0,
            "where": "data/probe_cost.json: ncu_filter_cutlass_short_name",
        },
        "reading": "the identifier a person copies out of a profiler is not a key: "
                   "in this data it matched nothing once and two different kernels "
                   "another time, and only the fully mangled signature is unique",
    }


def sample_size_limits() -> dict:
    """Every probed cell has six images or fewer. Stated, not footnoted."""
    under = cost.underpowered()
    return {
        "cells": under,
        "worst_claim_on_fewest_samples": max(
            (u for u in under), key=lambda u: u["claims_pct"] / max(u["n"], 1)),
        "reading": "the small ratios in the cost table come from six-image windows "
                   "and a 2.3% difference over six samples is not a difference; the "
                   "179x does not depend on this, and neither does anything about "
                   "the run that collected nothing",
    }


def all_findings() -> dict:
    return {
        "probe_cost": probe_cost(),
        "silent_failure": silent_failure(),
        "instrument_is_repeatable": instrument_is_repeatable(),
        "the_name_is_the_wrong_key": the_name_is_the_wrong_key(),
        "whole_run_spread": whole_run_spread(),
        "the_name_matches_two_things": the_name_matches_two_things(),
        "the_comparison_that_gets_refused": the_comparison_that_gets_refused(),
        "sample_size_limits": sample_size_limits(),
    }


__all__ = ["CostTable", "LaunchSet", "SummaryTable", "all_findings",
           "instrument_is_repeatable", "probe_cost", "sample_size_limits",
           "silent_failure", "the_comparison_that_gets_refused",
           "the_name_is_the_wrong_key", "the_name_matches_two_things",
           "whole_run_spread"]
