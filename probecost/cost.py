"""What the probe cost, and what the wall clock can and cannot tell you about it.

Two results live here and they point opposite ways.

The first is that the probe's cost is not a property of the probe. The same
Nsight Compute, on the same workload, costs 1.00x with a name filter and 179x
without one -- and the unfiltered run used *fewer* replay passes per launch
than the filtered ones. The cost is set by how many launches get replayed.

The second is the one worth carrying around. The wall clock cannot tell you
whether the probe collected anything. In the committed data the run whose name
filter matched **zero** kernels came in at 83.49 ms/image against an unprofiled
83.41 -- 0.1% away, and *between* the two runs that did collect data. It wrote a
report file. It printed a throughput. The only signal was one line on stderr,
between two lines that looked entirely normal.

So :func:`cost_table` reports the ratios, and :func:`silent_failures` reports
the runs that produced everything except measurements, and
:func:`indistinguishable_from_baseline` shows why the first cannot detect the
second.
"""

from __future__ import annotations

from .model import CostTable, Run


def _ratio(run: Run, base: Run) -> float:
    return run.mean_ms_per_image / base.mean_ms_per_image


def cost_table(table: CostTable | None = None) -> dict:
    t = table or CostTable.load()
    base = t.baseline
    rows = []
    for r in t.runs:
        rows.append({
            "id": r.id,
            "probe": r.probe,
            "filtered": r.filter is not None,
            "replay_passes": r.replay_passes,
            "launches_profiled": r.launches_profiled,
            "mean_ms_per_image": r.mean_ms_per_image,
            "cost_vs_baseline": _ratio(r, base),
            "n": r.images_in_steady_window,
            "collected": r.collected_anything,
        })
    with_probe = [x for x in rows if x["probe"]]
    ratios = [x["cost_vs_baseline"] for x in with_probe]
    return {
        "baseline": base.id,
        "baseline_ms": base.mean_ms_per_image,
        "baseline_n": base.images_in_steady_window,
        "rows": rows,
        "same_probe_cost_range": [min(ratios), max(ratios)] if ratios else None,
        "cost_spread": max(ratios) / min(ratios) if ratios else None,
        "reading": "the cost is set by how many launches are replayed, not by the "
                   "probe and not by the metric set -- the 179x run used 9 replay "
                   "passes per launch and the 1.00x runs used 21",
    }


def silent_failures(table: CostTable | None = None) -> list:
    """Runs that wrote a report, printed a number, and collected nothing."""
    t = table or CostTable.load()
    base = t.baseline
    out = []
    for r in t.empty_runs:
        out.append({
            "id": r.id,
            "filter": r.filter,
            "mean_ms_per_image": r.mean_ms_per_image,
            "distance_from_baseline_pct": abs(_ratio(r, base) - 1.0) * 100.0,
            "report_written": r.report_written,
            "only_signal": r.stderr_line,
            "note": r.note,
        })
    return out


def indistinguishable_from_baseline(table: CostTable | None = None,
                                    *, tolerance_pct: float = 5.0) -> dict:
    """Which runs the wall clock cannot separate from having no probe at all.

    The point is not that the tolerance is 5%. It is that the run which
    collected nothing and the runs which collected something fall on the *same
    side* of any tolerance you would plausibly pick, so no threshold on this
    number can be used to detect a failed probe.
    """
    t = table or CostTable.load()
    base = t.baseline
    inside = []
    for r in t.runs:
        if r.id == base.id:
            continue
        d = abs(_ratio(r, base) - 1.0) * 100.0
        if d <= tolerance_pct:
            inside.append({"id": r.id, "distance_pct": d,
                           "collected": r.collected_anything,
                           "n": r.images_in_steady_window})
    got = {x["collected"] for x in inside}
    return {
        "tolerance_pct": tolerance_pct,
        "inside": inside,
        "both_outcomes_inside": got == {True, False},
        "verdict": ("no threshold on wall-clock cost can detect a probe that "
                    "collected nothing: runs with and without measurements are "
                    "on the same side of it")
        if got == {True, False} else
        ("in this table the failed run happens to fall outside the tolerance; "
         "that is a property of this tolerance, not a detection method"),
    }


def underpowered(table: CostTable | None = None, *, need: int = 30) -> list:
    """Cells whose steady-state window is too small to resolve their own claim.

    Every probed cell here has six images or fewer. That is the binding limit
    on the small ratios and it is reported next to them rather than in a
    footnote, because a 2.3% difference over six samples is not a difference.
    """
    t = table or CostTable.load()
    return [{"id": r.id, "n": r.images_in_steady_window,
             "claims_pct": abs(_ratio(r, t.baseline) - 1.0) * 100.0}
            for r in t.runs if r.images_in_steady_window < need]


__all__ = ["cost_table", "indistinguishable_from_baseline", "silent_failures",
           "underpowered"]
