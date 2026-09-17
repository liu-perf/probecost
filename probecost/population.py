"""What a per-kernel number is an average *of*.

Every profiler summary is one row per kernel *name*. A kernel name is not a
unit of work -- the same name is launched with different grids, different
blocks, and (for a GEMM) different K, which does not appear in the launch
geometry at all. So the row is an average over a mixture, and two rows from two
instruments are averages over two different mixtures.

Three functions, in the order the argument runs:

:func:`within_shape` -- the control. Group the per-launch durations by
(name, grid, block) and look at the spread inside a group. If the instrument
were noisy this is where it would show, and it does not: most groups here are
tight to well under one percent.

:func:`across_shape` -- the same launches grouped by name only. Now the spread
is a factor of several, in a run of eight launches.

:func:`name_row_spread` -- the per-name summary rows from the other instrument,
over the whole run. Over half the names in both workloads span 2x or more
within the name, and the worst spans a thousand.

The conclusion is not that either instrument is wrong. Both are reporting
exactly what they measured. It is that **the kernel name is the wrong key**,
and every table built on it is an average over a set nobody wrote down.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict

from .instrument import InstrumentError
from .model import LaunchSet, SummaryTable


def _stats(xs) -> dict:
    xs = sorted(xs)
    mean = statistics.mean(xs)
    return {
        "n": len(xs), "min": xs[0], "max": xs[-1],
        "median": statistics.median(xs), "mean": mean,
        "spread": xs[-1] / xs[0] if xs[0] else float("inf"),
        "cv_pct": (statistics.pstdev(xs) / mean * 100.0) if mean else float("inf"),
    }


def within_shape(ls: LaunchSet | None = None) -> dict:
    """Group by (name, grid, block). The instrument's own repeatability."""
    ls = ls or LaunchSet.load()
    groups = defaultdict(list)
    for x in ls.launches:
        groups[x.shape].append(x.duration_ns)
    out = []
    for shape, xs in groups.items():
        if len(xs) < 2:
            continue
        s = _stats(xs)
        out.append({"kernel": shape[0], "grid": shape[1], "block": shape[2], **s})
    out.sort(key=lambda r: -r["spread"])
    return {
        "groups": out,
        "groups_with_repeats": len(out),
        "median_cv_pct": statistics.median(r["cv_pct"] for r in out) if out else None,
        "worst": out[0] if out else None,
        "tight_under_1pct": sum(1 for r in out if r["cv_pct"] < 1.0),
    }


def across_shape(ls: LaunchSet | None = None) -> dict:
    """Group by name only -- what a per-name row averages over."""
    ls = ls or LaunchSet.load()
    groups = defaultdict(list)
    for x in ls.launches:
        groups[x.kernel].append(x.duration_ns)
    shapes = defaultdict(set)
    for x in ls.launches:
        shapes[x.kernel].add((x.grid, x.block))
    out = []
    for name, xs in groups.items():
        out.append({"kernel": name, "distinct_shapes": len(shapes[name]), **_stats(xs)})
    out.sort(key=lambda r: -r["spread"])
    return {
        "kernels": out,
        "worst": out[0],
        "multi_shape": [r for r in out if r["distinct_shapes"] > 1],
    }


def same_shape_still_varies(ls: LaunchSet | None = None,
                            *, factor: float = 2.0) -> list:
    """Groups where the launch geometry is identical and the duration is not.

    The one that matters is the GEMM. A grid and a block do not carry K, so two
    launches with the same geometry can be doing several times the arithmetic.
    Which means conditioning on the geometry is *also* not enough, and there is
    no key available in a profiler summary that would be.
    """
    w = within_shape(ls)
    return [g for g in w["groups"] if g["spread"] >= factor]


def name_row_spread(st: SummaryTable | None = None, *, min_instances: int = 10,
                    factor: float = 2.0) -> dict:
    """The other instrument, whole run, per-name rows."""
    st = st or SummaryTable.load()
    out = {}
    for wl in st.workloads:
        rows = [r for r in st.rows(wl) if r.instances >= min_instances and r.min_ns > 0]
        wide = [r for r in rows if r.spread >= factor]
        rows_sorted = sorted(rows, key=lambda r: -r.spread)
        out[wl] = {
            "names_total": len(st.rows(wl)),
            "names_considered": len(rows),
            "names_spanning_factor": len(wide),
            "factor": factor,
            "median_cv_pct": statistics.median(r.cv_pct for r in rows) if rows else None,
            "worst": {"name": rows_sorted[0].name[:60], "spread": rows_sorted[0].spread,
                      "cv_pct": rows_sorted[0].cv_pct,
                      "instances": rows_sorted[0].instances} if rows_sorted else None,
            "total_launches": sum(r.instances for r in st.rows(wl)),
        }
    return out


#: Anonymous-namespace spellings. These are *not* template arguments, and the
#: first version of `_short_name` split on their `<` -- which collapsed seven
#: unrelated kernels to the token `native` and produced a headline collision of
#: 1273x that does not exist. A crude parser inventing a finding is worse than
#: no parser, so the tokens are named here and stripped before any splitting.
_ANON = (r"<unnamed>::", r"\(anonymous namespace\)::")


def _short_name(mangled: str) -> str:
    """The identifier a person copies out of a profiler's collapsed display.

    Operationally: strip anonymous-namespace markers, take everything before
    the first remaining ``<`` or ``(``, and keep the last ``::``-separated
    identifier. That is what ends up in an ``ncu -k`` filter and in a
    spreadsheet cell, and it is what the committed data shows both
    over-matching and under-matching.
    """
    head = mangled
    for pat in _ANON:
        head = re.sub(pat, "", head)
    head = re.split(r"[<(]", head, maxsplit=1)[0]
    parts = [p for p in re.split(r"::|\s+", head) if p and p not in
             ("void", "const", "unsigned", "long", "int", "float", "bool", "char")]
    return parts[-1] if parts else head.strip()


def ambiguous_short_names(st: SummaryTable | None = None) -> dict:
    """Short names that identify more than one row, and by how much they differ.

    The reason this is a separate finding rather than a lookup inconvenience:
    the *same* naming scheme fails in both directions in the committed data.
    A short name matched **zero** kernels and cost a whole profiling run
    (``data/probe_cost.json``), and a short name here matches **two** kernels
    whose medians are nearly nine times apart.
    """
    st = st or SummaryTable.load()
    out = {}
    for wl in st.workloads:
        groups = defaultdict(list)
        for r in st.rows(wl):
            groups[_short_name(r.name)].append(r)
        collisions = []
        for short, rows in groups.items():
            if len(rows) < 2:
                continue
            meds = [r.med_ns for r in rows]
            collisions.append({
                "short_name": short,
                "rows": len(rows),
                "instances": [r.instances for r in rows],
                "medians_ns": meds,
                "median_ratio": max(meds) / min(meds) if min(meds) else float("inf"),
                "full_names": [r.name[:120] for r in rows],
            })
        collisions.sort(key=lambda c: -c["median_ratio"])
        out[wl] = {
            "names_total": len(st.rows(wl)),
            "distinct_short_names": len(groups),
            "colliding_short_names": len(collisions),
            "collisions": collisions,
            "worst_ratio": collisions[0]["median_ratio"] if collisions else None,
        }
    return out


def compare_across_instruments(kernel_ncu: str, kernel_nsys_contains: str,
                               workload: str = "depth_anything_v2",
                               ls: LaunchSet | None = None,
                               st: SummaryTable | None = None):
    """Put one instrument's median beside the other's, and refuse the subtraction.

    This function exists to be *refused*. It assembles exactly the comparison a
    person would make -- same kernel, same machine, same workload, one number
    from each profiler -- and then raises, because the two numbers summarise
    different sets of launches. The numbers are returned on the exception's
    ``detail`` so they can be printed next to the refusal rather than hidden by
    it.
    """
    ls = ls or LaunchSet.load()
    st = st or SummaryTable.load()
    mine = [x.duration_ns for x in ls.by_kernel(kernel_ncu)]
    if not mine:
        raise InstrumentError(f"no launches for {kernel_ncu!r}")
    row = st.row(workload, kernel_nsys_contains)
    detail = {
        "kernel": kernel_ncu,
        "a": {"instrument": ls.instrument["name"],
              "sampling_rule": ls.instrument["sampling_rule"],
              "n": len(mine), "median_ns": statistics.median(mine),
              "spread": max(mine) / min(mine)},
        "b": {"instrument": st.instrument["name"],
              "sampling_rule": st.instrument["sampling_rule"],
              "n": row.instances, "median_ns": row.med_ns,
              "spread": row.spread},
    }
    detail["ratio_if_you_did_it_anyway"] = (
        detail["a"]["median_ns"] / detail["b"]["median_ns"])
    detail["population_size_ratio"] = row.instances / len(mine)
    err = InstrumentError(
        f"{kernel_ncu}: {ls.instrument['name']} took the first {len(mine)} matching "
        f"launches and {st.instrument['name']} took all {row.instances}; those are "
        f"different sets of launches spanning {detail['a']['spread']:.1f}x and "
        f"{detail['b']['spread']:.1f}x internally, so their medians are not two "
        f"estimates of one quantity"
    )
    err.detail = detail
    raise err


__all__ = ["across_shape", "ambiguous_short_names", "compare_across_instruments",
           "name_row_spread", "same_shape_still_varies", "within_shape"]
