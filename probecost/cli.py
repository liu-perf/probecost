"""Command line. Output format is shared with the sibling tools in this series::

    {location}: [{status}] {message}

``--json`` prints the underlying structure instead, and ``--fail-on`` decides
what makes the exit code non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import analysis, cost, population
from .instrument import RULE_NOTES, SAMPLING_RULES, InstrumentError
from .model import CostTable, LaunchSet, SummaryTable

ERROR, WARN, INFO = "error", "warn", "info"
_SEVERITY = {INFO: 0, WARN: 1, ERROR: 2}


def _emit(findings, args) -> int:
    if args.json:
        print(json.dumps(findings, indent=2, ensure_ascii=False, default=str))
    else:
        for f in findings:
            print(f"{f['location']}: [{f['status']}] {f['message']}")
        if not findings:
            print("no findings")
    threshold = _SEVERITY.get(args.fail_on)
    if threshold is None:
        return 0
    worst = max((_SEVERITY[f["status"]] for f in findings), default=-1)
    return 1 if worst >= threshold else 0


# -- subcommands -----------------------------------------------------------


def cmd_rules(args) -> int:
    for r in SAMPLING_RULES:
        print(f"{r:<18} {RULE_NOTES[r]}")
    print()
    print("Two measurements may be compared only if they agree on the sampling")
    print("rule, and on the instrument unless a bound on the probe's effect is")
    print("supplied. There is no default bound.")
    return 0


def cmd_cost(args) -> int:
    t = cost.cost_table()
    findings = []
    for r in t["rows"]:
        loc = f"probe_cost:{r['id']}"
        if not r["probe"]:
            findings.append({"location": loc, "status": INFO,
                             "message": (f"no probe, {r['mean_ms_per_image']:.2f} ms/image "
                                         f"over {r['n']} images -- the baseline"),
                             "detail": r})
            continue
        if r["collected"] is False:
            findings.append({
                "location": loc, "status": ERROR,
                "message": (f"collected nothing and still reported "
                            f"{r['mean_ms_per_image']:.2f} ms/image, "
                            f"{abs(r['cost_vs_baseline'] - 1) * 100:.2f}% from the "
                            f"unprofiled baseline"),
                "detail": r})
        else:
            status = ERROR if r["cost_vs_baseline"] > 2.0 else INFO
            findings.append({
                "location": loc, "status": status,
                "message": (f"{r['cost_vs_baseline']:.3f}x the unprofiled time "
                            f"({r['launches_profiled']} launches replayed, "
                            f"{r['replay_passes']} passes each)"),
                "detail": r})
        # The sample-size note applies to the empty run too: collecting nothing
        # does not make its six-image window bigger.
        if r["n"] < 30:
            effect = abs(r["cost_vs_baseline"] - 1) * 100
            # And it must not claim a 179x effect is noise. The first version of
            # this line was phrased unconditionally and printed "a 17845.74%
            # difference over 3 samples is not a difference", which is false and
            # would have discredited every other warning in the output.
            if effect < 10.0:
                tail = (f"a {effect:.2f}% difference over {r['n']} samples is not "
                        f"a difference")
            else:
                tail = (f"the {effect:.0f}% difference is far outside anything "
                        f"{r['n']} samples could produce by chance, so the sample "
                        f"size does not threaten it -- but nothing finer than "
                        f"'much larger' can be read off {r['n']} samples")
            findings.append({
                "location": loc, "status": WARN,
                "message": f"{r['n']} images in the steady window; {tail}"})
    if not args.json:
        print(f"same probe, cost range {t['same_probe_cost_range'][0]:.3f}x to "
              f"{t['same_probe_cost_range'][1]:.1f}x -- a spread of "
              f"{t['cost_spread']:.0f}x")
        print(f"  {t['reading']}")
        print()
    return _emit(findings, args)


def cmd_empty(args) -> int:
    """Runs that produced everything except measurements."""
    fails = cost.silent_failures()
    ind = cost.indistinguishable_from_baseline(tolerance_pct=args.tolerance)
    findings = []
    for f in fails:
        findings.append({
            "location": f"probe_cost:{f['id']}", "status": ERROR,
            "message": (f"filter {f['filter']!r} matched nothing; the run wrote a "
                        f"report ({f['report_written']}) and printed "
                        f"{f['mean_ms_per_image']:.2f} ms/image, "
                        f"{f['distance_from_baseline_pct']:.3f}% from unprofiled"),
            "detail": f})
        findings.append({
            "location": f"probe_cost:{f['id']}", "status": WARN,
            "message": f"the only signal was: {f['only_signal']}"})
    if ind["both_outcomes_inside"]:
        findings.append({
            "location": f"probe_cost:+/-{args.tolerance:g}%", "status": ERROR,
            "message": ind["verdict"], "detail": ind["inside"]})
    if not fails:
        print("no run in this table collected nothing")
        return 0
    return _emit(findings, args)


def cmd_names(args) -> int:
    amb = population.ambiguous_short_names()
    findings = []
    for wl, v in amb.items():
        for c in v["collisions"]:
            status = ERROR if c["median_ratio"] >= 2.0 else WARN
            findings.append({
                "location": f"{wl}:{c['short_name']}", "status": status,
                "message": (f"{c['rows']} different kernels share this short name; "
                            f"their medians span {c['median_ratio']:.1f}x "
                            f"({c['instances']} launches respectively)"),
                "detail": c})
        findings.append({
            "location": wl, "status": INFO,
            "message": (f"{v['colliding_short_names']} of {v['distinct_short_names']} "
                        f"short names cover more than one kernel "
                        f"({v['names_total']} rows total)")})
    return _emit(findings, args)


def cmd_spread(args) -> int:
    w = population.name_row_spread(factor=args.factor)
    findings = []
    for wl, v in w.items():
        findings.append({
            "location": wl, "status": ERROR if v["names_spanning_factor"] else INFO,
            "message": (f"{v['names_spanning_factor']} of {v['names_considered']} "
                        f"kernel names (>=10 launches) span {v['factor']:g}x or more "
                        f"within the name; median CV {v['median_cv_pct']:.1f}%, worst "
                        f"{v['worst']['spread']:.0f}x over {v['worst']['instances']} "
                        f"launches"),
            "detail": v})
    if not args.json:
        print("a per-name row is an average over a mixture; these are the mixtures")
        print()
    return _emit(findings, args)


def cmd_compare(args) -> int:
    """Assemble the cross-instrument comparison and print the refusal."""
    try:
        population.compare_across_instruments(args.kernel, args.nsys_name or args.kernel)
    except InstrumentError as exc:
        d = getattr(exc, "detail", None)
        print(f"compare:{args.kernel}: [error] {exc}")
        if d:
            for side in ("a", "b"):
                s = d[side]
                print(f"  {s['instrument']:<16} n={s['n']:<6} median="
                      f"{s['median_ns']:>12,.0f} ns  internal spread "
                      f"{s['spread']:>6.1f}x  ({s['sampling_rule']})")
            print(f"  ratio if you did it anyway: "
                  f"{d['ratio_if_you_did_it_anyway']:.3f}x")
            print(f"  the two populations differ in size by "
                  f"{d['population_size_ratio']:.0f}x")
        return 1 if args.fail_on in ("error", "warn") else 0
    print(f"compare:{args.kernel}: [warn] the comparison was allowed -- check why")
    return 0


def cmd_check(args) -> int:
    """Every check over the committed data. This is the CI gate's report."""
    findings = []
    t = cost.cost_table()
    for r in t["rows"]:
        if r["collected"] is False:
            findings.append({"location": f"probe_cost:{r['id']}", "status": ERROR,
                             "message": "collected nothing and reported a number"})
        elif r["probe"] and r["cost_vs_baseline"] > 2.0:
            findings.append({
                "location": f"probe_cost:{r['id']}", "status": ERROR,
                "message": f"probe cost {r['cost_vs_baseline']:.0f}x"})
    for wl, v in population.ambiguous_short_names().items():
        for c in v["collisions"]:
            if c["median_ratio"] >= 2.0:
                findings.append({
                    "location": f"{wl}:{c['short_name']}", "status": ERROR,
                    "message": (f"{c['rows']} kernels share this name, medians span "
                                f"{c['median_ratio']:.1f}x")})
    for wl, v in population.name_row_spread().items():
        findings.append({
            "location": f"{wl}:per-name-rows", "status": ERROR,
            "message": (f"{v['names_spanning_factor']}/{v['names_considered']} names "
                        f"span 2x or more within the name")})
    for g in population.same_shape_still_varies():
        findings.append({
            "location": f"launches:{g['kernel']}", "status": ERROR,
            "message": (f"identical grid {g['grid']} and block {g['block']}, duration "
                        f"still spans {g['spread']:.2f}x")})
    if args.json:
        print(json.dumps(findings, indent=2, ensure_ascii=False))
    else:
        for f in findings:
            print(f"{f['location']}: [{f['status']}] {f['message']}")
        print(f"\n{len(findings)} finding(s). Every one is a property of the "
              f"committed data, so `probecost check` is expected to be loud and "
              f"exits 0.")
    return 0


def cmd_claims(args) -> int:
    a = analysis.all_findings()
    if args.json:
        print(json.dumps(a, indent=2, ensure_ascii=False, default=str))
        return 0
    p = a["probe_cost"]
    print("what the probe cost")
    print(f"  unprofiled baseline  {p['baseline_ms']:.2f} ms/image over "
          f"{p['baseline_n']} images")
    for r in p["rows"]:
        if not r["probe"]:
            continue
        mark = "  (collected nothing)" if r["collected"] is False else ""
        print(f"  {r['id']:<34} {r['cost_vs_baseline']:>8.3f}x   "
              f"{r['launches_profiled']:>4} launches x {r['replay_passes']} passes"
              f"{mark}")
    print(f"  same probe, {p['cost_spread']:.0f}x spread; "
          f"filtered {p['filtered_range'][0]:.3f}-{p['filtered_range'][1]:.3f}x, "
          f"unfiltered {p['unfiltered']:.1f}x")
    print(f"  {p['passes_do_not_explain_it']['reading']}")

    s = a["silent_failure"]
    print("\nthe run that collected nothing")
    print(f"  {s['count']} run, {s['closest_to_baseline_pct']:.3f}% from the "
          f"unprofiled baseline, report written: {s['and_it_still_wrote_a_report']}")
    print(f"  wall clock can detect it: {not s['wall_clock_cannot_detect_it']}")
    print(f"  the only signal: {s['what_the_only_signal_was']}")

    r = a["instrument_is_repeatable"]
    print("\nthe control: is the instrument itself precise?")
    print(f"  grouped by (name, grid, block): median CV {r['median_cv_pct']:.3f}%, "
          f"{r['groups_under_1pct_cv']}/{r['groups_with_repeats']} groups under 1%")

    k = a["the_name_is_the_wrong_key"]
    print("\nthe same eight launches, grouped by name instead")
    print(f"  worst: {k['worst']['kernel'][:34]} spans {k['worst']['spread']:.2f}x "
          f"over {k['worst']['n']} launches, {k['worst']['distinct_shapes']} shapes")
    for g in k["same_geometry_still_varies"]:
        print(f"  and with grid {g['grid']} block {g['block']} held fixed it still "
              f"spans {g['spread']:.2f}x")

    n = a["the_name_matches_two_things"]
    print("\nthe short name is not a key, in both directions")
    for wl, v in n["by_workload"].items():
        print(f"  {wl:<20} {v['colliding_short_names']}/{v['distinct_short_names']} "
              f"short names cover more than one kernel")
    w = n["worst_collision"]
    print(f"  worst: {w['rows']} kernels called {w['short_name']!r}, medians span "
          f"{w['median_ratio']:.1f}x")
    print(f"  and the other way: filter {n['and_the_other_direction']['filter']!r} "
          f"matched {n['and_the_other_direction']['matched']}")

    print("\nwhole runs, per-name rows, two workloads")
    for wl, v in a["whole_run_spread"].items():
        print(f"  {wl:<20} {v['names_spanning_factor']:>3}/{v['names_considered']} "
              f"names span 2x+  median CV {v['median_cv_pct']:.1f}%  worst "
              f"{v['worst']['spread']:.0f}x  ({v['total_launches']:,} launches)")

    c = a["the_comparison_that_gets_refused"]
    print("\nputting the two instruments in one table")
    print(f"  {len(c['pairs'])} comparisons assembled, all refused: {c['all_refused']}")
    print(f"  if you ignored the refusal: "
          f"{c['ratio_range_if_you_ignored_the_refusal'][0]:.3f}x to "
          f"{c['ratio_range_if_you_ignored_the_refusal'][1]:.3f}x, "
          f"sign changes: {c['changes_sign']}")
    print(f"  {c['reading']}")

    sz = a["sample_size_limits"]
    print("\nwhat this data cannot carry")
    print(f"  {len(sz['cells'])} of the cells have fewer than 30 images: "
          f"{[(x['id'], x['n']) for x in sz['cells']]}")
    return 0


def cmd_show(args) -> int:
    t, ls, st = CostTable.load(), LaunchSet.load(), SummaryTable.load()
    print(f"probe_cost       {len(t.runs)} runs, baseline {t.baseline_id!r} "
          f"({t.baseline.images_in_steady_window} images)")
    print(f"                 workload: {t.workload['model']}")
    print(f"launch_durations {len(ls.launches)} launches, {len(ls.kernels)} kernels")
    print(f"                 {ls.instrument['name']} {ls.instrument['version']}, "
          f"rule: {ls.instrument['sampling_rule']}")
    print(f"name_summaries   {st.instrument['name']} {st.instrument['version']}, "
          f"rule: {st.instrument['sampling_rule']}")
    for wl, v in st.workloads.items():
        print(f"                 {wl}: {len(v['rows'])} names, "
              f"{sum(r.instances for r in v['rows']):,} launches")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="probecost", description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on", choices=["error", "warn", "none"], default="none")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, helptext in (
        ("rules", cmd_rules, "what each sampling rule means"),
        ("show", cmd_show, "what is in data/"),
        ("cost", cmd_cost, "what the probe cost, per run"),
        ("names", cmd_names, "short names that cover more than one kernel"),
        ("check", cmd_check, "every check over the committed data"),
        ("claims", cmd_claims, "this repository's findings, recomputed"),
    ):
        sub.add_parser(name, help=helptext).set_defaults(fn=fn)

    p = sub.add_parser("empty", help="runs that reported a number and measured nothing")
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="percent from baseline to call indistinguishable")
    p.set_defaults(fn=cmd_empty)

    p = sub.add_parser("spread", help="how wide a per-kernel-name row really is")
    p.add_argument("--factor", type=float, default=2.0)
    p.set_defaults(fn=cmd_spread)

    p = sub.add_parser("compare", help="try to compare two instruments' numbers")
    p.add_argument("kernel")
    p.add_argument("--nsys-name", help="substring identifying the summary row")
    p.set_defaults(fn=cmd_compare)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
