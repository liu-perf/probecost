"""The loaders are strict, and each refusal is a state a real dataset was in."""

from __future__ import annotations

import json

import pytest

from probecost.model import (
    DATA_DIR,
    CostTable,
    DataError,
    LaunchSet,
    SummaryTable,
    load_all,
)


def _cost_raw(tmp_path, mutate=None):
    raw = json.loads((DATA_DIR / "probe_cost.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(raw)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


def test_all_three_datasets_load():
    t, ls, st = load_all()
    assert len(t.runs) == 5
    assert len(ls.launches) == 40
    assert set(st.workloads) == {"depth_anything_v2", "flownet2"}


def test_a_run_missing_collected_anything_is_refused(tmp_path):
    def drop(raw):
        del raw["runs"][1]["collected_anything"]
    with pytest.raises(DataError, match="collected_anything"):
        CostTable.load(_cost_raw(tmp_path, drop))


def test_a_probed_baseline_is_refused(tmp_path):
    """A probe-cost table whose baseline is instrumented measures nothing."""
    def probe_the_baseline(raw):
        raw["runs"][0]["probe"] = "nsight-compute"
    with pytest.raises(DataError, match=r"baseline .* has a probe"):
        CostTable.load(_cost_raw(tmp_path, probe_the_baseline))


def test_the_baseline_is_the_only_cell_with_a_usable_sample():
    t = CostTable.load()
    assert t.baseline.probe is None
    assert t.baseline.images_in_steady_window == 760
    others = [r.images_in_steady_window for r in t.runs if r.id != t.baseline_id]
    assert max(others) <= 6, others


def test_the_empty_run_is_separated_from_the_collected_ones():
    t = CostTable.load()
    assert [r.id for r in t.empty_runs] == ["ncu_filter_cutlass_short_name"]
    assert t.empty_runs[0].report_written is True
    assert t.empty_runs[0].stderr_line


def test_a_run_becomes_a_measurement_carrying_its_provenance():
    t = CostTable.load()
    base = t.baseline.measurement
    probed = t.by_id("ncu_filter_magma").measurement
    assert base.instrument is None and probed.instrument == "nsight-compute"
    assert base.sampling_rule != probed.sampling_rule


def test_the_launch_set_declares_a_sampling_rule():
    ls = LaunchSet.load()
    assert ls.instrument["sampling_rule"].startswith("the first N")
    assert ls.instrument["replay_passes"] == 21


@pytest.mark.parametrize("how", ["delete", "null", "empty"])
def test_an_instrument_without_a_sampling_rule_is_refused(tmp_path, how):
    """All three spellings of absence.

    The first version only checked for the key being missing, so a rule set to
    ``null`` sailed through and blew up later with an AttributeError. A key
    present with no value is the same absence.
    """
    raw = json.loads((DATA_DIR / "launch_durations.json").read_text(encoding="utf-8"))
    if how == "delete":
        del raw["instrument"]["sampling_rule"]
    else:
        raw["instrument"]["sampling_rule"] = None if how == "null" else ""
    p = tmp_path / "l.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataError, match="sampling rule"):
        LaunchSet.load(p)


def test_a_name_lookup_that_matches_two_rows_is_refused():
    """The bug this package is about, enforced by its own loader.

    ``cunn_SoftMaxForwardReg`` names two rows in the summary table whose
    medians are nearly nine times apart. A lookup that silently returned the
    first would be the defect, not a convenience.
    """
    st = SummaryTable.load()
    with pytest.raises(DataError, match="matches 2 rows"):
        st.row("depth_anything_v2", "cunn_SoftMaxForwardReg")


def test_a_name_lookup_that_matches_nothing_is_also_refused():
    st = SummaryTable.load()
    with pytest.raises(DataError, match="matches 0 rows"):
        st.row("depth_anything_v2", "no_such_kernel_anywhere")


def test_an_unambiguous_lookup_works():
    st = SummaryTable.load()
    r = st.row("depth_anything_v2", "magma_sgemmEx_kernel")
    assert r.instances == 2376
    assert r.med_ns == 216992.0


def test_the_summary_rows_expose_their_own_internal_spread():
    st = SummaryTable.load()
    r = st.row("depth_anything_v2", "magma_sgemmEx_kernel")
    assert r.spread > 80.0
    assert r.cv_pct > 100.0


def test_every_launch_carries_its_geometry():
    for x in LaunchSet.load().launches:
        assert x.grid and x.block
        assert x.duration_ns > 0
        assert len(x.shape) == 3


def test_every_data_file_is_utf8_without_a_bom():
    for p in sorted(DATA_DIR.glob("*.json")):
        blob = p.read_bytes()
        assert not blob.startswith(b"\xef\xbb\xbf"), p.name
        json.loads(blob.decode("utf-8"))


def test_the_machine_note_says_the_findings_do_not_depend_on_it():
    t = CostTable.load()
    assert "survives the machine changing" in t.machine["note"]
