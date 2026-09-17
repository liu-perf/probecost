"""Exit codes and output shape. The CI gates depend on both."""

from __future__ import annotations

import json

import pytest

from probecost.cli import main


def test_rules_lists_all_three_and_states_the_comparison_condition(capsys):
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    for r in ("first-n-matching", "every-launch", "unknown"):
        assert r in out
    assert "no default bound" in out


def test_show_names_the_two_instruments_and_their_rules(capsys):
    assert main(["show"]) == 0
    out = capsys.readouterr().out
    assert "nsight-compute" in out and "nsight-systems" in out
    assert "760 images" in out
    assert "14,789" in out and "54,380" in out


def test_cost_flags_the_unfiltered_run_and_the_empty_one(capsys):
    assert main(["--fail-on", "error", "cost"]) == 1
    out = capsys.readouterr().out
    assert "collected nothing and still reported" in out
    assert "179.457x" in out
    assert "spread of 184x" in out


def test_cost_warns_on_every_underpowered_cell(capsys):
    """All four probed cells get the note, including the one that collected nothing."""
    main(["cost"])
    out = capsys.readouterr().out
    assert out.count("images in the steady window") == 4


def test_the_sample_size_note_does_not_call_a_179x_effect_noise(capsys):
    """A guard on this tool's own wording.

    The first version of the note was phrased unconditionally and printed
    "a 17845.74% difference over 3 samples is not a difference", which is
    false. One sentence like that discredits every other warning in the
    output, so the phrasing is conditional and this pins it.
    """
    main(["cost"])
    out = capsys.readouterr().out
    assert "17845" not in out
    assert "far outside anything" in out
    # and the genuinely small ones still get the blunt wording
    assert out.count("is not a difference") == 3


def test_empty_reports_the_undetectability_as_an_error(capsys):
    assert main(["--fail-on", "error", "empty"]) == 1
    out = capsys.readouterr().out
    assert "matched nothing" in out
    assert "No kernels were profiled" in out
    assert "no threshold on wall-clock cost" in out


@pytest.mark.parametrize("tol", ["1", "2", "5", "10"])
def test_empty_stays_undetectable_at_every_plausible_tolerance(tol, capsys):
    assert main(["--fail-on", "error", "empty", "--tolerance", tol]) == 1
    assert "no threshold" in capsys.readouterr().out


def test_names_reports_the_kernel2_collision(capsys):
    assert main(["--fail-on", "error", "names"]) == 1
    out = capsys.readouterr().out
    assert "Kernel2" in out
    assert "17 different kernels share this short name" in out
    assert "5 of 22 short names" in out


def test_spread_reports_both_workloads(capsys):
    assert main(["--fail-on", "error", "spread"]) == 1
    out = capsys.readouterr().out
    assert "29 of 41" in out
    assert "54 of 94" in out


def test_spread_factor_is_a_flag(capsys):
    main(["spread", "--factor", "1000"])
    out = capsys.readouterr().out
    assert "29 of 41" not in out


def test_compare_refuses_and_prints_both_numbers(capsys):
    assert main(["--fail-on", "error", "compare", "magma_sgemmEx_kernel"]) == 1
    out = capsys.readouterr().out
    assert "[error]" in out
    assert "nsight-compute" in out and "nsight-systems" in out
    assert "ratio if you did it anyway" in out
    assert "differ in size by" in out


def test_compare_on_an_ambiguous_name_fails_earlier(capsys):
    """The lookup refusal, reached through the CLI."""
    from probecost.model import DataError

    with pytest.raises(DataError, match="matches 2 rows"):
        main(["compare", "cunn_SoftMaxForwardReg"])


def test_check_is_loud_and_exits_zero(capsys):
    """``check`` is a report, not a gate.

    Every line is a property of the committed data, so a non-zero exit would
    mean this repository never builds. The gates live in ``tests/``.
    """
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert out.count("[error]") >= 10
    assert "expected to be loud" in out


def test_claims_prints_every_section(capsys):
    assert main(["claims"]) == 0
    out = capsys.readouterr().out
    for heading in ("what the probe cost", "the run that collected nothing",
                    "the control: is the instrument itself precise?",
                    "the short name is not a key, in both directions",
                    "whole runs, per-name rows, two workloads",
                    "putting the two instruments in one table",
                    "what this data cannot carry"):
        assert heading in out


def test_json_output_parses_for_every_reporting_subcommand(capsys):
    for argv in (["--json", "cost"], ["--json", "empty"], ["--json", "names"],
                 ["--json", "spread"], ["--json", "check"], ["--json", "claims"]):
        main(argv)
        json.loads(capsys.readouterr().out)


def test_fail_on_defaults_to_none(capsys):
    assert main(["cost"]) == 0
    capsys.readouterr()
