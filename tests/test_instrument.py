"""The device, on the things it exists to forbid."""

from __future__ import annotations

import pytest

from probecost.instrument import (
    EXHAUSTIVE,
    PREFIX,
    UNKNOWN,
    InstrumentError,
    NotCollectedError,
    UndeclaredInstrumentError,
    assert_collected,
    assert_comparable,
    check_declared,
)
from probecost.model import Measurement


def m(label, instrument, rule, collected=True, value=1.0):
    return Measurement(label=label, value=value, unit="ns", instrument=instrument,
                       sampling_rule=rule, collected=collected)


def test_a_number_with_no_sampling_rule_is_refused():
    with pytest.raises(UndeclaredInstrumentError):
        check_declared("x", "ncu", None)


def test_an_unknown_sampling_rule_is_refused():
    with pytest.raises(InstrumentError, match="unknown sampling rule"):
        check_declared("x", "ncu", "vibes")


def test_no_probe_is_a_legitimate_declaration():
    """`instrument=None` means no probe was attached, which is a fact, not a gap."""
    assert check_declared("x", None, EXHAUSTIVE) == (None, EXHAUSTIVE)


def test_a_measurement_constructed_without_a_rule_raises_at_construction():
    with pytest.raises(UndeclaredInstrumentError):
        Measurement(label="x", value=1.0, unit="ns", instrument="ncu",
                    sampling_rule=None, collected=True)


def test_collected_none_raises_because_nobody_checked():
    """The important one.

    ``None`` is not ``True``. The committed data contains a run where the
    answer was no and every printed number looked normal, so "nobody recorded
    it" cannot be allowed to pass as "it worked".
    """
    with pytest.raises(NotCollectedError, match="never recorded"):
        assert_collected("x", None)


def test_collected_false_raises():
    with pytest.raises(NotCollectedError, match="collected nothing"):
        assert_collected("x", False)


def test_collected_true_passes():
    assert_collected("x", True)   # returns None; the point is that it does not raise


def test_two_measurements_from_the_same_instrument_and_rule_compare():
    assert_comparable(m("a", "ncu", PREFIX), m("b", "ncu", PREFIX))


def test_different_sampling_rules_are_never_comparable():
    """Not even with a bound. A bound on the probe cannot repair a population."""
    with pytest.raises(InstrumentError, match="different sets of launches"):
        assert_comparable(m("a", "ncu", PREFIX), m("b", "nsys", EXHAUSTIVE),
                          probe_effect_bound=0.01)


def test_different_instruments_need_a_bound():
    a, b = m("a", "ncu", PREFIX), m("b", "nsys", PREFIX)
    with pytest.raises(InstrumentError, match="none was given"):
        assert_comparable(a, b)
    assert_comparable(a, b, probe_effect_bound=0.05)


def test_the_bound_is_keyword_only_with_no_permissive_default():
    """A caller who never bounded the probe cannot pass one by position."""
    a, b = m("a", "ncu", PREFIX), m("b", None, PREFIX)
    with pytest.raises(TypeError):
        assert_comparable(a, b, 0.05)


def test_a_negative_bound_is_refused():
    with pytest.raises(InstrumentError, match="cannot be negative"):
        assert_comparable(m("a", "ncu", PREFIX), m("b", None, PREFIX),
                          probe_effect_bound=-1.0)


def test_a_run_that_collected_nothing_is_refused_before_the_rules_are_checked():
    """Ordering matters: 'it measured nothing' beats every other objection."""
    with pytest.raises(NotCollectedError):
        assert_comparable(m("a", "ncu", PREFIX, collected=False),
                          m("b", "ncu", PREFIX))


def test_unknown_is_a_declarable_rule_and_still_blocks_comparison():
    """`unknown` is honest and useless, which is the correct combination."""
    with pytest.raises(InstrumentError, match="different sets of launches"):
        assert_comparable(m("a", "ncu", UNKNOWN), m("b", "ncu", PREFIX))
