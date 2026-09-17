"""The device: a number carries the instrument that produced it, or it is not a number.

Fifteenth in a series. The fourteen before it each separated two things that
looked identical in somebody's output. This one separates::

    "this kernel takes 216 microseconds"
        from
    "this kernel took 216 microseconds under one probe's sampling rule"

and, inside that, a second pair that is worse::

    "the probe measured it and it was fast"
        from
    "the probe measured nothing"

A profiler's output is a table of durations. The instrument that produced it is
not in the table -- it is in the command line, which is in somebody's shell
history. So two rows from two profilers land in one spreadsheet and get
subtracted, and nothing objects.

Three things every measurement must declare, and the reason each is required:

``instrument``
    Which probe, or ``None`` for none. Without it, two rows that were produced
    by different machinery are indistinguishable from two rows of the same
    measurement.

``sampling_rule``
    **What set of events the instrument looked at.** This is the field people
    do not expect and it is the one that does the damage. Nsight Compute's
    ``-c N`` takes the *first* N matching launches -- a prefix, not a sample.
    Nsight Systems takes every launch in the traced region. Two medians over
    those two sets are not two estimates of one quantity.

``collected``
    Whether anything came back. Not inferable from the timings: in the data
    committed here, the run whose filter matched **zero** kernels produced a
    throughput within 0.1% of the unprofiled baseline and sat *between* the two
    runs that worked. A `None` here means nobody checked, and `None` is not
    `True`.

The asymmetry that makes this a device rather than a schema is in
:func:`assert_comparable`. Two measurements may be compared only if they agree
on instrument *and* sampling rule, or if the caller supplies a bound on the
probe's effect for this comparison. There is no default bound. A caller who has
not bounded the probe cannot subtract one from the other by forgetting to.
"""

from __future__ import annotations

#: Sampling rules seen in the wild. The strings are the point: a rule that is
#: not one of these has to be spelled out before it can be reasoned about.
PREFIX = "first-n-matching"      # ncu -c N: a prefix of the matching launches
EXHAUSTIVE = "every-launch"      # nsys: every launch in the traced region
UNKNOWN = "unknown"

SAMPLING_RULES = (PREFIX, EXHAUSTIVE, UNKNOWN)

RULE_NOTES = {
    PREFIX: "the first N matching launches, in launch order -- a prefix, not a sample",
    EXHAUSTIVE: "every launch in the traced region",
    UNKNOWN: "not declared; nothing may be concluded about what this summarises",
}


class InstrumentError(Exception):
    """A measurement was used in a way its provenance does not permit."""


class UndeclaredInstrumentError(InstrumentError):
    """A number arrived with no instrument attached.

    Not a warning. A duration with no instrument is exactly the state every
    profiler summary table is in, and it is the state in which two
    incommensurable numbers get subtracted.
    """


class NotCollectedError(InstrumentError):
    """A measurement is being used from a run that collected nothing."""


def check_declared(label: str, instrument: object, sampling_rule: object) -> tuple:
    """Every measurement declares an instrument and a sampling rule.

    ``instrument=None`` is a legitimate, explicit declaration -- it means no
    probe was attached. It is not the same as the field being absent, and the
    loader keeps the two apart.
    """
    if sampling_rule is None:
        raise UndeclaredInstrumentError(
            f"{label}: no sampling rule declared; without it there is no way to say "
            f"what set of events this number summarises"
        )
    if sampling_rule not in SAMPLING_RULES:
        raise InstrumentError(
            f"{label}: unknown sampling rule {sampling_rule!r}; expected one of "
            f"{SAMPLING_RULES}"
        )
    return instrument, sampling_rule


def assert_collected(label: str, collected: object) -> None:
    """Raise unless the run is known to have collected something.

    ``None`` raises. It means nobody recorded whether anything came back, and
    the committed data contains a run where the answer was no and every printed
    number looked normal.
    """
    if collected is None:
        raise NotCollectedError(
            f"{label}: whether this run collected anything was never recorded; "
            f"the timings cannot tell you (see data/probe_cost.json: the run whose "
            f"filter matched zero kernels landed 0.1% from the unprofiled baseline)"
        )
    if not collected:
        raise NotCollectedError(
            f"{label}: this run collected nothing -- it produced a report file and a "
            f"throughput number and no measurements"
        )


def assert_comparable(a, b, *, probe_effect_bound=None) -> None:
    """Raise unless two measurements may be put in the same table.

    ``probe_effect_bound`` is keyword-only with no default that means "assume
    it is small". A caller who has not bounded the probe's effect on *this*
    comparison cannot get past this by omission.
    """
    for m in (a, b):
        assert_collected(m.label, m.collected)

    if a.sampling_rule != b.sampling_rule:
        raise InstrumentError(
            f"{a.label} and {b.label} summarise different sets of launches "
            f"({a.sampling_rule}: {RULE_NOTES[a.sampling_rule]}; "
            f"{b.sampling_rule}: {RULE_NOTES[b.sampling_rule]}); they are not two "
            f"estimates of one quantity and no bound on the probe repairs that"
        )
    if a.instrument == b.instrument:
        return
    if probe_effect_bound is None:
        raise InstrumentError(
            f"{a.label} came from {a.instrument or 'no probe'} and {b.label} from "
            f"{b.instrument or 'no probe'}; comparing them requires a bound on what "
            f"the probe does to this measurement, and none was given"
        )
    if probe_effect_bound < 0:
        raise InstrumentError("a probe-effect bound cannot be negative")


__all__ = [
    "EXHAUSTIVE", "PREFIX", "RULE_NOTES", "SAMPLING_RULES", "UNKNOWN",
    "InstrumentError", "NotCollectedError", "UndeclaredInstrumentError",
    "assert_collected", "assert_comparable", "check_declared",
]
