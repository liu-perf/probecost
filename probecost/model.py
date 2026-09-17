"""Loading the three datasets, and refusing ones that cannot be reasoned about."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .instrument import EXHAUSTIVE, PREFIX, check_declared

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


class DataError(Exception):
    """The committed data is not in a state this package can reason about."""


@dataclass(frozen=True)
class Measurement:
    """One number, with everything needed to know what it may be compared to."""

    label: str
    value: float
    unit: str
    instrument: str | None
    sampling_rule: str
    collected: bool | None
    n: int = 0
    note: str = ""

    def __post_init__(self):
        check_declared(self.label, self.instrument, self.sampling_rule)


@dataclass
class Run:
    """One execution of the workload under one probe configuration."""

    id: str
    probe: str | None
    filter: str | None
    replay_passes: int
    launches_profiled: int
    images_counted: int
    images_in_steady_window: int
    mean_ms_per_image: float
    images_per_second: float
    collected_anything: bool | None
    note: str = ""
    stderr_line: str | None = None
    report_written: bool | None = None

    @property
    def measurement(self) -> Measurement:
        return Measurement(
            label=self.id, value=self.mean_ms_per_image, unit="ms/image",
            instrument=self.probe,
            sampling_rule=PREFIX if self.probe else EXHAUSTIVE,
            collected=self.collected_anything,
            n=self.images_in_steady_window, note=self.note,
        )


@dataclass
class CostTable:
    what: str
    machine: dict
    workload: dict
    runs: list
    baseline_id: str

    @property
    def baseline(self) -> Run:
        return self.by_id(self.baseline_id)

    def by_id(self, rid: str) -> Run:
        for r in self.runs:
            if r.id == rid:
                return r
        raise DataError(f"no run {rid!r}")

    @property
    def collected(self) -> list:
        return [r for r in self.runs if r.collected_anything is not False]

    @property
    def empty_runs(self) -> list:
        """Runs that produced output and no measurements."""
        return [r for r in self.runs if r.collected_anything is False]

    @classmethod
    def load(cls, path=None) -> CostTable:
        raw = json.loads(pathlib.Path(path or DATA_DIR / "probe_cost.json")
                         .read_text(encoding="utf-8"))
        runs = []
        for d in raw["runs"]:
            missing = {"id", "probe", "replay_passes", "launches_profiled",
                       "mean_ms_per_image", "collected_anything"} - set(d)
            if missing:
                raise DataError(f"run {d.get('id')!r} is missing {sorted(missing)}")
            runs.append(Run(**{k: d.get(k) for k in Run.__dataclass_fields__}))
        if not runs:
            raise DataError("no runs")
        table = cls(what=raw["what"], machine=raw["machine"], workload=raw["workload"],
                    runs=runs, baseline_id=raw["baseline_id"])
        base = table.baseline
        if base.probe is not None:
            raise DataError(
                f"the baseline run {base.id!r} has a probe attached; a probe-cost "
                f"table whose baseline is itself instrumented measures nothing"
            )
        return table


@dataclass
class Launch:
    """One kernel launch, as an instrument reported it."""

    kernel: str
    grid: str
    block: str
    duration_ns: int

    @property
    def shape(self) -> tuple:
        return (self.kernel, self.grid, self.block)


@dataclass
class LaunchSet:
    what: str
    instrument: dict
    workload: dict
    launches: list

    def by_kernel(self, name: str) -> list:
        return [x for x in self.launches if x.kernel == name]

    @property
    def kernels(self) -> list:
        seen = []
        for x in self.launches:
            if x.kernel not in seen:
                seen.append(x.kernel)
        return seen

    @classmethod
    def load(cls, path=None) -> LaunchSet:
        raw = json.loads(pathlib.Path(path or DATA_DIR / "launch_durations.json")
                         .read_text(encoding="utf-8"))
        inst = raw["instrument"]
        # `not in` is not enough: a key present with a null value is the same
        # absence, and the fault-injection pass found exactly that hole by
        # setting the rule to null instead of deleting it.
        if not inst.get("sampling_rule"):
            raise DataError("the instrument does not declare a sampling rule")
        out = []
        for kernel, rows in raw["launches"].items():
            for r in rows:
                out.append(Launch(kernel=kernel, grid=r["grid"], block=r["block"],
                                  duration_ns=int(r["duration_ns"])))
        if not out:
            raise DataError("no launches")
        return cls(what=raw["what"], instrument=inst, workload=raw["workload"],
                   launches=out)


@dataclass
class NameSummary:
    """One row of a per-kernel-name profiler summary."""

    name: str
    instances: int
    avg_ns: float
    med_ns: float
    min_ns: float
    max_ns: float
    stddev_ns: float
    total_ns: int
    time_pct: float

    @property
    def spread(self) -> float:
        """max/min within the name. 1.0 means every launch took the same time."""
        return self.max_ns / self.min_ns if self.min_ns else float("inf")

    @property
    def cv_pct(self) -> float:
        return self.stddev_ns / self.avg_ns * 100.0 if self.avg_ns else float("inf")


@dataclass
class SummaryTable:
    what: str
    instrument: dict
    workloads: dict = field(default_factory=dict)

    def rows(self, workload: str) -> list:
        return self.workloads[workload]["rows"]

    def row(self, workload: str, contains: str) -> NameSummary:
        hits = [r for r in self.rows(workload) if contains in r.name]
        if len(hits) != 1:
            raise DataError(
                f"{contains!r} matches {len(hits)} rows in {workload}; a per-name "
                f"lookup that matches more than one name is the bug this package "
                f"is about"
            )
        return hits[0]

    @classmethod
    def load(cls, path=None) -> SummaryTable:
        raw = json.loads(pathlib.Path(path or DATA_DIR / "name_summaries.json")
                         .read_text(encoding="utf-8"))
        wl = {}
        for k, v in raw["workloads"].items():
            wl[k] = dict(v)
            wl[k]["rows"] = [NameSummary(**r) for r in v["rows"]]
        return cls(what=raw["what"], instrument=raw["instrument"], workloads=wl)


def load_all() -> tuple:
    return CostTable.load(), LaunchSet.load(), SummaryTable.load()


__all__ = ["CostTable", "DataError", "Launch", "LaunchSet", "Measurement",
           "NameSummary", "Run", "SummaryTable", "load_all"]
