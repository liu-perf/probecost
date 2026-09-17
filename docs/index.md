# probecost

**What the profiler cost, and what its output entitles you to say.**

| | |
|---|---|
| the same probe cost | **1.00×** with a kernel-name filter, **179×** without one |
| a run whose filter matched **zero** kernels | wrote a report, printed a throughput **0.1%** from unprofiled, and landed *between* the two runs that worked |
| one short kernel name | covers **17** different CUTLASS kernels whose medians span **127×** |

A profiler's output is a table of durations. The instrument that produced it is
not in the table — it is in the command line, which is in somebody's shell
history. So two rows from two profilers land in one spreadsheet and get
subtracted, and nothing objects.

## The device: `instrument`

Every number declares the probe, **what set of events the probe looked at**, and
whether anything came back. `ncu -c N` takes a *prefix* of the matching
launches; `nsys` takes all of them. Two medians over those two sets are not two
estimates of one quantity, and no bound on the probe repairs that.

## Reading order

* **[the-probe-that-measured-nothing.md](the-probe-that-measured-nothing.md)** —
  the cost table, why the metric set is not the explanation, and why no
  threshold on the timings can detect a probe that collected nothing.
* **[the-name-is-not-a-key.md](the-name-is-not-a-key.md)** — the control that
  shows the instrument is precise, then 5.47× in eight launches, 3.47× at
  identical geometry, and a short name that both over-matches and
  under-matches.
* **[refusing-the-subtraction.md](refusing-the-subtraction.md)** — assembling
  the cross-instrument comparison, the −22%/+29% reading that is available and
  wrong, and what the refusal returns instead.

## Zero dependencies, no GPU

Producing `data/` took a GPU, Nsight Systems and Nsight Compute. Re-checking
every conclusion is arithmetic over committed JSON, so CI is pure CPU and so is
anyone reproducing the claims.

Each finding was verified by fault injection, and `tests/test_ci_runs.py`
executes every `run:` block of the workflow under `bash -e` — because three
repositories in this series shipped a CI that was red on every push while
`pytest` and `ruff` stayed green.

## Sister projects

Sixteen tools, each separating two things that looked identical in somebody's
output. `probecost` is the seventeenth, and its device is the fifteenth:
**"this kernel takes 216 µs" against "this kernel took 216 µs under one probe's
sampling rule"** — and inside it, **"the probe measured it and it was fast"
against "the probe measured nothing."**
