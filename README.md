# probecost

**What the profiler cost, and what its output entitles you to say.**

Three results from one afternoon of real Nsight data, on two workloads:

| | |
|---|---|
| the same probe cost | **1.00×** with a kernel-name filter, **179×** without one |
| a run whose filter matched **zero** kernels | wrote a report, printed a throughput **0.1%** from unprofiled, and landed *between* the two runs that worked |
| one short kernel name | covers **17** different CUTLASS kernels whose medians span **127×** |

Zero dependencies, pure standard library, CPU-only. Producing `data/` took a
GPU, Nsight Systems and Nsight Compute. Re-deriving every conclusion below is
arithmetic over committed JSON and needs none of them.

```console
$ pip install -e .
$ probecost claims
what the probe cost
  unprofiled baseline  83.41 ms/image over 760 images
  ncu_filter_magma                         0.977x      8 launches x 21 passes
  ncu_filter_cutlass_short_name            1.001x      0 launches x 21 passes  (collected nothing)
  ncu_filter_cutlass_full_name             1.009x      8 launches x 21 passes
  ncu_no_filter                          179.457x    420 launches x 9 passes
```

---

## 1. The probe's cost is not a property of the probe

Same Nsight Compute, same workload, same machine. With a kernel-name filter it
is **indistinguishable from no profiler at all**; with no filter it is **179×**.

The obvious explanation is wrong and the data rules it out: **the expensive run
used _fewer_ replay passes per launch** — 9 for the basic metric set against 21
for the detailed one. The cost follows how many launches get replayed, not how
heavy the metrics are.

So "we profiled it, so the timings are inflated" and "we profiled it, so the
timings are fine" are both available, and which one is true depends on a
command-line flag that does not appear anywhere in the output.

## 2. A run that measured nothing, and no number that says so

One filter used the short kernel name `cutlass_80_simt_sgemm`. The kernel's
real name is `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)`.
Nothing matched.

```console
$ probecost empty
probe_cost:ncu_filter_cutlass_short_name: [error] filter 'cutlass_80_simt_sgemm' matched
nothing; the run wrote a report (True) and printed 83.49 ms/image, 0.096% from unprofiled
probe_cost:ncu_filter_cutlass_short_name: [warn] the only signal was: ==PROF== ... 
==WARNING== No kernels were profiled.
probe_cost:+/-5%: [error] no threshold on wall-clock cost can detect a probe that
collected nothing: runs with and without measurements are on the same side of it
```

It wrote a 7.5 MB report file. It printed a steady-state mean and a throughput.
Its number is **0.096%** from the unprofiled baseline and sits **between** the
two runs that did collect data (81.48 and 84.15 ms). The only signal was one
`==WARNING==` line on stderr, between two lines that looked entirely normal.

That last point is the one worth carrying: **no threshold on the timings can
detect this**, because the failed run is not an outlier — it is in the middle.
The test for it is parametrised over 1%, 2%, 5% and 10% deliberately, so that
the claim is about detection being impossible rather than about one badly
chosen constant.

## 3. The control: the instrument is precise

Before any of the below is read as "Nsight Compute's durations are unreliable"
— they are not. Group the per-launch durations by `(name, grid, block)`:

```console
$ probecost claims | grep -A1 "the control"
the control: is the instrument itself precise?
  grouped by (name, grid, block): median CV 0.229%, 7/9 groups under 1%
```

Eight launches of one CUTLASS GEMM at one geometry: 352,192 / 352,288 / 353,024
/ 353,216 / 353,312 / 353,376 / 353,408 / 352,608 ns. **A CV of 0.13%.** The
probe is a good ruler. The problem is what people measure with it.

## 4. The kernel name is the wrong key

Take those same eight launches of `magma_sgemmEx_kernel` and group them by name
only, as every profiler summary does:

| grid | duration (ns) |
|---|---|
| (6, 29, 1) | 70,720 |
| (6, 29, 1) | 71,232 |
| (6, 29, 1) | **245,184** |
| (18, 29, 1) | 192,896 |
| (18, 29, 1) | 194,496 |
| (24, 29, 1) | 238,272 |
| (29, 29, 6) | 385,536 |
| (29, 29, 6) | 387,168 |

**5.47× spread in eight launches**, across four geometries. And look at rows
one to three: **identical grid, identical block, 3.47× apart.** A GEMM's `K`
does not appear in the launch geometry, so conditioning on the geometry is
*also* not enough — there is no key available in a profiler summary that would
make that row a quantity.

Over whole runs, from the other instrument, on both workloads:

```console
$ probecost spread
depth_anything_v2: [error] 29 of 41 kernel names (>=10 launches) span 2x or more
within the name; median CV 35.8%, worst 1044x over 680 launches
flownet2: [error] 54 of 94 kernel names (>=10 launches) span 2x or more within the
name; median CV 33.6%, worst 2148x over 12338 launches
```

Two workloads that share a node and a harness and nothing else — one inference,
one training; 14,789 launches against 54,380. **Over half the names in each
span 2× or more, and the median CV lands within two points of each other.**

## 5. The short name fails in *both* directions

```console
$ probecost names
depth_anything_v2:Kernel2: [error] 17 different kernels share this short name; their
medians span 127.1x ([480, 39, 12, 51, 28, 58, 12, 37, 23, 17, 20, 19, 3, 16, 2, 4, 2]
launches respectively)
flownet2:Kernel: [error] 10 different kernels share this short name; their medians
span 194.6x
```

`ncu -k Kernel2` would profile seventeen different kernels as if they were one.
And in §2 above, `ncu -k cutlass_80_simt_sgemm` profiled zero. **The same
naming scheme under-matched and over-matched in one dataset**, and only the
fully mangled signature is unique.

This one is enforced by the package's own loader — a lookup that matches two
rows raises rather than returning the first:

```console
$ probecost compare cunn_SoftMaxForwardReg
DataError: 'cunn_SoftMaxForwardReg' matches 2 rows in depth_anything_v2; a per-name
lookup that matches more than one name is the bug this package is about
```

Those two rows: n=240 at a median of 690 µs, and n=12 at **6,070 µs**. Nearly
nine times apart, same short name.

## 6. Putting the two instruments in one table, and being refused

```console
$ probecost compare magma_sgemmEx_kernel
compare:magma_sgemmEx_kernel: [error] nsight-compute took the first 8 matching launches
and nsight-systems took all 2376; those are different sets of launches spanning 5.5x
and 85.5x internally, so their medians are not two estimates of one quantity
  nsight-compute   n=8      median=     216,384 ns  internal spread    5.5x  (the first N...)
  nsight-systems   n=2376   median=     216,992 ns  internal spread   85.5x  (every launch...)
  ratio if you did it anyway: 0.997x
  the two populations differ in size by 297x
```

Ignore the refusal across all four unambiguous kernels and the two profilers
appear to disagree by **−22% to +29%** — in both directions, so not a
calibration offset you could subtract. That reading is available and wrong.
**It is two different sets of launches being averaged**, and the repository says
so rather than reporting the ratios as a finding. `tests/test_claims.py`
asserts the negative claim, so an edit that promotes it fails the build.

---

## The device: `instrument`

Every number declares three things, and the third is the one nobody expects:

| field | why it is required |
|---|---|
| `instrument` | `None` means no probe — an explicit fact, not a missing field |
| `sampling_rule` | **what set of events the probe looked at.** `ncu -c N` takes a *prefix* of the matching launches; `nsys` takes all of them |
| `collected` | whether anything came back. `None` means nobody checked, and `None` is not `True` |

Comparison is gated, not schema-checked:

```python
>>> assert_comparable(ncu_median, nsys_median)
InstrumentError: ... comparing them requires a bound on what the probe does to this
measurement, and none was given
```

`probe_effect_bound` is keyword-only with no permissive default. And a bound
**cannot** repair a sampling-rule mismatch — different populations are refused
even with one, because no probe correction turns one set of launches into
another.

Two things it stops looking alike:

> **"this kernel takes 216 µs"** / **"this kernel took 216 µs under one probe's sampling rule"**

and, inside that:

> **"the probe measured it and it was fast"** / **"the probe measured nothing"**

---

## Using it

```console
$ probecost rules      # what each sampling rule means
$ probecost show       # what is in data/
$ probecost cost       # what the probe cost, per run
$ probecost empty      # runs that reported a number and measured nothing
$ probecost names      # short names covering more than one kernel
$ probecost spread     # how wide a per-kernel-name row really is
$ probecost compare <kernel>   # try to compare two instruments, get refused
$ probecost check      # everything, over the committed data
$ probecost claims     # this repository's findings, recomputed
```

`--json` prints the structure; `--fail-on error|warn` sets the exit code.
`probecost check` is deliberately **loud and exits 0**: every line is a property
of the committed data, so a non-zero exit would mean this repository never
builds. The gates that must stay green live in `tests/`.

## What is in `data/`

| file | what |
|---|---|
| `probe_cost.json` | 5 runs of one inference loop: unprofiled (760 images) plus four probe configurations |
| `launch_durations.json` | 40 individual launches with their geometry, exported from five `.ncu-rep` files |
| `name_summaries.json` | per-kernel-**name** rows from Nsight Systems, two workloads, 143 rows over 69,169 launches |

Card model, driver, CPU model and the measurements are published. The host, the
bus id, the process ids and the local paths are not, and `.gitignore` keeps
`.ncu-rep` / `.nsys-rep` out — a raw report is hundreds of megabytes and carries
all four.

**Sample sizes are stated, not footnoted.** The unprofiled baseline has 760
images in its steady window; **every probed cell has six or fewer**. `probecost
cost` prints that next to each ratio, so the 2.3% figures are visibly
unresolvable — and the 179×, the empty run and everything in §4–6 do not depend
on it.

## Tests

107, pure CPU, no dependencies beyond `pytest`. Each finding was verified by
fault injection — claim the empty run collected something, move it away from the
baseline, give the unfiltered run more passes, drop the sampling rule, flatten
the GEMM's same-geometry variation — and each makes a named test fail.

`tests/test_ci_runs.py` executes every `run:` block in the workflow under
`bash -e`, via PATH shims so the YAML runs verbatim. Three repositories in this
series shipped a CI that was red on every push while `pytest` and `ruff` stayed
green, because the defect was in the YAML and every test was in Python.

## What went wrong on the way here

Two of these changed a published number, and both were caught by running the
thing rather than by reading it.

1. **A crude parser invented a 1273× finding.** The first `_short_name` split on
   the `<` of `<unnamed>`, which collapsed seven unrelated kernels onto the
   token `native` and reported a headline collision that does not exist. The
   real worst case is `Kernel2` at 127×.
   `tests/test_population.py::test_no_collision_key_is_a_namespace` keeps it
   fixed, and a CI step checks it by name.
2. **The sample-size warning called a 179× effect noise.** The phrasing was
   unconditional and printed *"a 17845.74% difference over 3 samples is not a
   difference"*, which is false. One sentence like that discredits every other
   warning in the output. The wording is now conditional and
   `test_the_sample_size_note_does_not_call_a_179x_effect_noise` pins it.
3. **The loader accepted `sampling_rule: null`.** It checked for the key being
   absent, so a key present with a null value passed and blew up later with an
   `AttributeError`. The fault-injection pass found it by setting null instead
   of deleting. All three spellings of absence are now parametrised.

## Licence

MIT.
