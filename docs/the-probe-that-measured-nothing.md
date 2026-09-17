# The probe that measured nothing

## The cost table

One inference loop, five times. Only the probe configuration changes.

| run | probe | launches replayed | passes each | ms/image | vs unprofiled | n |
|---|---|---|---|---|---|---|
| `none` | — | 0 | — | **83.41** | 1.000× | **760** |
| `ncu_filter_magma` | ncu | 8 | 21 | 81.48 | 0.977× | 6 |
| `ncu_filter_cutlass_short_name` | ncu | **0** | 21 | 83.49 | 1.001× | 6 |
| `ncu_filter_cutlass_full_name` | ncu | 8 | 21 | 84.15 | 1.009× | 6 |
| `ncu_no_filter` | ncu | 420 | 9 | **14968.54** | **179.5×** | 3 |

Two things to read off it, in order.

## The cost is not a property of the probe

Same Nsight Compute, same workload, same machine, **184× between the cheapest
and the dearest configuration**. With a kernel-name filter it is
indistinguishable from having no profiler attached at all.

The obvious explanation — "the expensive run collected heavier metrics" — is
available and the data rules it out:

```python
def test_the_expensive_run_used_fewer_passes_than_the_cheap_ones():
    """Rules out "the metric set was heavier" as the explanation."""
    heavy = t.by_id("ncu_no_filter")
    light = t.by_id("ncu_filter_magma")
    assert heavy.replay_passes < light.replay_passes
```

**9 passes against 21.** The expensive run used the *basic* metric set and was
179× slower; the cheap runs used the *detailed* one and were free. The cost
follows how many launches get replayed, and nothing else.

Which means both of these sentences are true of "we ran it under Nsight
Compute", and which one applies depends on a command-line flag that appears
nowhere in the output:

* the timings are inflated beyond any use;
* the timings are within a percent of unprofiled.

## The run that collected nothing

Row three used the short kernel name `cutlass_80_simt_sgemm`. The kernel's
actual name is:

```
void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)
```

Nothing matched. Here is everything that run produced:

* a **7.5 MB** `.ncu-rep` file;
* `# 单张均值 83.49 ms   吞吐 11.977 张/s` — a steady-state mean and a
  throughput, in the harness's normal format, in the normal place;
* one line on stderr: `==WARNING== No kernels were profiled.`

and here is where its number landed:

```
ncu_filter_magma              81.48 ms
ncu_filter_cutlass_short      83.49 ms   <- collected nothing
none (no probe at all)        83.41 ms
ncu_filter_cutlass_full       84.15 ms
```

**0.096% from the unprofiled baseline, and between the two runs that worked.**

### Why no threshold can catch it

The tempting fix is a rule: "if the probed run is within X% of unprofiled,
suspect the probe did nothing." It cannot work, and the reason is in the
ordering above — the failed run is not an outlier, it is in the *middle*.

So the check is parametrised deliberately:

```python
@pytest.mark.parametrize("tol", [1.0, 2.0, 5.0, 10.0])
def test_no_tolerance_separates_collected_from_not_collected(tol):
    r = cost.indistinguishable_from_baseline(tolerance_pct=tol)
    assert r["both_outcomes_inside"] is True, (tol, r["inside"])
```

At every plausible tolerance, runs that collected data and the run that did not
are on the same side. That makes it a **detection impossibility**, not a badly
chosen constant — and the distinction matters, because the first has a fix
(record whether anything came back) and the second does not.

That fix is the device's third field:

```python
def assert_collected(label, collected):
    if collected is None:
        raise NotCollectedError(
            f"{label}: whether this run collected anything was never recorded; "
            f"the timings cannot tell you ..."
        )
```

`None` raises. Not because nothing came back, but because **nobody wrote down
whether anything came back** — and this dataset contains the case where the
answer was no and every printed number looked normal.

### And the verdict is not a constant either

One more fault injection, on the conclusion rather than the code:

```python
def test_the_verdict_tracks_the_data_and_is_not_a_constant(tmp_path):
    # move the empty run to 9999 ms
    r = cost.indistinguishable_from_baseline(t, tolerance_pct=5.0)
    assert r["both_outcomes_inside"] is False
    assert "property of this tolerance" in r["verdict"]
```

If the empty run *were* an outlier the function has to say so. A check that
returns "undetectable" whatever the data says is not measuring anything.

## What this table cannot carry

The baseline has **760 images** in its steady window. Every probed cell has
**six or fewer**. So:

* the **179×** does not depend on the sample size — no three-sample window
  produces a 17,846% difference by chance;
* the **0.096%** and the **2.3%** are unresolvable, and `probecost cost` says so
  next to each one;
* nothing in the "collected nothing" finding depends on the sample size at all,
  because it is a statement about what the run *printed*.

That distinction is why the warning's wording is conditional. The first version
was not, and printed:

> a 17845.74% difference over 3 samples is not a difference

which is false, and which — printed once — teaches the reader to skip every
other warning in the output. There is a test named after it.
