# Refusing the subtraction

The comparison a person would make: same kernel, same machine, same workload,
one number from each profiler. It is one cell of a spreadsheet away at all
times, and `probecost` exists partly to refuse it.

## What the refusal looks like

```console
$ probecost compare magma_sgemmEx_kernel
compare:magma_sgemmEx_kernel: [error] nsight-compute took the first 8 matching launches
and nsight-systems took all 2376; those are different sets of launches spanning 5.5x
and 85.5x internally, so their medians are not two estimates of one quantity
  nsight-compute   n=8      median=     216,384 ns  internal spread    5.5x  (first-n-matching)
  nsight-systems   n=2376   median=     216,992 ns  internal spread   85.5x  (every-launch)
  ratio if you did it anyway: 0.997x
  the two populations differ in size by 297x
```

Note what the refusal does **not** do: it does not hide the numbers. Both
medians, both sample sizes, both internal spreads and the ratio come back on
the exception's `detail`, so they can be printed next to the refusal. A guard
that withholds the evidence for its own verdict is asking to be worked around.

## Why a bound cannot fix it

The device's comparison gate has an escape hatch, and it deliberately does not
open this door:

```python
def assert_comparable(a, b, *, probe_effect_bound=None):
    if a.sampling_rule != b.sampling_rule:
        raise InstrumentError(...)          # <- before the bound is consulted
    if a.instrument == b.instrument:
        return
    if probe_effect_bound is None:
        raise InstrumentError(...)
```

Two different instruments *can* be compared, if the caller supplies a bound on
what the probe does to this measurement. Two different **sampling rules**
cannot, bound or no bound:

```python
def test_different_sampling_rules_are_never_comparable():
    """Not even with a bound. A bound on the probe cannot repair a population."""
    with pytest.raises(InstrumentError, match="different sets of launches"):
        assert_comparable(m("a", "ncu", PREFIX), m("b", "nsys", EXHAUSTIVE),
                          probe_effect_bound=0.01)
```

A correction for the probe's overhead is a statement about *one* measurement
being shifted. It cannot turn the first 8 launches of a kernel into all 2,376
of them. Those are different quantities, and no coefficient relates them.

`probe_effect_bound` is keyword-only with no permissive default, so a caller who
never bounded anything cannot get through by omission — and there is a test that
passing it positionally raises `TypeError`.

## The reading that is available and wrong

Ignore the refusal for all four unambiguous kernels and here is what comes out:

| kernel | ncu median | nsys median | ratio |
|---|---|---|---|
| `magma_sgemmEx_kernel` | 216,384 | 216,992 | **0.997×** |
| `cutlass_80_simt_sgemm_64x64_8x5` | 353,120 | 455,536 | **0.775×** |
| `softmax_warp_forward` | 361,216 | 394,112 | **0.917×** |
| `upsample_bicubic2d_out_frame` | 2,370,896 | 1,897,118 | **1.250×** |

Read as instrument disagreement this is a striking result: **−22% to +25%, and
the sign changes.** So it is not a calibration offset anyone could subtract, and
the first row agreeing to 0.3% makes it look even more like a per-kernel
pathology in one of the profilers.

**It is none of those things.** The ncu column is the first 8 launches of that
kernel in one image's forward pass; the nsys column is every launch of it across
the whole run. The magma row agrees to 0.3% by coincidence — its 8-launch prefix
happens to have a similar shape mixture to the full 2,376 — and the upsample row
differs by 25% because 4 launches of a 39-launch population is not a sample of
it.

So the repository reports the refusal and records the ratios as *what the
refusal was protecting against*, not as a finding:

```python
def test_the_apparent_disagreement_is_not_claimed_to_be_a_calibration_error():
    c = analysis.the_comparison_that_gets_refused()
    assert "is not one" in c["reading"]
    assert "different sets" in c["reading"]
```

An edit that promotes those ratios to a headline fails the build.

## The one comparison that never reaches the gate

`cunn_SoftMaxForwardReg` is absent from the table above, and the reason is one
step earlier: its short name matches **two** rows in the summary table, so there
is no single nsys median to compare against. The lookup refuses before the
instrument gate is consulted.

Those two rows are n=240 at 690 µs and n=12 at 6,070 µs. If the lookup had
returned the first match — which is the natural, convenient implementation — the
comparison would have proceeded, been refused for the right reason, and reported
a ratio computed against the wrong row.

**Two guards firing in the correct order is not redundancy.** The name problem
and the population problem are different, and collapsing them would have hidden
one behind the other.

## What the whole thing amounts to

Three fields on every number:

| field | the question it answers |
|---|---|
| `instrument` | what was attached to the machine |
| `sampling_rule` | what set of events it looked at |
| `collected` | whether anything came back |

None of the three is inferable from the duration column, and all three change
what the duration means. The third one is not even inferable from the *timings
of the whole run* — see
[the-probe-that-measured-nothing.md](the-probe-that-measured-nothing.md).
