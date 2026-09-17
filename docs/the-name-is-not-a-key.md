# The name is not a key

Every profiler summary is one row per kernel **name**. This chapter is about
what that row is an average of.

## First, the control

Before anything below is read as "Nsight Compute's durations are unreliable" —
they are not, and the data says so. Group the per-launch durations by
`(name, grid, block)` and look inside a group.

Eight launches of one CUTLASS GEMM, all at grid `(29, 1, 6)`, block `(128, 1, 1)`:

```
352,192   352,288   353,024   353,216   353,312   353,376   353,408   352,608 ns
```

**CV 0.13%.** Across all nine repeated geometries in the dataset the median CV
is **0.229%**, and seven of the nine are under 1%.

The probe is a good ruler. Everything below is about what people measure with
it.

## Group by name and the ruler stops mattering

The same eight launches of `magma_sgemmEx_kernel`, in launch order:

| grid | block | duration (ns) |
|---|---|---|
| (18, 29, 1) | (8, 16, 1) | 194,496 |
| (29, 29, 6) | (8, 16, 1) | 387,168 |
| (6, 29, 1) | (8, 16, 1) | 70,720 |
| (24, 29, 1) | (8, 16, 1) | 238,272 |
| (6, 29, 1) | (8, 16, 1) | **245,184** |
| (18, 29, 1) | (8, 16, 1) | 192,896 |
| (29, 29, 6) | (8, 16, 1) | 385,536 |
| (6, 29, 1) | (8, 16, 1) | 71,232 |

**5.47× spread in eight launches**, across four geometries. A per-name row over
these would read "≈216 µs" and describe none of them.

Now look at rows three, five and eight. **Same grid. Same block. 70,720 /
245,184 / 71,232 — a factor of 3.47.**

A GEMM's `K` is not in the launch geometry. Grid and block tell you how the
output is tiled; they say nothing about how much arithmetic each tile does. So:

> **Conditioning on the launch geometry is also not enough, and there is no key
> available in a profiler summary that would be.**

That is the part that makes this a methodology problem rather than a "group by
shape instead" tip.

## Whole runs, the other instrument, two workloads

The launch-level view above is 40 launches. Here is the per-name view over
complete runs, from Nsight Systems:

| | names | ≥10 launches | span ≥2× | median CV | worst | launches |
|---|---|---|---|---|---|---|
| Depth-Anything-V2 (inference) | 49 | 41 | **29** | 35.8% | 1044× | 14,789 |
| FlowNet2 (training) | 94 | 94 | **54** | 33.6% | 2148× | 54,380 |

Two workloads that share a node and a harness and nothing else — one inference
and one training, different models, 3.7× different launch counts. **Over half
the names in each span a factor of two or more within the name, and the median
CV lands within two points of each other.**

The test asserts the replication rather than the numbers:

```python
assert abs(da["median_cv_pct"] - of["median_cv_pct"]) < 3.0
```

## The short name fails in both directions

There is a level above the key problem: the name people actually type is not
even unique.

```console
$ probecost names
depth_anything_v2:Kernel2: [error] 17 different kernels share this short name;
their medians span 127.1x
flownet2:Kernel: [error] 10 different kernels share this short name; their
medians span 194.6x
```

`ncu -k Kernel2` would profile seventeen different CUTLASS kernels as if they
were one thing. Meanwhile, in the cost table, `ncu -k cutlass_80_simt_sgemm`
profiled **zero**.

**Same naming scheme, one dataset, both failure directions.** And the fix for
both is the same: only the fully mangled signature is unique.

Counted across the two workloads: 5 of 22 short names in the inference run and
11 of 58 in the training run cover more than one kernel.

The package enforces this on itself — a lookup that matches two rows raises
rather than quietly returning the first:

```console
$ probecost compare cunn_SoftMaxForwardReg
DataError: 'cunn_SoftMaxForwardReg' matches 2 rows in depth_anything_v2; a
per-name lookup that matches more than one name is the bug this package is about
```

Those two rows differ only in a template argument buried in the mangled name,
and they are **n=240 at a median of 690 µs** against **n=12 at 6,070 µs**.
Nearly nine times apart.

## A false finding this chapter used to contain

The first version of `_short_name` took everything before the first `<`. That
made `void at::native::<unnamed>::cunn_SoftMaxForwardReg<...>` collapse to the
token **`native`**, because `<unnamed>` is an anonymous-namespace marker and not
a template argument.

Result: seven unrelated kernels grouped under `native`, and a headline
collision of **1273×** that does not exist. It was in the first run of
`probecost claims` and it was wrong.

The real worst case is `Kernel2` at 127×, which is a better finding anyway
because it is one *family* of kernels with one *shared* generic name, rather
than an artifact of a namespace separator.

The guard is by name, and there is a CI step for it too:

```python
def test_no_collision_key_is_a_namespace():
    banned = {"native", "at", "detail", "cuda", "unnamed", "cudnn", "cutlass"}
    for wl, v in population.ambiguous_short_names().items():
        keys = {c["short_name"] for c in v["collisions"]}
        assert not (keys & banned), (wl, keys & banned)
```

> **A crude parser inventing a finding is worse than no parser.** The same
> lesson as a checker whose failures are its own: one of those and the reader
> stops believing the output.
