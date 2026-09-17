"""What a per-kernel number is an average of."""

from __future__ import annotations

import pytest

from probecost import population
from probecost.instrument import InstrumentError
from probecost.population import _short_name


def test_the_instrument_is_precise_within_a_launch_shape():
    """The control. Without it the rest reads as "Nsight Compute is unreliable"."""
    w = population.within_shape()
    assert w["median_cv_pct"] < 1.0
    assert w["tight_under_1pct"] >= 7


def test_grouping_by_name_alone_opens_the_spread_up():
    a = population.across_shape()
    worst = a["worst"]
    assert worst["kernel"] == "magma_sgemmEx_kernel"
    assert worst["spread"] > 5.0
    assert worst["distinct_shapes"] > 1
    assert worst["n"] == 8


def test_identical_geometry_still_varies_for_the_gemm():
    """A grid and a block do not carry K, so geometry is not enough either."""
    same = population.same_shape_still_varies()
    assert len(same) == 1
    g = same[0]
    assert "magma" in g["kernel"]
    assert g["spread"] > 3.0


def test_over_half_the_names_span_a_factor_of_two_in_both_workloads():
    w = population.name_row_spread()
    for wl, v in w.items():
        frac = v["names_spanning_factor"] / v["names_considered"]
        assert frac > 0.5, (wl, frac)
        assert v["median_cv_pct"] > 30.0
        assert v["worst"]["spread"] > 500.0


def test_the_factor_is_a_parameter_not_a_constant():
    loose = population.name_row_spread(factor=1000.0)
    tight = population.name_row_spread(factor=1.1)
    for wl in loose:
        assert (loose[wl]["names_spanning_factor"]
                < tight[wl]["names_spanning_factor"])


def test_short_names_collide_in_both_workloads():
    amb = population.ambiguous_short_names()
    da = amb["depth_anything_v2"]
    assert da["colliding_short_names"] == 5
    worst = da["collisions"][0]
    assert worst["short_name"] == "Kernel2"
    assert worst["rows"] == 17
    assert worst["median_ratio"] > 100.0
    assert amb["flownet2"]["colliding_short_names"] == 11


def test_no_collision_key_is_a_namespace():
    """A named regression test for a false finding this package produced itself.

    The first ``_short_name`` split on the ``<`` of ``<unnamed>``, which
    collapsed seven unrelated kernels onto the token ``native`` and reported a
    headline collision of 1273x that does not exist. A crude parser inventing a
    finding is worse than no parser.
    """
    amb = population.ambiguous_short_names()
    banned = {"native", "at", "detail", "cuda", "unnamed", "cudnn", "cutlass"}
    for wl, v in amb.items():
        keys = {c["short_name"] for c in v["collisions"]}
        assert not (keys & banned), (wl, keys & banned)


@pytest.mark.parametrize("mangled,expected", [
    ("void at::native::<unnamed>::cunn_SoftMaxForwardReg<float, float>(T3 *)",
     "cunn_SoftMaxForwardReg"),
    ("void <unnamed>::softmax_warp_forward<float, float>(T2 *)",
     "softmax_warp_forward"),
    ("void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)",
     "Kernel2"),
    ("void magma_sgemmEx_kernel<float, float>(int, int)", "magma_sgemmEx_kernel"),
    ("void cudnn::engines_precompiled::nchwToNhwcKernel<float>(T1)",
     "nchwToNhwcKernel"),
    ("sm80_xmma_fprop_implicit_gemm_tf32f32_execute_kernel__5x_cudnn",
     "sm80_xmma_fprop_implicit_gemm_tf32f32_execute_kernel__5x_cudnn"),
    ("void at::native::(anonymous namespace)::foo_kernel<float>(int)", "foo_kernel"),
])
def test_short_name_extraction(mangled, expected):
    assert _short_name(mangled) == expected


def test_the_cross_instrument_comparison_is_refused():
    with pytest.raises(InstrumentError, match="different sets of launches"):
        population.compare_across_instruments("magma_sgemmEx_kernel",
                                              "magma_sgemmEx_kernel")


def test_the_refusal_carries_the_numbers_it_refused_to_subtract():
    """Refusing is not hiding: both medians come back on the exception."""
    try:
        population.compare_across_instruments("softmax_warp_forward",
                                              "softmax_warp_forward")
    except InstrumentError as exc:
        d = exc.detail
        assert d["a"]["n"] == 8 and d["b"]["n"] == 228
        assert d["population_size_ratio"] > 25
        assert 0.5 < d["ratio_if_you_did_it_anyway"] < 2.0
    else:
        raise AssertionError("the comparison was not refused")


def test_a_kernel_with_no_launches_is_an_error_not_an_empty_answer():
    with pytest.raises(InstrumentError, match="no launches"):
        population.compare_across_instruments("not_a_kernel", "magma_sgemmEx_kernel")
