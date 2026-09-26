"""校准曲线精简纯算法单测：最少保留点、字典序并列、逐折点误差核对。"""
from __future__ import annotations

import itertools

import pytest

from app.simplification import simplify_curve


# ---------- 基本形态 ----------

def test_two_points_always_kept_with_zero_error():
    kept, max_dev, max_dev_at = simplify_curve([(1000.0, 1.0), (2000.0, 3.0)], 0.0)
    assert kept == [0, 1]
    assert max_dev == 0.0
    assert max_dev_at == 1000.0  # 偏差处处为 0，最早发生时点即首折点


def test_collinear_points_reduce_to_endpoints():
    points = [(float(t), 2.0 * t) for t in range(6)]
    kept, max_dev, _ = simplify_curve(points, 0.0)
    assert kept == [0, 5]
    assert max_dev == 0.0


def test_middle_point_within_tolerance_is_dropped():
    kept, max_dev, max_dev_at = simplify_curve([(0.0, 0.0), (1.0, 0.4), (2.0, 0.0)], 0.5)
    assert kept == [0, 2]
    assert max_dev == pytest.approx(0.4)
    assert max_dev_at == 1.0


def test_middle_point_beyond_tolerance_is_kept():
    kept, max_dev, _ = simplify_curve([(0.0, 0.0), (1.0, 0.6), (2.0, 0.0)], 0.5)
    assert kept == [0, 1, 2]
    assert max_dev == 0.0  # 全部保留：精简曲线与原曲线重合


def test_error_is_checked_at_every_crossed_breakpoint_not_just_endpoints():
    # 端点偏差为 0 但中间折点偏差 10：只查端点会误判为可精简
    kept, _, _ = simplify_curve([(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)], 1.0)
    assert kept == [0, 1, 2]


# ---------- 最少点数与并列 ----------

def test_minimal_count_with_lexicographically_earliest_tie_break():
    # 拱形折线：最少 3 点的方案有 [0,1,4] / [0,2,4] / [0,3,4]，字典序最早为 [0,1,4]
    points = [(0.0, 0.0), (1.0, 0.3), (2.0, 0.6), (3.0, 0.3), (4.0, 0.0)]
    kept, max_dev, max_dev_at = simplify_curve(points, 0.5)
    assert kept == [0, 1, 4]
    assert max_dev == pytest.approx(0.4)
    assert max_dev_at == 2.0


def test_non_contiguous_feasible_edges_are_explored():
    # 段 (0,2) 不合法但更长的段 (0,3) 合法：不能“遇到不合法就停止扩展”
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5), (3.0, 1.5)]
    kept, max_dev, max_dev_at = simplify_curve(points, 0.6)
    assert kept == [0, 3]
    assert max_dev == pytest.approx(0.5)
    assert max_dev_at == 1.0  # t=1 与 t=2 偏差同为 0.5，取最早


def test_zero_max_error_keeps_all_non_collinear_points():
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
    kept, _, _ = simplify_curve(points, 0.0)
    assert kept == [0, 1, 2, 3]


def test_huge_max_error_keeps_only_endpoints():
    points = [(0.0, 0.0), (1.0, 100.0), (2.0, -50.0), (3.0, 7.0)]
    kept, max_dev, max_dev_at = simplify_curve(points, 1e9)
    assert kept == [0, 3]
    # 实际误差仍按真实折点偏差报告，与宽松的上界无关
    assert max_dev == pytest.approx(100.0 - 7.0 / 3.0)
    assert max_dev_at == 1.0


# ---------- 入参校验 ----------

def test_fewer_than_two_points_rejected():
    with pytest.raises(ValueError):
        simplify_curve([], 1.0)
    with pytest.raises(ValueError):
        simplify_curve([(1.0, 2.0)], 1.0)


@pytest.mark.parametrize("bad", [-1.0, -0.0 - 1e-9, float("nan"), float("inf"), float("-inf")])
def test_invalid_max_error_rejected(bad):
    with pytest.raises(ValueError):
        simplify_curve([(0.0, 0.0), (1.0, 1.0)], bad)


# ---------- 与穷举参考实现交叉验证 ----------

def _brute_force_kept(points: list[tuple[float, float]], max_error: float) -> list[int]:
    """穷举所有含首尾的子序列，返回 (点数最少, 并列字典序最早) 的下标序列。"""
    n = len(points)

    def feasible(idx: list[int]) -> bool:
        for a, b in zip(idx, idx[1:]):
            t_a, b_a = points[a]
            t_b, b_b = points[b]
            slope = (b_b - b_a) / (t_b - t_a)
            for k in range(a + 1, b):
                t_k, b_k = points[k]
                if abs(b_a + slope * (t_k - t_a) - b_k) > max_error:
                    return False
        return True

    best = None
    for r in range(n - 1):
        for combo in itertools.combinations(range(1, n - 1), r):
            idx = [0, *combo, n - 1]
            if feasible(idx) and (best is None or (len(idx), idx) < (len(best), best)):
                best = idx
    return best


@pytest.mark.parametrize("seed", range(30))
def test_matches_brute_force_on_jagged_curves(seed):
    # 锯齿小曲线 + 多档误差上界：保留序列须与穷举的全局最优完全一致
    import random

    rng = random.Random(seed)
    n = rng.randint(2, 8)
    ts = sorted(rng.sample(range(100), n))
    points = [(float(t), rng.uniform(-3.0, 3.0)) for t in ts]
    for max_error in (0.0, 0.5, 1.5, 4.0):
        kept, max_dev, max_dev_at = simplify_curve(points, max_error)
        assert kept == _brute_force_kept(points, max_error)
        assert max_dev <= max_error + 1e-9
        # 报告的实际误差与最早发生时点须与独立复算一致
        expect_dev, expect_at = 0.0, points[0][0]
        for a, b in zip(kept, kept[1:]):
            t_a, b_a = points[a]
            t_b, b_b = points[b]
            slope = (b_b - b_a) / (t_b - t_a)
            for k in range(a + 1, b):
                dev = abs(b_a + slope * (points[k][0] - t_a) - points[k][1])
                if dev > expect_dev:
                    expect_dev, expect_at = dev, points[k][0]
        assert max_dev == pytest.approx(expect_dev)
        assert max_dev_at == expect_at
