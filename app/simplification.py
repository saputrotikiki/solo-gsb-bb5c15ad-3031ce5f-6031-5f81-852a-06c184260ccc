"""只读校准曲线精简：在误差上界内选出点数全局最少的原观测子序列。

纯计算模块，不触碰数据库：输入按时间升序的原观测折点 [(t, b)] 与有限非负
max_error，输出保留折点的下标序列、精简曲线与原曲线在完整覆盖范围内的实际
最大绝对偏差及其最早发生时点。

精简曲线只取原折点的子序列（首尾必选），相邻保留折点间线性插值。原曲线与精简
曲线都是分段线性的，两者之差在每个原折点区间内仍是线性函数，绝对偏差的峰值
必在原折点处取得——因此误差判定逐一核对被跨越的每个原折点即可，既不采样也
不只查端点。
"""
from __future__ import annotations

import math


def _feasible_successors(points: list[tuple[float, float]], i: int, max_error: float):
    """按升序产出所有满足误差上界的精简段终点下标 j > i。

    段 (i, j) 的插值直线过 (t_i, b_i) 与 (t_j, b_j)，斜率 s_j = (b_j − b_i)/(t_j − t_i)。
    被跨越的原折点 k（i < k < j）逐一施加约束 |b_i + s_j·(t_k − t_i) − b_k| ≤ max_error，
    即 s_j ∈ [L_k, U_k]（L_k/U_k 为折点 k 允许的斜率下/上界）。随 j 增大把每个新跨越
    折点的约束并入可行斜率区间 [lo, hi]：段 (i, j) 合法当且仅当 s_j 落在当前区间内
    （合法段集合不必连续，短段不合法不代表更长的段也不合法）。j = i + 1 时区间尚无
    约束，相邻段恒合法，保证解必然存在。
    """
    t_i, b_i = points[i]
    lo, hi = -math.inf, math.inf
    for j in range(i + 1, len(points)):
        t_j, b_j = points[j]
        slope = (b_j - b_i) / (t_j - t_i)
        if lo <= slope <= hi:
            yield j
        # 折点 j 成为后续更长段的被跨越折点：收缩可行斜率区间
        span = t_j - t_i
        lo = max(lo, (b_j - max_error - b_i) / span)
        hi = min(hi, (b_j + max_error - b_i) / span)


def simplify_curve(
    points: list[tuple[float, float]], max_error: float
) -> tuple[list[int], float, float]:
    """求保留点数全局最少的精简子序列。

    points 为按时间严格升序的原观测折点 [(t, b)]（至少 2 个）；max_error 为有限
    非负误差上界。首尾折点必保留；保留点数并列最少时取时点序列字典序最早者。
    返回 (保留下标序列, 实际最大误差, 实际最大误差最早发生时点)。

    两阶段求解（全程 O(n²)，不采样）：
    1. 自右向左在 DAG 上做最短路 DP：g[i] = 从折点 i 覆盖到末点所需的最少保留
       点数（含 i 与末点），g[i] = 1 + min{ g[j] : 段 (i, j) 合法 }；
    2. 自左向右贪心重建字典序最早序列：每一步取仍能达到最少点数的最早下一折点
       （合法且 g[j] = g[cur] − 1 的最小 j）。
    """
    if len(points) < 2:
        raise ValueError(f"need at least 2 points, got {len(points)}")
    if not max_error >= 0 or math.isinf(max_error):  # 同时拒绝负数、NaN 与 ±inf
        raise ValueError(f"max_error must be a finite non-negative number, got {max_error}")
    n = len(points)
    g = [0] * n
    g[n - 1] = 1
    for i in range(n - 2, -1, -1):
        g[i] = 1 + min(g[j] for j in _feasible_successors(points, i, max_error))
    kept = [0]
    while kept[-1] != n - 1:
        cur = kept[-1]
        kept.append(next(j for j in _feasible_successors(points, cur, max_error) if g[j] == g[cur] - 1))
    # 实际最大误差：逐折点核对（相邻原折点间偏差为线性，峰值必在折点处）；
    # 按时间升序扫描且严格大于才更新，最早发生时点自然保留
    max_dev = 0.0
    max_dev_at = points[0][0]
    for a, b in zip(kept, kept[1:]):
        t_a, b_a = points[a]
        t_b, b_b = points[b]
        slope = (b_b - b_a) / (t_b - t_a)
        for k in range(a + 1, b):
            t_k, b_k = points[k]
            dev = abs(b_a + slope * (t_k - t_a) - b_k)
            if dev > max_dev:
                max_dev = dev
                max_dev_at = t_k
    return kept, max_dev, max_dev_at
