"""校准曲线精简 API 端到端测试：每个用例使用独立临时 SQLite 库。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DB_PATH", str(tmp_path / "test.db"))
    from app.main import app  # 环境变量设置后再导入

    with TestClient(app) as c:
        yield c


def make_observation(client, ts, bias):
    """登记一条偏差恰为 bias 的观测：bias = reference − 读数中位数。"""
    value = 100.0 - bias
    resp = client.post(
        "/observations",
        json={"timestamp": ts, "reference": 100.0, "tolerance": 0.5,
              "readings": [value, value, value]},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def simplify(client, max_error):
    return client.post("/calibration-curves/simplify", json={"max_error": max_error})


# 2026-09-18 08:00~12:00（整点）的 Unix 秒
T08, T09, T10, T11, T12 = (1789718400.0, 1789722000.0, 1789725600.0, 1789729200.0, 1789732800.0)


@pytest.fixture()
def five_points(client):
    """bias 依次为 1.0、1.75、2.0、1.5、2.0 的五条整点观测。"""
    return [make_observation(client, ts, bias)
            for ts, bias in zip((T08, T09, T10, T11, T12), (1.0, 1.75, 2.0, 1.5, 2.0))]


# ---------- 基本精简 ----------

def test_simplify_basic_drops_points_within_tolerance(client, five_points):
    resp = simplify(client, 0.4)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["max_error"] == pytest.approx(0.4)
    # 段 (08:00→12:00) 在 09:00/10:00 偏差 0.5 超限；最少保留 4 点：08:00、09:00、11:00、12:00
    assert [p["observation_id"] for p in body["kept"]] == [
        five_points[0]["id"], five_points[1]["id"], five_points[3]["id"], five_points[4]["id"]
    ]
    assert [p["timestamp"] for p in body["kept"]] == [T08, T09, T11, T12]
    assert [p["bias"] for p in body["kept"]] == [1.0, 1.75, 1.5, 2.0]
    assert body["kept"][0]["timestamp_iso"] == "2026-09-18T08:00:00+00:00"
    # 实际最大误差在 10:00：段 (09:00, 1.75)→(11:00, 1.5) 的插值 1.625 与原偏差 2.0 之差
    assert body["actual_max_error"] == pytest.approx(0.375)
    assert body["actual_max_error_at"] == pytest.approx(T10)
    assert body["actual_max_error_at_iso"] == "2026-09-18T10:00:00+00:00"


def test_simplify_generous_error_keeps_only_endpoints(client, five_points):
    body = simplify(client, 0.5).json()
    # 首尾连线在 09:00/10:00 的偏差恰为 0.5，不超上界 → 只留首尾
    assert [p["observation_id"] for p in body["kept"]] == [five_points[0]["id"], five_points[4]["id"]]
    assert body["actual_max_error"] == pytest.approx(0.5)
    assert body["actual_max_error_at"] == pytest.approx(T09)  # 09:00 与 10:00 同为 0.5，取最早


def test_simplify_zero_max_error_keeps_every_point(client, five_points):
    body = simplify(client, 0.0).json()
    assert [p["observation_id"] for p in body["kept"]] == [o["id"] for o in five_points]
    assert body["actual_max_error"] == pytest.approx(0.0)
    assert body["actual_max_error_at"] == pytest.approx(T08)


def test_simplify_two_observations_keeps_both(client):
    o1 = make_observation(client, 1000.0, 1.0)
    o2 = make_observation(client, 2000.0, 3.0)
    body = simplify(client, 0.0).json()
    assert [p["observation_id"] for p in body["kept"]] == [o1["id"], o2["id"]]
    assert body["actual_max_error"] == pytest.approx(0.0)
    assert body["actual_max_error_at"] == pytest.approx(1000.0)


def test_simplify_lexicographically_earliest_among_ties(client):
    # 拱形折线：最少 3 点方案有 (0,1,4)/(0,2,4)/(0,3,4)，字典序最早为 (0,1,4)
    obs = [make_observation(client, float(t), bias)
           for t, bias in enumerate((0.0, 0.3, 0.6, 0.3, 0.0))]
    body = simplify(client, 0.5).json()
    assert [p["observation_id"] for p in body["kept"]] == [obs[0]["id"], obs[1]["id"], obs[4]["id"]]
    assert body["actual_max_error"] == pytest.approx(0.4)
    assert body["actual_max_error_at"] == pytest.approx(2.0)


def test_simplify_uses_current_bias_after_revision(client):
    make_observation(client, T08, 1.0)
    make_observation(client, T09, 1.75)
    o3 = make_observation(client, T10, 100.0)   # 偏差极大，随后修订回 2.0
    make_observation(client, T11, 1.5)
    make_observation(client, T12, 2.0)
    resp = client.put(
        f"/observations/{o3['id']}",
        json={"timestamp": T10, "reference": 100.0, "tolerance": 0.5,
              "readings": [98.0, 98.0, 98.0], "revision": 1},
    )
    assert resp.status_code == 200, resp.text
    body = simplify(client, 0.4).json()
    # 按修订后的当前偏差精简，结果与无修订场景一致
    assert [p["timestamp"] for p in body["kept"]] == [T08, T09, T11, T12]
    assert body["actual_max_error"] == pytest.approx(0.375)


# ---------- 冲突与参数校验 ----------

def test_simplify_without_any_observation_conflicts(client):
    resp = simplify(client, 1.0)
    assert resp.status_code == 409
    assert "at least 2 observations" in resp.json()["detail"]


def test_simplify_with_single_observation_conflicts(client):
    make_observation(client, 1000.0, 1.0)
    assert simplify(client, 1.0).status_code == 409


def test_simplify_rejects_invalid_max_error(client):
    make_observation(client, 1000.0, 1.0)
    make_observation(client, 2000.0, 3.0)
    assert simplify(client, -0.5).status_code == 422          # 负数
    assert simplify(client, "much").status_code == 422        # 非数值
    assert client.post("/calibration-curves/simplify", json={}).status_code == 422  # 缺字段
    # NaN 与 ±Infinity 不是有限数
    for token in ("NaN", "Infinity", "-Infinity"):
        resp = client.post(
            "/calibration-curves/simplify",
            content=f'{{"max_error": {token}}}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, token


# ---------- 只读保证 ----------

def test_simplify_is_read_only(client):
    o1 = make_observation(client, T08, 1.0)
    make_observation(client, T09, 1.75)
    make_observation(client, T10, 2.0)
    client.put(
        f"/observations/{o1['id']}",
        json={"timestamp": T08, "reference": 100.0, "tolerance": 0.5,
              "readings": [99.1, 99.1, 99.1], "revision": 1},
    )
    before = (
        client.get("/observations").json(),
        client.get("/intervals").json(),
        client.get("/observations/1/revisions").json(),
    )
    # 成功与失败（422）的精简请求都不得改动任何现状数据
    assert simplify(client, 0.4).status_code == 200
    assert simplify(client, -1.0).status_code == 422
    after = (
        client.get("/observations").json(),
        client.get("/intervals").json(),
        client.get("/observations/1/revisions").json(),
    )
    assert after == before
