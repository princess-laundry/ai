import math

import pytest

from posture_ai.estimator import (
    AWAY,
    FHP,
    LEAN,
    NORMAL,
    UNKNOWN,
    Observation,
    PostureEstimator,
    Thresholds,
)
from posture_ai.features import BodySide, Point, compute_features, facing_vote, rotate_body, select_side


def body(head_ratio=0.0, torso_deg=0.0, facing=1, length=200.0, hip=(240.0, 520.0), conf=0.9):
    """원하는 head_ratio·몸통 각도를 정확히 갖는 측면 몸(이미지 좌표, y 아래)."""
    th = math.radians(torso_deg)
    u = (math.sin(th), math.cos(th))
    n = (math.cos(th), -math.sin(th))
    sh = (length * u[0], length * u[1])
    ear = (sh[0] + 0.42 * length * u[0] + head_ratio * length * n[0], sh[1] + 0.42 * length * u[1] + head_ratio * length * n[1])

    def img(p):
        return Point(hip[0] + facing * p[0], hip[1] - p[1], conf)

    return BodySide(img(ear), img(sh), img((0.0, 0.0)))


def nose_for(b: BodySide, facing=1):
    return Point(b.ear.x + facing * 40, b.ear.y, 0.9)


def run(est, frames, start=0, step=200):
    out = None
    for k, (b, nose) in enumerate(frames):
        out = est.update(Observation(start + k * step, True, right=b, nose=nose))
    return out


def steady(n, **kw):
    b = body(**kw)
    return [(b, nose_for(b, kw.get("facing", 1)))] * n


def test_features_recover_geometry_for_both_facings():
    for facing in (1, -1):
        f = compute_features(body(head_ratio=0.3, torso_deg=25, facing=facing), facing)
        assert f.head_ratio == pytest.approx(0.3)
        assert f.torso_angle == pytest.approx(25)
        assert f.length == pytest.approx(200)


def test_rigid_lean_does_not_look_like_forward_head():
    f = compute_features(body(head_ratio=0.0, torso_deg=30), 1)
    assert f.head_ratio == pytest.approx(0.0, abs=1e-9)
    assert f.neck_angle == pytest.approx(0.0, abs=1e-9)


def test_leaning_back_is_negative_not_lean():
    f = compute_features(body(torso_deg=-25), 1)
    assert f.torso_angle == pytest.approx(-25)


def test_roll_correction_restores_upright_geometry():
    b = body(head_ratio=0.1, torso_deg=10)
    tilted = rotate_body(b, -8.0)  # 폰이 8도 기울어 찍힌 화면
    f = compute_features(rotate_body(tilted, 8.0), 1)
    assert f.torso_angle == pytest.approx(10)
    assert f.head_ratio == pytest.approx(0.1)


def test_side_selection_prefers_higher_confidence_and_ignores_low():
    good, weak = body(conf=0.9), body(conf=0.6)
    assert select_side(weak, good) == (good, 1)
    assert select_side(body(conf=0.4), None) == (None, 0)


def test_facing_vote_uses_nose_then_visible_side():
    b = body(facing=-1)
    assert facing_vote(b, 1, nose_for(b, -1)) == (-1.0, 0.25)
    assert facing_vote(b, -1, None) == (0.0, -0.25)


def test_normal_then_forward_head_switches_after_window():
    est = PostureEstimator()
    assert run(est, steady(11)).state == NORMAL
    result = run(est, steady(5, head_ratio=0.5), start=2200)
    assert result.state == NORMAL  # 1초로는 70%에 못 미침
    result = run(est, steady(6, head_ratio=0.5), start=3200)
    assert result.state == FHP and result.head_forward is True and result.torso_lean is False


def test_brief_movement_is_ignored():
    est = PostureEstimator()
    run(est, steady(16))
    result = run(est, steady(3, head_ratio=0.6), start=3200)
    assert result.state == NORMAL


def test_lean_and_head_are_independent_attributes():
    est = PostureEstimator()
    result = run(est, steady(16, torso_deg=30))
    assert result.state == LEAN and result.head_forward is False
    est = PostureEstimator()
    result = run(est, steady(16, torso_deg=30, head_ratio=0.5))
    assert result.state == FHP and result.head_forward and result.torso_lean


def test_missing_person_becomes_away_and_occluded_unknown():
    est = PostureEstimator()
    run(est, steady(16))
    assert est.update(Observation(3200, False)).state == NORMAL  # 짧은 결측은 유지
    assert est.update(Observation(5200, False)).state == AWAY
    est2 = PostureEstimator()
    run(est2, steady(16))
    est2.update(Observation(3200, True))
    assert est2.update(Observation(5200, True)).state == UNKNOWN


def test_distance_change_requests_reset():
    est = PostureEstimator()
    run(est, steady(10))
    result = run(est, steady(5, length=300.0), start=2000)
    assert result.needs_reset and result.state == UNKNOWN and not result.framed


def test_unknown_until_first_full_window():
    est = PostureEstimator()
    assert run(est, steady(5)).state == UNKNOWN


def test_thresholds_from_server_config():
    t = Thresholds.from_server({"version": 2, "t1_ratio": 0.2, "t2_angle": 18.5, "stable_sec": 2.0, "analysis_fps": 5})
    assert (t.version, t.head_ratio, t.lean_degrees, t.stable_ms) == (2, 0.2, 18.5, 2000)
    with pytest.raises(ValueError):
        Thresholds(head_ratio=math.nan)
