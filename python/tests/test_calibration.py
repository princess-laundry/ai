import math

from posture_ai import synthetic
from posture_ai.calibration import DONE, MOVING, NOT_UPRIGHT, NOT_VISIBLE, CalibrationCollector
from posture_ai.estimator import FHP, NORMAL, Observation, PostureEstimator, Thresholds
from posture_ai.evaluate import calibrate_recording, simulate_app
from posture_ai.landmarks import observations
from posture_ai.windows import Recording

from test_estimator import body, nose_for


def calibrate(frame_for_time):
    est, collector = PostureEstimator(), CalibrationCollector()
    result = None
    for t in range(0, 5001, 200):
        result = collector.add(t, est.update(frame_for_time(t)))
    return result


def observe(t, b):
    return Observation(t, True, right=b, nose=nose_for(b))


def observe_pitch(t, b, pitch):
    return Observation(t, True, right=b, nose=nose_for(b, 1, pitch))


def test_calibration_records_head_pitch_and_judges_bowing_relative_to_it():
    usual = body()
    result = calibrate(lambda t: observe_pitch(t, usual, 32.0))   # 평소 고개를 조금 숙이는 사람
    assert result.status == DONE and abs(result.calibration.head_pitch - 32.0) < 1e-9
    est = PostureEstimator(Thresholds(), result.calibration)
    out = None
    for t in range(0, 3001, 200):
        out = est.update(observe_pitch(t, usual, 44.0))           # 절대 기준(42°)은 넘지만 평소보다 12°만 더 숙임
    assert out.state == NORMAL
    for t in range(3200, 6001, 200):
        out = est.update(observe_pitch(t, usual, 52.0))           # 평소보다 20° 더 숙임 → 거북목
    assert out.state == FHP


def test_calibration_lets_slouched_baseline_read_as_normal():
    usual = body(head_ratio=0.30, torso_deg=10.0)   # 기울기 약 21.9°: 절대 기준(15°)이면 늘 거북목, 캘리브레이션 한도(15°+9.5°) 안
    result = calibrate(lambda t: observe(t, usual))
    assert result.status == DONE
    assert abs(result.calibration.axis_angle - (10 + math.degrees(math.atan2(0.30, 1.42)))) < 1e-9

    absolute, calibrated = PostureEstimator(), PostureEstimator(Thresholds(), result.calibration)
    for t in range(0, 3001, 200):
        a, c = absolute.update(observe(t, usual)), calibrated.update(observe(t, usual))
    assert a.state == FHP and c.state == NORMAL and c.calibrated


def test_calibration_failures():
    assert calibrate(lambda t: observe(t, body(torso_deg=0 if t // 1000 % 2 == 0 else 25))).status == MOVING
    assert calibrate(lambda t: Observation(t, False)).status == NOT_VISIBLE
    assert calibrate(lambda t: observe(t, body(torso_deg=35))).status == NOT_UPRIGHT   # 35° > 15° + 9.5°
    assert calibrate(lambda t: observe_pitch(t, body(), 70.0)).status == NOT_UPRIGHT   # 고개 70° > 42° + 15°
    assert calibrate(lambda t: observe_pitch(t, body(), 20.0 if t // 1000 % 2 == 0 else 45.0)).status == MOVING


def test_result_is_fixed_after_finishing():
    est, collector = PostureEstimator(), CalibrationCollector()
    b = body()
    results = [collector.add(t, est.update(observe(t, b))) for t in range(0, 8001, 200)]
    assert results[-1] == results[25] and results[-1].status == DONE


def test_calibrated_simulation_helps_subjects_with_unusual_baselines():
    subject = synthetic.Subject("slouch", head_neutral=0.12, torso_neutral=11.0, head_shift=0.30, lean_shift=18.0)
    labels = synthetic.script_intervals(synthetic.COLLECTION_SCRIPT)
    rec = Recording("slouch", "S", synthetic.generate(subject, labels, seed=1), labels)
    t = Thresholds(lean_degrees=12.0, lean_delta_degrees=9.0)
    assert calibrate_recording(rec, t).status == DONE
    absolute = simulate_app([(rec, t)])
    calibrated = simulate_app([(rec, t)], calibrate=True)
    assert calibrated.calibration_failures == 0
    assert calibrated.frame_accuracy > absolute.frame_accuracy + 0.15
