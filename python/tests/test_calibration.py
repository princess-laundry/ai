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


def test_calibration_lets_slouched_baseline_read_as_normal():
    usual = body(head_ratio=0.40, torso_deg=22.0)
    result = calibrate(lambda t: observe(t, usual))
    assert result.status == DONE
    assert abs(result.calibration.head_ratio - 0.40) < 1e-9

    absolute, calibrated = PostureEstimator(), PostureEstimator(Thresholds(), result.calibration)
    for t in range(0, 3001, 200):
        a, c = absolute.update(observe(t, usual)), calibrated.update(observe(t, usual))
    assert a.state == FHP and c.state == NORMAL and c.calibrated


def test_calibration_failures():
    assert calibrate(lambda t: observe(t, body(torso_deg=0 if t // 1000 % 2 == 0 else 25))).status == MOVING
    assert calibrate(lambda t: Observation(t, False)).status == NOT_VISIBLE
    assert calibrate(lambda t: observe(t, body(head_ratio=0.7))).status == NOT_UPRIGHT


def test_result_is_fixed_after_finishing():
    est, collector = PostureEstimator(), CalibrationCollector()
    b = body()
    results = [collector.add(t, est.update(observe(t, b))) for t in range(0, 8001, 200)]
    assert results[-1] == results[25] and results[-1].status == DONE


def test_calibrated_simulation_helps_subjects_with_unusual_baselines():
    subject = synthetic.Subject("slouch", head_neutral=0.25, torso_neutral=16.0, head_shift=0.25, lean_shift=18.0)
    labels = synthetic.script_intervals(synthetic.COLLECTION_SCRIPT)
    rec = Recording("slouch", "S", synthetic.generate(subject, labels, seed=1), labels)
    t = Thresholds(head_ratio=0.2, lean_degrees=12.0, head_delta=0.12, lean_delta_degrees=9.0)
    assert calibrate_recording(rec, t).status == DONE
    absolute = simulate_app([(rec, t)])
    calibrated = simulate_app([(rec, t)], calibrate=True)
    assert calibrated.calibration_failures == 0
    assert calibrated.frame_accuracy > absolute.frame_accuracy + 0.2
