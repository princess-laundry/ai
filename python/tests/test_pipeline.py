import json
import math

import numpy as np
import pandas as pd
import pytest

from posture_ai import export, synthetic, tune
from posture_ai.estimator import Thresholds
from posture_ai.evaluate import labeler_agreement, simulate_app
from posture_ai.labels import Interval, label_at, read_labels, write_labels
from posture_ai.landmarks import COLUMNS, row_to_observation
from posture_ai.windows import Recording, build_samples, make_windows


@pytest.fixture(scope="module")
def small_dataset():
    return [Recording(n, s.name, f, l) for n, s, f, l in synthetic.team_dataset(seed=3, repeats=1)]


def test_labels_roundtrip_and_transition_margin(tmp_path):
    intervals = [Interval(0, 10, False, False), Interval(10, 20, True, False), Interval(20, 30, True, True, exclude=True)]
    write_labels(intervals, tmp_path / "a.csv")
    assert read_labels(tmp_path / "a.csv") == intervals
    assert label_at(intervals, 5).combo == "NORMAL"
    assert label_at(intervals, 9.5) is None and label_at(intervals, 10.8) is None   # 전환 ±1초 제외
    assert label_at(intervals, 15).combo == "FHP"
    assert label_at(intervals, 25) is None                                           # 제외 구간


def test_overlapping_labels_are_rejected(tmp_path):
    pd.DataFrame([[0, 10, 0, 0, 0, 0], [5, 12, 1, 0, 0, 0]], columns=["start_s", "end_s", "head_forward", "torso_lean", "away", "exclude"]
                 ).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        read_labels(tmp_path / "bad.csv")


def test_same_posture_split_is_not_a_transition():
    intervals = [Interval(0, 10, False, False), Interval(10, 20, False, False)]
    assert label_at(intervals, 10.2) is not None


def test_row_conversion_matches_app_rules():
    row = {c: math.nan for c in COLUMNS}
    row.update(t_ms=200, width=480, height=640, person=1, roll_deg=math.nan)
    for i in (0, 7, 8, 11, 12, 23, 24):
        row.update({f"x{i}": 0.5, f"y{i}": 0.5, f"z{i}": 0, f"vis{i}": 0.9, f"pres{i}": 0.7})
    row["x8"] = 1.2  # 화면 밖 점은 신뢰도 0
    obs = row_to_observation(pd.Series(row))
    assert obs.left.ear.x == 240 and obs.left.ear.y == 320 and obs.left.ear.confidence == pytest.approx(0.7)
    assert obs.right.ear.confidence == 0.0
    assert obs.device_roll_deg is None


def test_windows_have_balanced_labels_and_calibration(small_dataset):
    samples = make_windows(small_dataset[0])
    assert set(samples["combo"]) == {"NORMAL", "FHP", "LEAN", "BOTH"}
    assert samples[["d_axis", "d_head", "d_torso"]].notna().all().all()
    normal = samples[samples["combo"] == "NORMAL"]
    assert normal["d_head"].abs().max() < 0.1   # 기준 자세 대비 차이는 0 근처


def test_threshold_tuning_picks_middle_of_tied_range():
    samples = pd.DataFrame({"axis_mean": [3.0, 6.0, 25.0, 30.0], "fhp": [0, 0, 1, 1]})
    threshold = tune.fit(samples, "abs")
    assert list(threshold.predict(samples)) == [0, 0, 1, 1]
    assert 12 <= threshold.value <= 19          # 6°와 25° 사이 가운데쯤


def test_single_posture_videos_share_the_subjects_baseline():
    from posture_ai.labels import Interval
    from posture_ai.windows import trim_to_labels
    subject = synthetic.random_subject("S9", np.random.default_rng(1))
    normal_labels = [Interval(3, 40, False, False)]
    fhp_labels = [Interval(3, 40, True, False)]
    normal = Recording("S9_normal", "S9", synthetic.generate(subject, [Interval(0, 41, False, False)], seed=1), normal_labels)
    fhp = Recording("S9_fhp", "S9", synthetic.generate(subject, [Interval(0, 41, True, False)], seed=2), fhp_labels)
    assert trim_to_labels(normal.landmarks, normal_labels)["t_ms"].min() >= 3000   # 자리 잡는 앞부분은 버림
    samples = build_samples([normal, fhp])
    fhp_rows = samples[samples["recording"] == "S9_fhp"]
    assert fhp_rows["d_head"].notna().all() and (fhp_rows["fhp"] == 1).all()        # 바른 자세 영상의 기준을 씀
    normal_rows = samples[samples["recording"] == "S9_normal"]
    assert (normal_rows["t_start"] >= 9).all()                                       # 기준을 구한 4~9초 창은 평가에서 뺌


def test_app_simulation_counts_events(small_dataset):
    sim = simulate_app([(r, Thresholds(head_ratio=0.2, lean_degrees=12)) for r in small_dataset])
    assert sim.events == 9 and sim.frames > 0 and 0 <= sim.miss_rate <= 1


def test_kappa_is_one_for_identical_labels():
    a = synthetic.script_intervals(synthetic.COLLECTION_SCRIPT)
    assert labeler_agreement(a, a, 80)["head_kappa"] == pytest.approx(1.0)


def test_threshold_export_matches_server_config_shape(tmp_path):
    t = Thresholds(version=3, head_ratio=0.2, lean_degrees=17.5)
    export.write_thresholds(t, tmp_path, "note")
    config = json.loads((tmp_path / "thresholds.json").read_text())
    assert config == {"version": 3, "t1_ratio": 0.2, "t2_angle": 17.5, "stable_sec": 2.0, "analysis_fps": 5,
                      "t2_delta": 9.5}
    assert Thresholds.from_server(config) == t
    sql = (tmp_path / "thresholds.sql").read_text(encoding="utf-8")
    assert sql.startswith("-- note") and "t2_delta=9.5" in sql and "VALUES (3, 0.2, 17.5, 2.0, 5, false)" in sql


def test_onnx_export_matches_sklearn(tmp_path, small_dataset):
    from posture_ai import train
    samples = build_samples(small_dataset)
    model = train.make_model("rf", {"n_estimators": 20, "max_depth": 4}).fit(samples[train.columns("abs")].to_numpy(), samples["combo"])
    info = export.to_onnx(model, train.columns("abs"), tmp_path / "m.onnx", samples[train.columns("abs")].to_numpy(), {})
    assert info["sklearn_onnx_agreement"] == 1.0


def test_full_pipeline_writes_report(tmp_path, small_dataset):
    from posture_ai.pipeline import analyze
    result = analyze(small_dataset, tmp_path, synthetic=True)
    for name in ("report.md", "comparison.csv", "thresholds.json", "thresholds.sql", "app_simulation.json", "samples.csv"):
        assert (tmp_path / name).exists(), name
    assert len(result["comparison"]) == 6
    assert "합성 데이터" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_mediapipe_extraction_runs_on_a_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from posture_ai.landmarks import extract_video
    from posture_ai.models import ensure_lite_model

    try:
        model = ensure_lite_model(tmp_path / "pose_landmarker_lite.task")
    except Exception as exc:  # 네트워크가 없고 frontend asset도 없으면 건너뜀
        pytest.skip(f"모델을 준비할 수 없음: {exc}")
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(60):
        writer.write(np.full((240, 320, 3), 200, np.uint8))
    writer.release()
    frame = extract_video(video, model, fps=5)
    assert list(frame.columns) == COLUMNS
    assert 9 <= len(frame) <= 11            # 2초 × 5fps
    assert frame["person"].sum() == 0       # 빈 화면엔 사람이 없음
