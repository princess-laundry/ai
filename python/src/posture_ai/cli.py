"""명령줄 진입점: python -m posture_ai <명령> (자세한 사용법은 README.md)

데이터 폴더 구조 (기본 ../data, git에 올리지 않음):
  data/videos/<recording>.mp4     실험 영상(동의 후, 프로젝트 종료 시 삭제)
  data/landmarks/<recording>.csv  extract 결과
  data/labels/<recording>.csv     정답 라벨
  data/subjects.csv               recording,subject,side,location,labeler[,rotate]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


def load_dataset(data: Path):
    from .labels import read_labels, read_subjects
    from .landmarks import read_csv
    from .windows import Recording

    subjects = read_subjects(data / "subjects.csv")
    recordings, skipped = [], []
    for row in subjects.itertuples():
        lm, lb = data / "landmarks" / f"{row.recording}.csv", data / "labels" / f"{row.recording}.csv"
        if lm.exists() and lb.exists():
            recordings.append(Recording(str(row.recording), str(row.subject), read_csv(lm), read_labels(lb)))
        else:
            skipped.append(str(row.recording))
    if skipped:
        print(f"랜드마크나 라벨이 없어 건너뜀: {', '.join(skipped)}")
    return recordings


def cmd_model(args) -> None:
    from .models import ensure_lite_model
    print(f"모델 준비됨: {ensure_lite_model()}")


def cmd_extract(args) -> None:
    import pandas as pd
    from .landmarks import extract_video, write_csv
    from .models import ensure_lite_model

    model = ensure_lite_model()
    rotate = {}
    subjects_csv = args.data / "subjects.csv"
    if subjects_csv.exists():
        frame = pd.read_csv(subjects_csv)
        if "rotate" in frame.columns:
            rotate = dict(zip(frame["recording"].astype(str), frame["rotate"].fillna(0).astype(int)))
    videos = sorted(p for p in (args.data / "videos").glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"))
    if args.only:
        videos = [v for v in videos if v.stem in args.only]
    if not videos:
        print(f"영상이 없습니다: {args.data / 'videos'}")
    for video in videos:
        out = args.data / "landmarks" / f"{video.stem}.csv"
        if out.exists() and not args.force:
            print(f"이미 있음(건너뜀, 다시 하려면 --force): {out.name}")
            continue
        frame = extract_video(video, model, fps=args.fps, rotate_degrees=args.rotate if args.rotate is not None else rotate.get(video.stem, 0))
        write_csv(frame, out)
        detected = frame["person"].mean() if len(frame) else 0
        print(f"{video.name}: {len(frame)}프레임, 사람 검출 {detected:.0%} → {out}")


def cmd_label_script(args) -> None:
    from .label_tool import from_script
    out = args.data / "labels" / f"{args.recording}.csv"
    intervals = from_script(args.script, args.start, args.repeats, out)
    print(f"{out}: {len(intervals)}개 구간")


def cmd_label_keys(args) -> None:
    from .label_tool import record_keys
    record_keys(args.data / "labels" / f"{args.recording}.csv", args.offset)


def cmd_kappa(args) -> None:
    from .evaluate import labeler_agreement
    from .labels import read_labels
    a, b = read_labels(args.a), read_labels(args.b)
    end = max(max(i.end_s for i in a), max(i.end_s for i in b))
    print(json.dumps(labeler_agreement(a, b, end), indent=2))


def cmd_analyze(args) -> None:
    from .pipeline import analyze
    recordings = load_dataset(args.data)
    if not recordings:
        raise SystemExit("분석할 녹화가 없습니다. extract와 라벨링을 먼저 하세요.")
    result = analyze(recordings, args.out, synthetic=False, version=args.version)
    _print_result(result, args.out)


def cmd_demo(args) -> None:
    from . import synthetic
    from .pipeline import analyze
    from .windows import Recording

    recordings = [Recording(n, s.name, f, l) for n, s, f, l in synthetic.team_dataset(seed=args.seed)]
    if args.save_data:
        from .labels import write_labels
        from .landmarks import write_csv
        for r in recordings:
            write_csv(r.landmarks, args.out / "data" / "landmarks" / f"{r.name}.csv")
            write_labels(r.labels, args.out / "data" / "labels" / f"{r.name}.csv")
    result = analyze(recordings, args.out, synthetic=True, version=args.version, seed=args.seed)
    _print_result(result, args.out)


def _print_result(result, out: Path) -> None:
    print(result["comparison"].round(3).to_string(index=False))
    print("임계값:", json.dumps(result["thresholds"]))
    sim = result["simulation"]
    print(f"앱 시뮬레이션: 프레임 정확도 {sim.frame_accuracy:.1%}, 시간당 오탐 {sim.false_alerts_per_hour:.1f}, "
          f"미탐률 {sim.miss_rate:.1%}, 확정 지연 중앙값 {sim.latency_median_s:.1f}s")
    print(f"ML 적용: {'후보' if result['adopt_ml'] else '규칙 MVP 유지'} · 리포트: {out / 'report.md'}")


def cmd_golden(args) -> None:
    from .golden import write
    for path in write(args.out):
        print(path)


def cmd_visualize(args) -> None:
    from .estimator import Thresholds
    from .labels import read_labels
    from .landmarks import read_csv
    from .visualize import render

    video = next((p for p in (args.data / "videos").glob(f"{args.recording}.*")), None)
    if video is None:
        raise SystemExit(f"영상이 없습니다: {args.recording}")
    labels_path = args.data / "labels" / f"{args.recording}.csv"
    thresholds = Thresholds(head_ratio=args.t1, lean_degrees=args.t2)
    out = render(video, read_csv(args.data / "landmarks" / f"{args.recording}.csv"), args.out / f"{args.recording}_overlay.mp4",
                 thresholds, read_labels(labels_path) if labels_path.exists() else None, args.rotate)
    print(out)


def main(argv: list[str] | None = None) -> None:
    _utf8_console()
    parser = argparse.ArgumentParser(prog="posture_ai", description="측면 자세 인식 AI 파트 PC 도구")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=func)
        p.add_argument("--data", type=Path, default=DATA, help="데이터 폴더 (기본: 저장소의 data/)")
        return p

    add("model", cmd_model, "MediaPipe lite 모델 준비(앱과 같은 파일인지 확인)")
    p = add("extract", cmd_extract, "영상 → 5fps 랜드마크 CSV")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None, help="영상 회전(기본: subjects.csv의 rotate)")
    p.add_argument("--only", nargs="*", help="이 녹화 이름만")
    p.add_argument("--force", action="store_true")
    p = add("label-script", cmd_label_script, "대본대로 촬영한 영상의 라벨 만들기")
    p.add_argument("recording")
    p.add_argument("--start", type=float, required=True, help="영상에서 대본 첫 자세가 시작된 시각(초)")
    p.add_argument("--script", choices=["collection", "scenario150"], default="collection")
    p.add_argument("--repeats", type=int, default=3)
    p = add("label-keys", cmd_label_keys, "촬영하면서 키 입력으로 라벨 기록")
    p.add_argument("recording")
    p.add_argument("--offset", type=float, default=0.0, help="영상 시각 보정(초)")
    p = add("kappa", cmd_kappa, "두 라벨러의 Cohen's κ")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p = add("analyze", cmd_analyze, "실제 데이터: 샘플 → 임계값/SVM/RF 비교 → 앱 시뮬레이션 → 내보내기")
    p.add_argument("--out", type=Path, default=OUTPUTS / "real")
    p.add_argument("--version", type=int, default=2, help="서버에 등록할 threshold_configs 버전")
    p = add("demo", cmd_demo, "합성 데이터로 전체 파이프라인 시연(실제 성능 아님)")
    p.add_argument("--out", type=Path, default=OUTPUTS / "demo")
    p.add_argument("--version", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-data", action="store_true", help="합성 랜드마크·라벨 CSV도 저장")
    p = add("golden", cmd_golden, "Kotlin 일치 테스트용 golden 파일 다시 만들기")
    p.add_argument("--out", type=Path, default=REPO / "fixtures" / "golden")
    p = add("visualize", cmd_visualize, "영상에 스켈레톤·상태·라벨을 겹쳐 그리기")
    p.add_argument("recording")
    p.add_argument("--t1", type=float, default=0.35)
    p.add_argument("--t2", type=float, default=20.0)
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    p.add_argument("--out", type=Path, default=OUTPUTS / "overlay")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
