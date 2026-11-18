"""정답 라벨 만들기 도구 두 가지.

1) 대본 기반: 촬영을 정해진 대본(네 자세 × 20초 × 3회)대로 했다면, 영상 속 시작 시각만 알면 라벨이 나옵니다.
2) 키 입력 기록: 진행자가 촬영하면서 자세가 바뀔 때 숫자 키를 누르면 시각을 기록합니다.
   1=바른 자세  2=전방머리  3=상체 기울임  4=둘 다  0=자리 비움  x=모호(제외) 켜기/끄기  q=끝
   영상 녹화 시작과 동시에 Enter를 누르세요. 어긋나면 --offset 으로 보정합니다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .labels import Interval, write_labels
from .synthetic import COLLECTION_SCRIPT, SCENARIO_150S, script_intervals

KEYS = {"1": "NORMAL", "2": "FHP", "3": "LEAN", "4": "BOTH", "0": "AWAY"}
SCRIPTS = {"collection": COLLECTION_SCRIPT, "scenario150": SCENARIO_150S}


def _interval(combo: str, start: float, end: float, exclude: bool = False) -> Interval:
    return Interval(start, end, combo in ("FHP", "BOTH"), combo in ("LEAN", "BOTH"), combo == "AWAY", exclude)


def from_script(script: str, start_s: float, repeats: int, out: Path) -> list[Interval]:
    """대본 시작 시각(영상 기준 초)부터 대본 구간을 라벨로 씁니다. 대본 전 구간은 제외(exclude)로 둡니다."""
    intervals = [Interval(i.start_s + start_s, i.end_s + start_s, i.head_forward, i.torso_lean, i.away)
                 for i in script_intervals(SCRIPTS[script], repeats)]
    if start_s > 0:
        intervals.insert(0, Interval(0.0, start_s, False, False, exclude=True))
    write_labels(intervals, out)
    return intervals


def _read_key() -> str:
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.getwch()
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def record_keys(out: Path, offset_s: float = 0.0) -> list[Interval]:
    print(__doc__)
    input("영상 녹화를 시작하는 순간 Enter ▶ ")
    t0 = time.monotonic()
    combo, since, exclude = "NORMAL", 0.0, False
    intervals: list[Interval] = []
    print("기록 중: NORMAL")
    while True:
        key = _read_key().lower()
        now = time.monotonic() - t0
        if key not in KEYS and key not in ("x", "q"):
            continue
        if now > since:
            intervals.append(_interval(combo, since + offset_s, now + offset_s, exclude))
        since = now
        if key == "q":
            break
        if key == "x":
            exclude = not exclude
        else:
            combo = KEYS[key]
        print(f"{now:7.1f}s  {combo}{'  (제외)' if exclude else ''}")
    intervals = [i for i in intervals if i.end_s > max(0.0, i.start_s)]
    write_labels(intervals, out)
    print(f"저장: {out} ({len(intervals)}개 구간)")
    return intervals
