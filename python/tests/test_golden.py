import json
from pathlib import Path

from posture_ai.golden import build_cases

GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "golden"


def test_golden_files_are_up_to_date():
    """판정 로직을 고쳤다면 `python -m posture_ai golden`으로 다시 만들고 Kotlin 테스트도 돌리세요."""
    for case in build_cases():
        stored = json.loads((GOLDEN / f"{case['name']}.json").read_text(encoding="utf-8"))
        assert stored == json.loads(json.dumps(case)), f"{case['name']}.json이 현재 Python 판정기와 다릅니다"
