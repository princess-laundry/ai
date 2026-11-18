# 전처리·특징값·판정 규칙 명세

앱(Kotlin)과 PC(Python)가 **똑같이** 따르는 기준 문서입니다. PC에서 정한 임계값이 앱에서도 그대로 통하려면 이 규칙이 양쪽에서 같아야 합니다.

| 구현 | 파일 |
|---|---|
| Python | `python/src/posture_ai/features.py`, `estimator.py` |
| Kotlin | `kotlin/src/main/kotlin/com/sidefocus/core/PostureFeatures.kt`, `PostureEstimator.kt` |
| 일치 검증 | `fixtures/golden/*.json` ↔ `GoldenParityTest.kt` (실수 오차 1e-6 이내) |

규칙을 바꿀 때 순서: Python 수정 → `python -m posture_ai golden` → Kotlin 수정 → `gradlew test` 통과.

## 입력 (프레임 하나 = 분석 이미지 1장, 기본 5fps)

| 필드 | 내용 |
|---|---|
| `timestampMs` | 단조 증가 시각(ms) |
| `personDetected` | MediaPipe가 사람을 찾았는지 |
| `left` / `right` | 귀·어깨·골반 (MediaPipe 7·11·23 / 8·12·24). **픽셀 좌표**(정규화 좌표 × 이미지 폭·높이). 신뢰도 = min(visibility, presence), 화면 밖이면 0 |
| `nose` (선택) | 코(0번). 앞쪽 방향 결정용 |
| `deviceRollDeg` (선택) | 거치 기울기. 이미지 좌표(y 아래)에서 중력 방향이 (sin r, cos r)일 때의 r |

## 처리 순서

1. **품질 필터:** 신뢰도 0.5 미만인 점이 하나라도 있는 쪽은 버립니다. 좌표가 NaN이거나 무한대여도 버립니다.
2. **측면 선택:** 남은 쪽 중 세 점의 신뢰도 평균이 높은 쪽을 씁니다. 같으면 왼쪽을 씁니다.
3. **결측 처리:** 쓸 수 있는 쪽이 없으면 직전 상태를 유지합니다. 결측이 `stableMs`(2초) 이상 이어지면 이렇게 바꿉니다.
   - 사람이 안 보이면 → `AWAY`
   - 사람은 있는데 귀·어깨·골반이 가려졌으면 → `UNKNOWN`
   - 이때 스무딩·창·속성도 초기화합니다.
4. **기울기 보정:** `deviceRollDeg`가 있으면 모든 점을 원점 기준으로 r만큼 회전합니다.
   x' = x·cos r − y·sin r, y' = x·sin r + y·cos r
5. **앞쪽 방향(facing)은 세션 내내 누적 투표로 정합니다.**
   - 코 투표: |코.x − 귀.x| ≥ 0.05 × 어깨–골반 길이이면, 코가 귀보다 +x 쪽일 때 +1, 아니면 −1
   - 보이는 쪽 투표: 오른쪽 몸이 선택되면 +0.25, 왼쪽이면 −0.25. 사람의 왼쪽 몸이 카메라를 향하면 앞쪽이 이미지 −x입니다. 앱의 분석 프레임은 좌우 반전되지 않은 원본이라 이 규칙이 성립합니다.
   - 결정 순서: 코 투표 합이 0이 아니면 그 부호를 씁니다. 0이면 보이는 쪽 투표 합의 부호, 그것도 0이면 +1입니다.
6. **특징값.** 앞쪽 = +f, 위쪽 = +u 좌표로 바꿔 계산합니다(s = facing).
   - 몸통 벡터 T = (s·(어깨.x − 골반.x), 골반.y − 어깨.y), L = |T|. L < 0.001이면 판정하지 않습니다.
   - `torsoAngle` = atan2(T.f, T.u) [도]. 앞으로 숙이면 +, 뒤로 젖히면 −.
   - 머리 벡터 H = (s·(귀.x − 어깨.x), 어깨.y − 귀.y)
   - `headRatio` = (H.f·T.u − H.u·T.f) / L². H에서 **몸통 축에 수직인 앞쪽 성분**을 L로 나눈 값입니다.
     → 상체를 통째로 숙여도 머리가 몸통과 같은 선에 있으면 0 근처라서 전방머리로 판정되지 않습니다.
   - `neckAngle` = atan2(H.f, H.u) − torsoAngle [도, (−180, 180]]. 보조 특징이고 ML에 씁니다.
7. **스무딩:** 각 특징값과 L은 최근 5프레임의 중앙값을 씁니다. 짝수 개면 정렬한 뒤 `size/2`번째, 즉 위쪽 중앙값입니다.
8. **재세팅 감지:**
   - 처음 1초 동안의 원시 L의 중앙값을 기준 길이로 정합니다.
   - 그 뒤 스무딩된 L이 기준에서 ±30%를 넘게 변하면 `needsReset`을 켜고 `UNKNOWN`으로 둡니다. 스무딩·창·속성도 초기화합니다.
9. **프레임 판정:** `head = headRatio > T₁`, `lean = torsoAngle > T₂` (서버 `t1_ratio`, `t2_angle`)
10. **시간 안정화 (2초 창):**
    - 최근 `stableMs` 안의 프레임으로 창을 만듭니다(`t > now − stableMs`).
    - 창이 처음 `stableMs`만큼 차기 전에는 직전 상태를 유지합니다(시작할 때는 `UNKNOWN`).
    - 속성마다 처음 정할 때는 창의 과반(≥50%)을 따릅니다.
    - 그 뒤로는 창의 **70% 이상**이 반대 판정일 때만 바꿉니다(히스테리시스). 정수 비교 `count·100 ≥ 70·total`로 계산합니다.
    - 따로 "2초 동안 계속 유지되기를 기다리는" 단계는 없습니다.
11. **단일 상태 (서버 enum):** head나 lean 중 하나라도 참이면 `FHP`(거북목), 둘 다 아니면 `NORMAL`. **`LEAN`은 내보내지 않습니다**(2026-11-18 팀 결정: 상체 기울임도 거북목으로 통합). 두 속성은 `PoseFeedback.headForward` / `torsoLean`으로 따로 남아서, 안내 문구("턱을 당기세요" / "허리를 펴세요")와 분석에 씁니다.

## 개인 캘리브레이션

| 구현 | 파일 |
|---|---|
| Python | `python/src/posture_ai/calibration.py` |
| Kotlin | `kotlin/src/main/kotlin/com/sidefocus/core/PostureCalibration.kt` |

1. **수집:** 캘리브레이션 없이 만든 판정기의 출력(스무딩된 `headRatio`, `torsoAngle`)을 프레임마다 수집기에 넣어요. 첫 프레임부터 **5초**가 지나면 아래 순서로 확인해요.
   - 유효 프레임(특징값 있음, 재세팅 아님)이 5개 미만이거나 80% 미만 → `NOT_VISIBLE`
   - 머리 비율 표준편차(모집단) > 0.05 또는 몸통 > 4° → `MOVING`
   - 기준 = 두 값 각각의 위쪽 중앙값
   - 기준 머리 비율 > T₁ + T₁Δ 또는 기준 몸통 > T₂ + T₂Δ → `NOT_UPRIGHT`. 절대 임계값만으로 막으면 평소 자세가 기준선 근처인 사람은 캘리브레이션을 못 해서 Δ만큼 여유를 둬요.
   - 그 외 → `DONE`. 이후 결과는 고정돼요.
2. **판정:** 판정기를 기준과 함께 만들면 9번 프레임 판정이 이렇게 바뀌어요. 나머지 단계는 같아요.
   `head = headRatio − 기준.headRatio > T₁Δ`, `lean = torsoAngle − 기준.torsoAngle > T₂Δ`
3. **저장:** 기준은 백엔드 `user_settings.calib_ratio` / `calib_angle`에 대응해요. Δ 임계값은 `Thresholds.headDelta` / `leanDeltaDegrees`(서버 `t1_delta` / `t2_delta`, 아직 서버에 컬럼 없음)이고, 기본값 0.15 / 12°는 실험 전 임시값이에요.

## 상수

| 이름 | 값 | 비고 |
|---|---|---|
| MIN_CONFIDENCE | 0.5 | 발표 4-3 ② |
| MEDIAN_FRAMES | 5 | 약 1초 |
| AGREEMENT_PERCENT | 70 | 실험으로 조정 가능 |
| RESET_TOLERANCE | 0.30 | 발표 4-3 |
| BASELINE_MS | 1000 | |
| NOSE_MIN_OFFSET | 0.05 | |
| SIDE_VOTE_WEIGHT | 0.25 | |

임계값(T₁, T₂, stableMs, fps)은 서버 `GET /api/v1/config`에서 받습니다. 위 상수는 코드에 고정돼 있어요.

## PC 학습용 샘플 (`windows.py`)

- 녹화 영상을 같은 5fps로 추출하고, 위 판정기를 그대로 돌려 스무딩된 특징값을 얻습니다.
- 비중첩 2초 창(10프레임)마다 head·torso·neck의 평균과 표준편차를 내서 샘플 1개를 만듭니다.
- 이런 창은 버립니다.
  - 라벨이 바뀌는 지점 ±1초에 걸친 창
  - 제외 구간, 자리 비움
  - 라벨이 섞인 창
  - 특징값이 20% 넘게 빠진 창
- Δ특징: 첫 바른 자세 구간에서 전환 1초 뒤부터 5초 동안의 중앙값(= 앱의 5초 캘리브레이션)을 뺀 값이에요.
- 분할: 항상 **피험자 단위**로 나눕니다(LOSO). 같은 사람의 영상이 학습과 평가에 동시에 들어가지 않아요.
