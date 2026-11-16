# 프론트·백엔드 연동 안내 (AI 파트 → 팀원)

AI 파트 결과물을 앱과 서버에 붙이는 방법이에요. **ai 저장소는 프론트·백엔드 저장소를 직접 수정하지 않아요.** 아래 단계는 각 담당자가 자기 저장소에서 진행해 주세요.

---

## 프론트 (Android, `frontend/core`)

### 1. 판정기 파일 교체 (필수)

| ai 저장소 | → frontend |
|---|---|
| `kotlin/src/main/kotlin/com/sidefocus/core/PostureEstimator.kt` | `core/src/main/kotlin/com/sidefocus/core/PostureEstimator.kt` (덮어쓰기) |
| `kotlin/src/main/kotlin/com/sidefocus/core/PostureFeatures.kt` | 같은 폴더에 새로 추가 |
| `kotlin/src/test/kotlin/com/sidefocus/core/PostureEstimatorTest.kt` | `core/src/test/.../PostureEstimatorTest.kt` (덮어쓰기) |

- `PostureStub.kt`는 복사하지 마세요. 프론트에는 `Session.kt`에 `Posture`가 이미 있어요.
- `GoldenParityTest.kt`는 선택이에요. 복사해도 golden 폴더가 없으면 자동으로 건너뛰어요.
- **공개 API는 그대로예요.** `PostureEstimator(thresholds).update(PoseObservation)`, `PoseFeedback.copy(...)`, `framed`, `SetupGate`를 그대로 쓰면 돼요. 앱 코드는 고치지 않아도 컴파일돼요.
- 교체 전에 ai 저장소에서 `kotlin/gradlew verifyAgainstFrontend`를 돌리면, 프론트 core 소스·테스트와 함께 컴파일하고 테스트까지 해요(현재 통과).

### 2. 바뀌는 동작

| 항목 | 기존 | 새 판정기 |
|---|---|---|
| 머리 특징 | \|귀.x − 어깨.x\| ÷ 길이 | 몸통 축에 수직인 앞쪽 성분 ÷ 길이(부호 있음) |
| 몸통 각도 | 절댓값(뒤로 젖혀도 +) | 부호 있음(앞 +, 뒤 −) |
| 앞쪽 방향 | 구분 없음 | 코 → 보이는 쪽 순서로 판단, 세션 다수결 |
| 전환 | 2초 연속 같은 판정일 때 | 최근 2초 창에서 70% 이상일 때(200ms마다 갱신) |
| 동시 발생 | FHP만 | `state`는 FHP, `headForward`·`torsoLean`으로 둘 다 확인 가능 |
| 재세팅 기준 길이 | 첫 프레임 | 처음 1초 중앙값 |

새로 생긴 `PoseFeedback` 필드(모두 기본값 있음): `headForward`, `torsoLean`, `neckAngle`, `facing`. 피드백 문구에 활용할 수 있어요.

### 3. 코 좌표 전달 (권장)

코가 없으면 "보이는 쪽"으로 방향을 추정하는데, 측면 영상에서는 MediaPipe가 좌우를 헷갈릴 수 있어요. `CameraSurface.kt` 분석 부분에서 0번(코)을 넘겨 주세요.

```kotlin
val landmarks = result.landmarks().firstOrNull()
fun point(i: Int) = landmarks!![i].let { p -> Point(p.x().toDouble() * bitmap.width, p.y().toDouble() * bitmap.height,
    if (p.x() in 0f..1f && p.y() in 0f..1f) minOf(p.visibility().orElse(0f), p.presence().orElse(0f)) else 0f) }
fun side(ear: Int, shoulder: Int, hip: Int): BodySide? =
    if (landmarks == null || landmarks.size < 25) null else BodySide(point(ear), point(shoulder), point(hip))
val nose = if (landmarks != null && landmarks.isNotEmpty()) point(0) else null
val frame = CameraFrame(PoseObservation(now, landmarks != null, side(7, 11, 23), side(8, 12, 24), nose = nose), bitmap.width, bitmap.height)
```

### 4. 거치 기울기 전달 (선택)

폰이 옆으로 기울어 있으면 몸통 각도가 그만큼 틀어져요. `deviceRollDeg` = 분석 이미지 좌표(x 오른쪽, y 아래)에서 중력 방향 (sin r, cos r)의 r이에요.
- 세로 거치에서 가속도 (ax, ay) 기준 예: `r = Math.toDegrees(atan2(-ax, ay))`
- **부호는 기기마다 확인해 주세요.** 폰을 10° 기울인 채 똑바로 앉았을 때 `leanDegrees`가 0 근처로 유지되면 맞아요.
- 값을 넘기지 않으면 보정하지 않아요(기존과 같음).

### 5. 임계값

새 판정기는 머리 특징의 정의가 달라졌어요. 그래서 실험 데이터로 정한 **v2 임계값**을 서버에 등록한 뒤 교체하는 게 가장 좋아요. 지금 기본값(0.35 / 20°)과 서버 v1(0.25 / 20°)은 둘 다 실험 전 임시값이에요.

---

## 백엔드 (`threshold_configs`)

- 스키마 변경은 **필요 없어요.** `t1_ratio`는 새 정의의 `headRatio` 임계값, `t2_angle`은 몸통 각도 임계값이에요.
- 실험 데이터로 `python -m posture_ai analyze`를 돌리면 `outputs/real/thresholds.sql`이 나와요. 새 버전 행을 넣고 활성 버전을 바꾸는 SQL이에요(기존 버전은 수정하지 않음).
  ```sql
  BEGIN;
  INSERT INTO threshold_configs (version, t1_ratio, t2_angle, stable_sec, analysis_fps, is_active)
  VALUES (2, <t1>, <t2>, 2.0, 5, false);
  UPDATE threshold_configs SET is_active = false WHERE is_active;
  UPDATE threshold_configs SET is_active = true WHERE version = 2;
  COMMIT;
  ```
- `POST /admin/configs`가 생기면 같은 JSON(`thresholds.json`)을 그대로 보내면 돼요.
- 프론트가 새 판정기로 교체하는 시점과 v2 활성화 시점을 맞춰 주세요. 세션마다 `threshold_version`이 저장되니, 버전별로 전후 비교도 할 수 있어요.

---

## 확인이 필요한 미정 사항

1. **동시 발생 저장 방식:** 서버 enum에는 상태가 하나만 있어서 BOTH가 FHP로 저장돼요. 리포트에서 "둘 다"를 따로 보여 줄지 팀이 정해 주세요(필요하면 enum에 `BOTH`를 추가하거나 `posture_segments`에 플래그를 추가).
2. **ML 모델 탑재:** 발표 3-4 기준(Macro F1 +0.05, 피험자 과반 개선, 처리 p95 ≤ 200ms)을 만족할 때만 ONNX 모델을 앱에 넣어요. 그 전까지는 규칙 판정기만 써요.
