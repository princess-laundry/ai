# 프론트·백엔드 연동 안내 (AI 파트 → 팀원)

AI 파트 결과물을 앱과 서버에 붙이는 방법이에요. **ai 저장소는 프론트·백엔드 저장소를 직접 수정하지 않아요.** 아래 단계는 각 담당자가 자기 저장소에서 진행해 주세요.

---

## 프론트 (Android, `frontend/core`)

### 1. 판정기 파일 교체 (필수)

| ai 저장소 | → frontend |
|---|---|
| `kotlin/src/main/kotlin/com/sidefocus/core/PostureEstimator.kt` | `core/src/main/kotlin/com/sidefocus/core/PostureEstimator.kt` (덮어쓰기) |
| `kotlin/src/main/kotlin/com/sidefocus/core/PostureFeatures.kt` | 같은 폴더에 새로 추가 |
| `kotlin/src/main/kotlin/com/sidefocus/core/PostureCalibration.kt` | 같은 폴더에 새로 추가 (개인 캘리브레이션) |
| `kotlin/src/test/kotlin/com/sidefocus/core/PostureEstimatorTest.kt` | `core/src/test/.../PostureEstimatorTest.kt` (덮어쓰기) |

- `PostureStub.kt`는 복사하지 마세요. 프론트에는 `Session.kt`에 `Posture`가 이미 있어요.
- `GoldenParityTest.kt`는 선택이에요. 복사해도 golden 폴더가 없으면 자동으로 건너뛰어요.
- `PostureLatency.kt`(+ 테스트 `PostureLatencyTest.kt`)는 실기기 처리 지연 측정용이에요. 아래 7번을 보세요.
- **공개 API는 그대로예요.** `PostureEstimator(thresholds).update(PoseObservation)`, `PoseFeedback.copy(...)`, `framed`, `SetupGate`를 그대로 쓰면 돼요. 앱 코드는 고치지 않아도 컴파일돼요.
- **이미 PR #1 버전을 넣었다면(2026-11-18 프론트 PR #2):** 위 4개 파일을 최신 버전으로 다시 덮어쓰면 돼요. 판정 방식이 "골반→귀 선 기울기 하나"로 바뀌었고, 앱 코드(`CameraSurface`의 코 좌표 등)는 그대로 둬도 돼요.
- 교체 전에 ai 저장소에서 `kotlin/gradlew verifyAgainstFrontend`를 돌리면, 프론트 core 소스·테스트와 함께 컴파일하고 테스트까지 해요(현재 통과).

### 2. 바뀌는 동작

| 항목 | 기존 | 새 판정기 |
|---|---|---|
| 판정 값 | 머리 비율·몸통 각도 따로 | ① **골반→귀 선 기울기** (`leanDegrees`) 또는 ② **고개 숙임**(귀→코, `headPitch`) 중 하나라도 넘으면 거북목. 머리를 내밀어도, 허리를 숙여도, 고개만 숙여도 잡혀요 |
| 기준값 | `t1_ratio`·`t2_angle` | ① `t2_angle`(서버), ② `pitchDegrees`(서버 칸 없음 → 코드 기본값 42°). `t1_ratio`는 호환용으로만 남음 |
| 앞쪽 방향 | 구분 없음 | 코 → 보이는 쪽 순서로 판단, 세션 다수결 |
| 전환 | 2초 연속 같은 판정일 때 | 최근 2초 창에서 70% 이상일 때(200ms마다 갱신) |
| 상체 기울임 | `LEAN` | **`FHP`(거북목)로 통합**. `LEAN`은 더 이상 나오지 않음 |
| 재세팅 기준 길이 | 첫 프레임 | 처음 1초 중앙값 |

`PoseFeedback` 필드(모두 기본값 있음): `leanDegrees` = 골반→귀 선 기울기(판정 값 ①), `headPitch` = 고개 숙임(판정 값 ②, 코가 안 보이면 null), `headRatio`·`torsoAngle`·`neckAngle` = 참고용, `facing`, `calibrated`. **코 좌표를 꼭 넘겨 주세요**(이미 연결됨). 코가 없으면 고개만 숙인 자세를 못 잡아요. PR #1에 있던 `headForward`·`torsoLean`은 없어졌어요(앱에서 쓰는 곳 없음).

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

### 5. 개인 캘리브레이션 (지금은 쓰지 않음 · 화면 만들 필요 없음)

**2026-11-18 결정: 지금은 앱에서 쓰지 않아요.** 3명·4회 촬영 데이터에서 같은 기준이 정확도 96.5%, 개인 기준이 77%였어요. 바른 자세에서도 기울기가 0~14°로 흔들려서 5초 기준값이 평소 자세를 대표하지 못했어요. 코드는 남겨 두고, 테스트 인원이 늘면 분석 리포트에서 다시 비교해요.
아래는 나중에 다시 켤 때를 위한 사용법이에요.

사람마다 평소 앉는 자세가 달라서, 절대 기준만 쓰면 원래 머리가 앞에 있는 사람은 계속 경고를 받아요. 시작할 때 바른 자세로 5초 앉으면 그 자세를 기준으로 판정해요. 합성 데이터에서는 정확도가 81.7%에서 88.0%로 올랐지만 실측은 아니에요.

```kotlin
// 세팅 화면에서 카운트다운이 끝난 뒤(또는 설정의 '기준 자세 다시 재기')
val collector = CalibrationCollector(sessionThresholds)
// 프레임마다: 캘리브레이션 없는 판정기의 결과를 넣음
val result = collector.add(lastFrameAt, feedback)
if (result.status == CalibrationStatus.PENDING) showProgress(result.progress, result.status.message)
else if (result.status == CalibrationStatus.DONE) {
    preferences.saveCalibration(result.calibration!!)                  // Calibration은 @Serializable
    estimator = PostureEstimator(sessionThresholds, result.calibration) // 이후 판정은 개인 기준
} else showRetry(result.status.message)                                // NOT_VISIBLE / MOVING / NOT_UPRIGHT
```

- 저장한 기준은 다음 세션에도 쓸 수 있어요. 다만 폰 위치나 의자가 바뀌면 다시 재는 게 좋아요(재세팅 안내가 뜰 때 함께).
- `PoseFeedback.calibrated`로 개인 기준 판정 중인지 화면에 표시할 수 있어요.
- 사용자가 건너뛰거나 계속 실패하면 절대 임계값으로 판정해요.

### 6. 임계값

새 판정기는 머리 특징의 정의가 달라졌어요. 그래서 실험 데이터로 정한 **v2 임계값**을 서버에 등록한 뒤 교체하는 게 가장 좋아요. 지금 기본값(0.35 / 20°)과 서버 v1(0.25 / 20°)은 둘 다 실험 전 임시값이에요.

---

### 7. 처리 지연 측정 (실기기 확인용)

발표 목표 "판정 1회 처리 p95 ≤ 200ms"를 재는 도구예요. 프레임 분석 시작부터 판정기 결과가 나올 때까지(비트맵 변환 + MediaPipe + 판정기)를 재요.

1. `kotlin/src/main/kotlin/com/sidefocus/core/PostureLatency.kt`를 `core/src/main/kotlin/com/sidefocus/core/`에 복사해요.
2. `FocusViewModel.kt`에 두 줄을 넣어요. `frame.observation.timestampMs`가 `CameraSurface`에서 분석을 시작한 시각(`now`)이라 그대로 쓰면 돼요.
   ```kotlin
   private val latency = LatencyMeter()

   fun frame(frame: CameraFrame) {
       ...
       feedback = estimator.update(frame.observation)
       latency.add(SystemClock.elapsedRealtime() - frame.observation.timestampMs, frame.observation.timestampMs)  // 추가
       ...
   }
   ```
3. 세션이 끝날 때(또는 디버그 화면에서) 결과를 남겨요.
   ```kotlin
   Log.i("SideFocus", latency.summary().report())
   latency.reset()
   ```
   Logcat 예: `처리 지연 n=1500 p50=62ms p95=118ms max=240ms 200ms 초과 3회 · 분석 5.0fps (목표 p95 ≤ 200ms: 통과)`
4. 실기기에서 **10분 이상** 켜 두고 잰 p95를 AI 담당에게 알려 주세요. 평가 리포트에 `--device-p95`로 넣어요.
   - 같은 세션에서 배터리 %도 시작·끝을 적어 두면 "1시간 배터리 ≤ 15%"도 계산할 수 있어요.

## 백엔드 (`threshold_configs`)

- 기본 판정에는 스키마 변경이 **필요 없어요.** `t2_angle`이 골반→귀 선 기울기 임계값(도)이고, `t1_ratio`는 판정에 안 써요(NOT NULL이라 아무 값이나 그대로 두면 돼요).
- 실험 데이터로 `python -m posture_ai analyze`를 돌리면 `outputs/real/thresholds.sql`이 나와요. 새 버전 행을 넣고 활성 버전을 바꾸는 SQL이에요(기존 버전은 수정하지 않음).
- **지금 등록할 v2 (2026-11-18, 3명·4회 촬영):** `t2_angle` = 15.0. `t1_ratio`는 판정에 안 써서 0.35를 그대로 넣어요.
  고개 숙임 기준(42°)은 서버 칸이 없어서 앱 코드 기본값을 써요. 나중에 서버에서 바꾸고 싶으면 `t3_pitch` 컬럼을 추가하면 돼요(선택).
  ```sql
  BEGIN;
  INSERT INTO threshold_configs (version, t1_ratio, t2_angle, stable_sec, analysis_fps, is_active)
  VALUES (2, 0.35, 15.0, 2.0, 5, false);
  UPDATE threshold_configs SET is_active = false WHERE is_active;
  UPDATE threshold_configs SET is_active = true WHERE version = 2;
  COMMIT;
  ```
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

### 개인 캘리브레이션 지원 (지금은 필요 없음, F8과 연결)

캘리브레이션을 앱에서 쓰지 않기로 해서(2026-11-18) 아래 작업은 **당분간 안 해도 돼요.**

- **기준값 저장:** `user_settings.calib_angle` = `Calibration.axisAngle`(바른 자세의 골반→귀 기울기). `calib_ratio`는 안 써요. 테이블은 이미 있어요. 기획서의 `GET/PUT /users/me/settings`가 구현되면 앱이 백업·복원할 수 있어요.
- **Δ 임계값 (필요):** 캘리브레이션했을 때 쓰는 임계값 `t2_delta` **하나**만 `threshold_configs`에 nullable REAL 컬럼으로 추가하고 `/config` 응답에 넣어 주세요(V3 마이그레이션). 값이 없으면 앱은 기본값(9.5°)을 써요. `t1_delta`는 필요 없어졌어요.
  ```sql
  ALTER TABLE threshold_configs ADD COLUMN t2_delta REAL;
  ```

---

## 팀 결정 사항 (2026-11-18 회의)

1. **상태 전환:** 최근 2초 창에서 70% 이상 같은 판정이면 전환해요(발표 13쪽 방식). "2초 연속 유지" 방식은 쓰지 않아요.
2. **판정 방식:** ① 골반→귀 선이 수직에서 앞으로 기운 각도(부호 있음, 뒤로 기대면 음수)와 ② 고개 숙임(귀→코 선이 수평 아래로 숙인 각도) 중 하나라도 기준을 넘으면 거북목이에요. 1차 실험에서 머리·몸통을 따로 보던 방식보다 정확했어요(정확도 74% → 97%, 3명 예비 결과).
3. **상체 기울임도 거북목:** 머리만 나오든, 허리만 숙이든, 둘 다든 상태는 전부 `FHP`(거북목)예요. 판정기는 `LEAN`을 내보내지 않아요.
   - 📱 프론트: 고치지 않아도 동작해요. 원하면 범례·데모 버튼·리포트에서 `LEAN`(상체 기울임)을 빼세요(`Components.kt`, `SessionScreens.kt` 등). 안내 문구는 "턱을 당기고 허리를 펴 보세요" 하나예요.
   - 🖥 백엔드: 고칠 것 없어요. enum의 `LEAN`은 남겨 둬도 값이 안 들어올 뿐이에요.
   - 결과보고서에는 "상체 기울임은 거북목으로 통합해 집계"라고 적어요.
4. **개인 캘리브레이션:** ~~넣어요~~ → **지금은 쓰지 않아요**(같은 날 데이터 분석 후 변경). **2026-11-18 결정: 지금은 앱에서 쓰지 않아요.** 3명·4회 촬영 데이터에서 같은 기준이 정확도 96.5%, 개인 기준이 77%였어요. 바른 자세에서도 기울기가 0~14°로 흔들려서 5초 기준값이 평소 자세를 대표하지 못했어요. 코드는 남겨 두고, 테스트 인원이 늘면 분석 리포트에서 다시 비교해요.
5. **서버 기준값 v2:** `t2_angle` = 15°(골반→귀 기울기), 고개 숙임 42°(앱 기본값). 앱 시험용 임시값이고, 테스트 인원이 늘면 다시 정해요. SQL은 위 백엔드 절에 있어요.
6. **고개만 숙여도 거북목:** 허리는 펴고 고개만 숙인 자세도 거북목이에요. 골반→귀 기울기로는 잘 안 잡혀서 고개 숙임(귀→코) 각도를 함께 봐요.
   3명·4회 촬영에서 정확도 96.5% → 97.3%, 거북목 놓침 0%. ⚠️ 책·노트를 내려다보며 공부할 때도 거북목으로 잡힐 수 있어서, 파일럿 테스트에서 오탐을 꼭 확인해요.

## 아직 미정

1. **판정기 교체 시점:** 실험 후 v2 임계값이 나온 뒤에 교체하는 걸 권장해요.
2. **ML 모델 탑재:** 발표 3-4 기준(Macro F1 +0.05, 피험자 과반 개선, 처리 p95 ≤ 200ms)을 만족할 때만 ONNX 모델을 앱에 넣어요. 그 전까지는 규칙 판정기만 써요.
