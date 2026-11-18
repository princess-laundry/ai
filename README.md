# ai — 측면 자세 인식 AI 파트

측면 자세 인식 기반 집중 관리 앱의 **자세 판정 로직**과 **PC 분석 도구**를 담은 저장소예요.
앱(frontend)과 PC가 똑같은 전처리·판정 규칙을 쓰고, 두 구현이 실제로 같은 결과를 내는지 자동 테스트로 확인해요.

```
ai/
├── docs/
│   ├── feature-spec.md        전처리·특징값·판정 규칙 (앱·PC 공통 기준)
│   ├── recording-protocol.md  실험 촬영 방법과 대본
│   ├── labeling-guide.md      정답 라벨 기준, Cohen's κ
│   └── integration.md         프론트·백엔드 팀원 연동 안내 ← 팀원은 여기부터
├── python/                    PC 분석 도구 (posture_ai 패키지)
├── kotlin/                    프론트 core에 넣을 판정기 + 테스트
└── fixtures/golden/           Python ↔ Kotlin 일치 검증 데이터
```

## 무엇을 하나요

1. **판정기** (Python·Kotlin 동일)
   - 귀·어깨·골반 좌표로 앞쪽 방향을 통일하고, 거치 기울기를 보정해요.
   - 몸통 기준 머리 비율과 몸통 각도를 계산해요.
   - 머리 전방·상체 기울임을 따로 판정한 뒤, 2초 창에서 70% 이상 일치하면 상태를 바꿔요.
   - 둘 중 하나라도 해당하면 상태는 **거북목(FHP)**이에요. 상체 기울임(LEAN)을 따로 내보내지 않아요(팀 결정).
   - **개인 캘리브레이션:** 시작할 때 바른 자세로 5초 앉으면 그 사람 기준(Δ)으로 판정해요. 움직였거나, 안 보였거나, 처음부터 구부정했으면 다시 하라고 안내해요. 결과는 백엔드 `user_settings.calib_ratio`·`calib_angle`에 대응해요.
2. **PC 분석:** 실험 영상 → MediaPipe 랜드마크(5fps) → 2초 창 샘플 → 다음 비교 → 리포트
   - 비교: 임계값 / SVM / RF × 절대 / Δ특징, 피험자 단위 LOSO
   - 리포트에 들어가는 것: 임계값 SQL, ONNX 모델, 앱 시뮬레이션(오탐·미탐·지연)
3. **합성 데이터 데모:** 영상이 없어도 전체 흐름을 끝까지 돌려 볼 수 있어요. **나오는 수치는 실제 성능이 아니에요.**

## 시작하기 (Windows)

```powershell
cd python
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest            # Python 테스트
.venv\Scripts\python -m posture_ai demo   # 합성 데이터로 전체 파이프라인 → outputs/demo/report.md
```

Kotlin 판정기 테스트(JDK 21 필요):

```powershell
cd kotlin
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"   # JAVA_HOME이 잘못 잡혀 있을 때만
.\gradlew test                    # 단위 테스트 + Python golden 일치 테스트
.\gradlew verifyAgainstFrontend   # 옆 폴더 ../frontend/core 소스·테스트와 함께 컴파일·테스트
```

> 경로에 한글이 있으면 Gradle 테스트가 클래스를 못 찾는 문제가 있어서, 빌드 결과물은 자동으로 사용자 임시 폴더에 생겨요.

## 실제 데이터로 분석하기

촬영 방법은 [recording-protocol.md](docs/recording-protocol.md), 라벨 기준은 [labeling-guide.md](docs/labeling-guide.md)에 있어요.

```powershell
cd python
.venv\Scripts\python -m posture_ai model                       # MediaPipe lite 모델 준비(앱과 같은 파일인지 SHA 확인)
.venv\Scripts\python -m posture_ai extract                     # data/videos → data/landmarks (5fps)
.venv\Scripts\python -m posture_ai label-script S1_R --start 4.2 --repeats 3
.venv\Scripts\python -m posture_ai visualize S1_R              # 스켈레톤·상태·라벨을 겹친 검수 영상
.venv\Scripts\python -m posture_ai kappa A.csv B.csv           # 라벨러 일치도
.venv\Scripts\python -m posture_ai analyze --version 2         # → outputs/real/report.md, thresholds.sql, model_*.onnx
```

`data/`와 `outputs/`는 git에 올라가지 않아요. 실험 영상은 동의를 받은 뒤 팀 PC에만 보관하고, 프로젝트가 끝나면 삭제해요.

## 판정 규칙을 고칠 때

1. `python/src/posture_ai/features.py`·`estimator.py`를 수정해요.
2. `python -m posture_ai golden`으로 golden 파일을 다시 만들어요.
3. `kotlin/.../PostureFeatures.kt`·`PostureEstimator.kt`를 같은 규칙으로 고쳐요.
4. `pytest`와 `gradlew test`가 모두 통과해야 해요. golden이 오래됐거나 두 언어 결과가 다르면 테스트가 실패해요.
5. [feature-spec.md](docs/feature-spec.md)도 함께 고쳐요.

## 현재 상태와 한계

- 실험 영상이 아직 없어서 **정확도·오탐·지연 수치는 측정 전**이에요. demo 리포트는 합성 데이터라 성능 근거가 될 수 없어요.
- MediaPipe 추출은 실제 모델로 동작하는 것까지만 확인했어요(빈 화면 영상 테스트).
- 기기 처리 지연(ms), 배터리, 발열은 실기기에서 따로 측정해야 해요.
- 앱 반영은 프론트 담당자가 [integration.md](docs/integration.md)를 따라 진행해요.
