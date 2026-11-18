package com.sidefocus.core

import kotlinx.serialization.Serializable

/**
 * 컴파일용 사본입니다. 프론트에는 복사하지 마세요.
 * 실제 정의는 frontend/core/src/main/kotlin/com/sidefocus/core/Session.kt 에 있습니다.
 */
@Serializable
enum class Posture(val label: String) {
    NORMAL("바른 자세"), FHP("머리가 앞으로"), LEAN("상체 기울임"),
    AWAY("자리 비움"), UNKNOWN("자세 확인 중"), PAUSED("일시정지")
}
