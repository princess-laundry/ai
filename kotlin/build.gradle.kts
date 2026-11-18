// 프론트 core 모듈에 그대로 넣을 판정기(PostureEstimator.kt, PostureFeatures.kt)를 단독으로 빌드·검증합니다.
// Kotlin·serialization·JDK 버전은 frontend와 같게 맞춥니다.
plugins {
    kotlin("jvm") version "2.1.20"
    kotlin("plugin.serialization") version "2.1.20"
}

kotlin { jvmToolchain(21) }

// Windows에서 프로젝트 경로에 한글이 있으면(예: "바탕 화면") Gradle 테스트 프로세스가 클래스를 못 찾습니다.
// 그럴 때만 빌드 결과물을 ASCII 경로(사용자 임시 폴더)에 둡니다. 소스·golden 파일 위치는 그대로입니다.
if (projectDir.absolutePath.any { it.code > 127 }) {
    val tmp = File(System.getProperty("java.io.tmpdir"))
    if (tmp.absolutePath.all { it.code <= 127 }) layout.buildDirectory.set(tmp.resolve("posture-engine-build"))
}

val goldenDir = layout.projectDirectory.dir("../fixtures/golden")
val frontendCore = layout.projectDirectory.dir(providers.gradleProperty("frontendCore").getOrElse("../../frontend/core"))

sourceSets {
    main { kotlin.srcDir("src/stub/kotlin") }   // Posture enum 사본(프론트에서는 Session.kt에 있음)
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
    testImplementation(kotlin("test-junit"))
}

tasks.test {
    systemProperty("golden.dir", goldenDir.asFile.absolutePath)
    inputs.dir(goldenDir)
    testLogging { events("failed"); exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
}

// ---------------------------------------------------------------------------------------------
// verifyAgainstFrontend: 프론트 core 소스(Session.kt 등)와 우리 판정기를 함께 컴파일하고,
// 프론트 core 테스트 + 우리 테스트를 돌려 "파일만 바꿔 넣으면 된다"는 것을 확인합니다.
// 프론트 폴더는 읽기만 합니다(build/ 아래로 복사해서 씀).
// ---------------------------------------------------------------------------------------------
val syncFrontendMain by tasks.registering(Sync::class) {
    // 프론트에 들어간 (옛 버전) AI 파일은 빼고, 이 저장소의 최신 AI 파일로 컴파일해요
    from(frontendCore.dir("src/main/kotlin")) { exclude("**/PostureEstimator.kt", "**/PostureFeatures.kt", "**/PostureCalibration.kt") }
    from("src/main/kotlin")
    into(layout.buildDirectory.dir("compat/main"))
}
val syncFrontendTest by tasks.registering(Sync::class) {
    from(frontendCore.dir("src/test/kotlin")) { exclude("**/PostureEstimatorTest.kt") }
    from("src/test/kotlin")
    into(layout.buildDirectory.dir("compat/test"))
}

val compat by sourceSets.creating {
    kotlin.srcDir(syncFrontendMain)
}
val compatTest by sourceSets.creating {
    kotlin.srcDir(syncFrontendTest)
    compileClasspath += compat.output
    runtimeClasspath += compat.output
}
dependencies {
    "compatImplementation"("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
    "compatTestImplementation"("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
    "compatTestImplementation"(kotlin("test-junit"))
}

tasks.register<Test>("verifyAgainstFrontend") {
    group = "verification"
    description = "프론트 core 소스·테스트와 함께 컴파일하고 테스트합니다 (-PfrontendCore=경로 로 위치 변경 가능)."
    onlyIf("frontend core 폴더가 있어야 합니다") { frontendCore.dir("src/main/kotlin").asFile.exists() }
    testClassesDirs = compatTest.output.classesDirs
    classpath = compatTest.runtimeClasspath
    systemProperty("golden.dir", goldenDir.asFile.absolutePath)
    testLogging { events("passed", "failed"); exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
}
