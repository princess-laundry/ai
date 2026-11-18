package com.sidefocus.core

import kotlinx.serialization.json.*
import java.io.File
import kotlin.math.abs
import kotlin.test.*

/**
 * Python(PC 분석)과 Kotlin(앱)이 같은 입력에 같은 판정을 내는지 확인합니다.
 * 입력·기대 출력은 ai 저장소의 `python -m posture_ai golden`이 만든 fixtures/golden/\*.json 입니다.
 * 이 테스트를 프론트 저장소에 복사했다면 golden 폴더가 없으므로 건너뜁니다.
 */
class GoldenParityTest {
    private val dir = File(System.getProperty("golden.dir") ?: "../fixtures/golden")

    private fun point(e: JsonElement?): Point? =
        if (e == null || e is JsonNull) null
        else e.jsonArray.let { Point(it[0].jsonPrimitive.double, it[1].jsonPrimitive.double, it[2].jsonPrimitive.float) }

    private fun side(e: JsonElement?): BodySide? =
        if (e == null || e is JsonNull) null else e.jsonArray.let { BodySide(point(it[0])!!, point(it[1])!!, point(it[2])!!) }

    private fun JsonObject.doubleOrNull(key: String) = this[key]?.jsonPrimitive?.doubleOrNull
    private fun JsonObject.boolOrNull(key: String) = this[key]?.jsonPrimitive?.booleanOrNull

    private fun assertClose(expected: Double?, actual: Double?, what: String) {
        if (expected == null) { assertNull(actual, what); return }
        assertNotNull(actual, what)
        assertTrue(abs(expected - actual) <= 1e-6, "$what: expected $expected but was $actual")
    }

    private fun files() = dir.listFiles { f -> f.extension == "json" }?.sortedBy { it.name }.orEmpty()
    private fun read(file: File) = Json.parseToJsonElement(file.readText()).jsonObject

    private fun thresholds(t: JsonObject): Thresholds {
        val defaults = Thresholds()
        return Thresholds(
            t["version"]!!.jsonPrimitive.int, t["headRatio"]!!.jsonPrimitive.double,
            t["leanDegrees"]!!.jsonPrimitive.double, t["stableMs"]!!.jsonPrimitive.long, t["analysisFps"]!!.jsonPrimitive.int,
            t.doubleOrNull("leanDeltaDegrees") ?: defaults.leanDeltaDegrees,
            t.doubleOrNull("pitchDegrees") ?: defaults.pitchDegrees,
            t.doubleOrNull("pitchDeltaDegrees") ?: defaults.pitchDeltaDegrees)
    }

    private fun observation(f: JsonObject) = PoseObservation(
        f["t"]!!.jsonPrimitive.long, f["person"]!!.jsonPrimitive.boolean,
        side(f["left"]), side(f["right"]), point(f["nose"]), f.doubleOrNull("roll"))

    @Test fun `kotlin calibration collector matches python golden outputs`() {
        val file = File(dir, "calibration_collector.json")
        if (!file.exists()) { println("golden 파일 없음: ${file.absolutePath} — 건너뜀"); return }
        val cases = read(file)["cases"]!!.jsonArray
        assertEquals(4, cases.size)
        val seen = mutableSetOf<String>()
        for (element in cases) {
            val case = element.jsonObject
            val t = thresholds(case["thresholds"]!!.jsonObject)
            val estimator = PostureEstimator(t)
            val collector = CalibrationCollector(t)
            val expected = case["expected"]!!.jsonArray
            case["frames"]!!.jsonArray.forEachIndexed { i, frame ->
                val f = frame.jsonObject
                val result = collector.add(f["t"]!!.jsonPrimitive.long, estimator.update(observation(f)))
                val e = expected[i].jsonObject
                val where = "calibration ${case["name"]!!.jsonPrimitive.content} frame $i"
                assertEquals(e["status"]!!.jsonPrimitive.content, result.status.name, "$where status")
                assertClose(e.doubleOrNull("progress"), result.progress, "$where progress")
                assertClose(e.doubleOrNull("axisAngle"), result.calibration?.axisAngle, "$where axisAngle")
                assertClose(e.doubleOrNull("headPitch"), result.calibration?.headPitch, "$where headPitch")
                if (result.finished) seen += result.status.name
            }
        }
        assertEquals(setOf("DONE", "MOVING", "NOT_UPRIGHT", "NOT_VISIBLE"), seen)
    }

    @Test fun `kotlin estimator matches python golden outputs`() {
        val files = files().filter { it.name != "calibration_collector.json" }
        if (files.isEmpty()) { println("golden 파일 없음: ${dir.absolutePath} — 건너뜀"); return }
        for (file in files) {
            val root = read(file)
            val calibration = (root["calibration"] as? JsonObject)?.let {
                Calibration(it["axisAngle"]!!.jsonPrimitive.double, it.doubleOrNull("headPitch"))
            }
            val estimator = PostureEstimator(thresholds(root["thresholds"]!!.jsonObject), calibration)
            val frames = root["frames"]!!.jsonArray
            val expected = root["expected"]!!.jsonArray
            assertEquals(frames.size, expected.size)
            frames.forEachIndexed { i, element ->
                val f = element.jsonObject
                val result = estimator.update(observation(f))
                val e = expected[i].jsonObject
                val where = "${file.name} frame $i (t=${f["t"]})"
                assertEquals(e["state"]!!.jsonPrimitive.content, result.state.name, "$where state")
                assertEquals(e["needsReset"]!!.jsonPrimitive.boolean, result.needsReset, "$where needsReset")
                assertEquals(e["framed"]!!.jsonPrimitive.boolean, result.framed, "$where framed")
                assertEquals(e["facing"]!!.jsonPrimitive.int, result.facing, "$where facing")
                assertEquals(e.boolOrNull("calibrated") ?: false, result.calibrated, "$where calibrated")
                assertClose(e.doubleOrNull("headRatio"), result.headRatio, "$where headRatio")
                assertClose(e.doubleOrNull("axisAngle"), result.leanDegrees, "$where axisAngle")
                assertClose(e.doubleOrNull("headPitch"), result.headPitch, "$where headPitch")
                assertClose(e.doubleOrNull("torsoAngle"), result.torsoAngle, "$where torsoAngle")
                assertClose(e.doubleOrNull("neckAngle"), result.neckAngle, "$where neckAngle")
            }
        }
    }
}
