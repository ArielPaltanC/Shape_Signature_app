package com.shapesignature.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.shapesignature.app/processor"

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "processImage" -> {
                        val imagePath = call.argument<String>("imagePath")
                        if (imagePath != null) {
                            val descriptor = processImageNative(imagePath)
                            if (descriptor != null) {
                                result.success(descriptor.toList())
                            } else {
                                result.error("PROCESS_ERROR", "Failed to process image", null)
                            }
                        } else {
                            result.error("INVALID_ARGS", "imagePath required", null)
                        }
                    }
                    "processContour" -> {
                        val xPoints = call.argument<List<Double>>("xPoints")?.map { it.toFloat() }?.toFloatArray()
                        val yPoints = call.argument<List<Double>>("yPoints")?.map { it.toFloat() }?.toFloatArray()
                        if (xPoints != null && yPoints != null) {
                            val descriptor = nativeProcessContour(xPoints, yPoints)
                            if (descriptor != null) {
                                result.success(descriptor.toList())
                            } else {
                                result.error("PROCESS_ERROR", "Failed to process contour", null)
                            }
                        } else {
                            result.error("INVALID_ARGS", "xPoints and yPoints required", null)
                        }
                    }
                    "detectContour" -> {
                        val imagePath = call.argument<String>("imagePath")
                        if (imagePath != null) {
                            val contour = detectContourNative(imagePath)
                            if (contour != null) {
                                result.success(contour.toList())
                            } else {
                                result.error("PROCESS_ERROR", "Failed to detect contour", null)
                            }
                        } else {
                            result.error("INVALID_ARGS", "imagePath required", null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun loadBitmapOriented(imagePath: String): Bitmap? {
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
            inPremultiplied = true
        }
        var bitmap = BitmapFactory.decodeFile(imagePath, opts) ?: return null

        val exifRotation = try {
            val exif = ExifInterface(imagePath)
            when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        } catch (e: Exception) {
            0f
        }

        if (exifRotation != 0f) {
            val matrix = Matrix().apply { postRotate(exifRotation) }
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }

        return bitmap
    }

    private fun bitmapToPixels(bitmap: Bitmap): Triple<IntArray, Int, Int> {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return Triple(pixels, width, height)
    }

    private fun processImageNative(imagePath: String): FloatArray? {
        val bitmap = loadBitmapOriented(imagePath) ?: return null
        val (pixels, width, height) = bitmapToPixels(bitmap)
        return nativeProcessImage(pixels, width, height)
    }

    private fun detectContourNative(imagePath: String): FloatArray? {
        val bitmap = loadBitmapOriented(imagePath) ?: return null
        val (pixels, width, height) = bitmapToPixels(bitmap)
        return nativeDetectContour(pixels, width, height)
    }

    private external fun nativeProcessImage(pixels: IntArray, width: Int, height: Int): FloatArray?
    private external fun nativeProcessContour(xPoints: FloatArray, yPoints: FloatArray): FloatArray?
    private external fun nativeDetectContour(pixels: IntArray, width: Int, height: Int): FloatArray?

    companion object {
        init {
            System.loadLibrary("native_processor")
        }
    }
}
