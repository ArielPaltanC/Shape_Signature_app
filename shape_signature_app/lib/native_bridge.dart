import 'package:flutter/services.dart';

class NativeBridge {
  static const _channel = MethodChannel('com.shapesignature.app/processor');

  static Future<List<double>?> processImage(String imagePath) async {
    try {
      final result = await _channel.invokeMethod('processImage', {
        'imagePath': imagePath,
      });
      if (result == null) return null;
      return List<double>.from(result as List);
    } catch (e) {
      rethrow;
    }
  }

  static Future<List<double>?> processContour(
      List<double> xPoints, List<double> yPoints) async {
    try {
      final result = await _channel.invokeMethod('processContour', {
        'xPoints': xPoints,
        'yPoints': yPoints,
      });
      if (result == null) return null;
      return List<double>.from(result as List);
    } catch (e) {
      rethrow;
    }
  }

  static Future<List<double>?> detectContour(String imagePath) async {
    try {
      final result = await _channel.invokeMethod('detectContour', {
        'imagePath': imagePath,
      });
      if (result == null) return null;
      return List<double>.from(result as List);
    } catch (e) {
      rethrow;
    }
  }
}
