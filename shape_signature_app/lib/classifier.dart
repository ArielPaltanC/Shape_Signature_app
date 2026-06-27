import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';

class Classifier {
  List<Map<String, dynamic>> _trainData = [];
  List<int> _classes = [];
  bool _ready = false;
  int _k = 3;

  Future<void> loadDescriptors() async {
    final jsonStr = await rootBundle.loadString('assets/descriptors.json');
    final data = json.decode(jsonStr);
    _classes = List<int>.from(data['classes']);
    _trainData = List<Map<String, dynamic>>.from(data['train']);
    _ready = true;
  }

  bool get isReady => _ready;

  int classify(List<double> descriptor) {
    if (_trainData.isEmpty) return -1;

    List<_Distance> distances = [];
    for (final item in _trainData) {
      final trainDesc = List<double>.from(item['descriptor']);
      final dist = _euclideanDistance(descriptor, trainDesc);
      distances.add(_Distance(dist, item['class'] as int));
    }

    distances.sort((a, b) => a.dist.compareTo(b.dist));

    Map<int, int> votes = {};
    int k = _k < distances.length ? _k : distances.length;
    for (int i = 0; i < k; i++) {
      votes[distances[i].label] = (votes[distances[i].label] ?? 0) + 1;
    }

    int bestLabel = -1;
    int bestCount = 0;
    votes.forEach((label, count) {
      if (count > bestCount) {
        bestCount = count;
        bestLabel = label;
      }
    });

    return bestLabel;
  }

  double _euclideanDistance(List<double> a, List<double> b) {
    double sum = 0;
    for (int i = 0; i < a.length && i < b.length; i++) {
      sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
  }
}

class _Distance {
  final double dist;
  final int label;
  _Distance(this.dist, this.label);
}
