import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';
import 'native_bridge.dart';

class LeafClassifierScreen extends StatefulWidget {
  const LeafClassifierScreen({super.key});

  @override
  State<LeafClassifierScreen> createState() => _LeafClassifierScreenState();
}

class _LeafClassifierScreenState extends State<LeafClassifierScreen> {
  final Classifier _classifier = Classifier();
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  ui.Image? _displayImage;
  List<Offset> _contourPoints = [];
  Size _imageSize = Size.zero;
  Size _displaySize = Size.zero;
  Offset _imageOffset = Offset.zero;

  String _especieClasificada = '-';
  String _descriptores = 'Esperando captura...';
  bool _isProcessing = false;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initClassifier();
  }

  Future<void> _initClassifier() async {
    await _classifier.loadDescriptors();
    setState(() => _isInitialized = true);
  }

  Future<void> _pickImage() async {
    final xfile = await _picker.pickImage(source: ImageSource.gallery);
    if (xfile == null) return;

    final file = File(xfile.path);
    final bytes = await file.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;

    setState(() {
      _selectedImage = file;
      _displayImage = image;
      _imageSize = Size(image.width.toDouble(), image.height.toDouble());
      _contourPoints = [];
      _especieClasificada = '-';
      _descriptores = 'Detectando contorno...';
      _isProcessing = true;
    });

    await _detectAndClassify(file.path);
  }

  Future<void> _detectAndClassify(String imagePath) async {
    try {
      final contourData = await NativeBridge.detectContour(imagePath);
      if (contourData == null || contourData.isEmpty) {
        setState(() {
          _descriptores = 'No se detectó ningún contorno';
          _isProcessing = false;
        });
        return;
      }

      final contour = <Offset>[];
      for (int i = 0; i < contourData.length; i += 2) {
        contour.add(Offset(contourData[i], contourData[i + 1]));
      }

      final xs = contour.map((p) => p.dx.toDouble()).toList();
      final ys = contour.map((p) => p.dy.toDouble()).toList();

      final descriptor = await NativeBridge.processContour(xs, ys);

      setState(() {
        _contourPoints = contour;
        _isProcessing = false;
      });

      if (descriptor != null && _classifier.isReady) {
        final predictedClass = _classifier.classify(descriptor);
        setState(() {
          _especieClasificada = '$predictedClass';
          _descriptores = descriptor.map((d) => d.toStringAsFixed(4)).join(', ');
        });
      } else if (descriptor != null) {
        setState(() => _descriptores = 'Cargando descriptores...');
      } else {
        setState(() => _descriptores = 'No se pudo procesar el contorno');
      }
    } catch (e) {
      setState(() {
        _descriptores = 'Error: $e';
        _isProcessing = false;
      });
    }
  }

  void _clearAll() {
    setState(() {
      _selectedImage = null;
      _displayImage = null;
      _contourPoints = [];
      _especieClasificada = '-';
      _descriptores = 'Esperando captura...';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('Shape Signature - G6'),
        backgroundColor: Colors.lightBlue[200],
        actions: [
          if (_selectedImage != null)
            IconButton(
              icon: const Icon(Icons.clear),
              onPressed: _clearAll,
              tooltip: 'Limpiar',
            ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              if (_selectedImage == null) ...[
                const Spacer(),
                Icon(Icons.eco, size: 80, color: Colors.teal[300]),
                const SizedBox(height: 16),
                const Text(
                  'Selecciona una imagen de hoja\npara clasificarla automáticamente',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.grey),
                ),
                const SizedBox(height: 24),
                ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.lightBlue[300],
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  icon: const Icon(Icons.image),
                  label: const Text('Seleccionar imagen',
                      style: TextStyle(fontSize: 16, color: Colors.black87)),
                  onPressed: _pickImage,
                ),
                const Spacer(),
              ] else ...[
                Expanded(
                  flex: 3,
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final maxW = constraints.maxWidth;
                      final maxH = constraints.maxHeight;
                      final imgW = _imageSize.width;
                      final imgH = _imageSize.height;
                      final scale = (imgW / imgH > maxW / maxH)
                          ? maxW / imgW
                          : maxH / imgH;
                      final dispW = imgW * scale;
                      final dispH = imgH * scale;
                      final offX = (maxW - dispW) / 2;
                      final offY = (maxH - dispH) / 2;

                      _displaySize = Size(dispW, dispH);
                      _imageOffset = Offset(offX, offY);

                      return ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.blueAccent, width: 2),
                          ),
                          child: Stack(
                            children: [
                              if (_displayImage != null)
                                Positioned(
                                  left: offX,
                                  top: offY,
                                  child: SizedBox(
                                    width: dispW,
                                    height: dispH,
                                    child: RawImage(
                                      image: _displayImage,
                                      fit: BoxFit.fill,
                                    ),
                                  ),
                                ),
                              if (_contourPoints.isNotEmpty)
                                CustomPaint(
                                  size: Size(maxW, maxH),
                                  painter: _ContourPainter(
                                    contourPoints: _contourPoints,
                                    imageOffset: _imageOffset,
                                    displaySize: _displaySize,
                                    imageSize: _imageSize,
                                  ),
                                ),
                              if (_isProcessing)
                                const Center(
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      CircularProgressIndicator(),
                                      SizedBox(height: 8),
                                      Text('Procesando...',
                                          style: TextStyle(color: Colors.white)),
                                    ],
                                  ),
                                ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 12),
                Expanded(
                  flex: 2,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'ESPECIE\nCLASIFICADA: $_especieClasificada',
                        style: const TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 8),
                      Expanded(
                        child: SingleChildScrollView(
                          child: Text(
                            'Descriptor de Fourier:\n$_descriptores',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.lightBlue[300],
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30),
                          ),
                        ),
                        icon: const Icon(Icons.image),
                        label: const Text('Nueva imagen',
                            style: TextStyle(fontSize: 16, color: Colors.black87)),
                        onPressed: _pickImage,
                      ),
                    ),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _ContourPainter extends CustomPainter {
  final List<Offset> contourPoints;
  final Offset imageOffset;
  final Size displaySize;
  final Size imageSize;

  _ContourPainter({
    required this.contourPoints,
    required this.imageOffset,
    required this.displaySize,
    required this.imageSize,
  });

  Offset _imageToScreen(Offset imgPt) {
    final sx = imageOffset.dx + (imgPt.dx / imageSize.width) * displaySize.width;
    final sy = imageOffset.dy + (imgPt.dy / imageSize.height) * displaySize.height;
    return Offset(sx, sy);
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (contourPoints.isEmpty) return;

    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final fillPaint = Paint()
      ..color = Colors.red.withOpacity(0.15)
      ..style = PaintingStyle.fill;

    final path = Path();
    final first = _imageToScreen(contourPoints.first);
    path.moveTo(first.dx, first.dy);

    for (int i = 1; i < contourPoints.length; i++) {
      final pt = _imageToScreen(contourPoints[i]);
      path.lineTo(pt.dx, pt.dy);
    }
    path.close();

    canvas.drawPath(path, fillPaint);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _ContourPainter oldDelegate) => true;
}
