import 'package:flutter/material.dart';
import 'leaf_classifier_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ShapeSignatureApp());
}

class ShapeSignatureApp extends StatelessWidget {
  const ShapeSignatureApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shape Signature',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.teal,
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      home: const LeafClassifierScreen(),
    );
  }
}
