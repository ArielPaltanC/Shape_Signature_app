# Shape Signature — Clasificador de hojas por descriptores de Fourier

App Flutter + C++ (OpenCV) que identifica especies de hojas a partir de su contorno usando **descriptores de forma basados en la Transformada de Fourier (FFT)**.

---

## Arquitectura

```
Flutter (Dart)                    Android (Kotlin)              C++ (OpenCV)
┌─────────────────────┐          ┌──────────────────┐         ┌─────────────────────┐
│  LeafClassifierScreen │ ──────→ │  MainActivity.kt  │ ──────→ │ native_processor.cpp │
│  (UI + dibujo)       │  k-NN   │  (MethodChannel)  │  JNI    │ (OpenCV + DFT)      │
│  Classifier (Dart)   │ ←────── │                   │ ←────── │                     │
└─────────────────────┘          └──────────────────┘         └─────────────────────┘
```

- **Flutter**: UI, selector de imagen, canvas de dibujo, clasificador k-NN (k=3)
- **Kotlin**: Puente MethodChannel entre Dart y C++
- **C++/OpenCV**: Procesamiento de imagen, detección de contornos, DFT, cálculo de descriptores

---

## Flujo de la app

```
1. Seleccionar imagen ──→ 2. Detectar contorno ──→ 3. Clasificar ──→ 4. Resultado
   (galería del          (automático con         (C++ procesa el     (especie +
    celular)              Otsu + findContours)    contorno y          descriptores
                                                  calcula            de Fourier)
                                                  descriptores)
```

### 1. Preprocesamiento (OpenCV)

1. **Escala de grises** — `cv::cvtColor(img, gray, COLOR_BGRA2GRAY)`
2. **Filtro Gaussiano** — `cv::GaussianBlur(gray, blurred, Size(5,5), 1.5)` para reducir ruido
3. **Umbral Otsu** — `cv::threshold(blurred, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU)` para binarizar
4. **Operaciones morfológicas** — `MORPH_CLOSE` y `MORPH_OPEN` para limpiar el contorno
5. **Detección de contornos** — `cv::findContours()` con `RETR_EXTERNAL`

### 2. Descriptor de Fourier (Firma de forma)

El contorno más grande se procesa así:

1. **Centroide** — `cv::moments()` calcula el centro de masa
2. **Señal compleja** — Cada punto del contorno se representa como `(x - cx) + i(y - cy)`
3. **FFT** — `cv::dft()` aplica la Transformada Rápida de Fourier
4. **Normalización** — Las magnitudes se dividen por `|F(1)|` (primera componente no DC) para invarianza a escala
5. **Se descarta la fase** para invarianza a rotación
6. **Descriptor** — Vector de 12 componentes (magnitudes de Fourier normalizadas) + 1 de compactness

### 3. Clasificador (k-NN, k=3)

- Compara el descriptor de entrada contra todos los del dataset (`shape_signature_app/assets/descriptors.json`)
- Usa **distancia euclidiana**: `sqrt(sum((desc_i - train_i)^2))`
- Vota entre los 3 vecinos más cercanos
- Devuelve la clase de la muestra más cercana (clases 3–9 del dataset Flavia)

---

## Dataset: Flavia

- 1907 imágenes de hojas escaneadas sobre fondo blanco
- 32 especies diferentes (uso subset: clases 3–9)
- Entrenamiento: 340 descriptores, Prueba: 89 (split 80/20)
- Descriptor: vector de 13 componentes float (12 Fourier + 1 compactness)

| Clase | Especie                     |
|-------|------------------------------|
| 3     | Cercis chinensis             |
| 4     | Viburnum awabuki             |
| 5     | Chimonanthus praecox         |
| 6     | Cedrus deodara               |
| 7     | Ilex cornuta                 |
| 8     | Photinia serrulata           |
| 9     | Toona sinensis               |

---

## Requisitos

- Flutter SDK 3.44.3+
- Android NDK 27+
- Android SDK 35+
- OpenCV 4.x (incluido en `shape_signature_app/android/app/src/main/jniLibs/`)

---

## Cómo ejecutar

### 1. Compilar APK

```bash
cd shape_signature_app
flutter build apk --split-per-abi
```

Los APKs se generan en:
```
build/app/outputs/flutter-apk/
  app-arm64-v8a-release.apk   ← para la mayoría de celulares modernos
  app-armeabi-v7a-release.apk  ← para dispositivos 32 bits
  app-x86_64-release.apk       ← para emuladores
```

### 2. Instalar en celular

Con cable USB y depuración USB activada:

```bash
adb install shape_signature_app/build/app/outputs/flutter-apk/app-arm64-v8a-release.apk
```

O copia el APK al celular e instálalo manualmente.

### 3. Probar

1. Abre la app → "Seleccionar imagen"
2. Elige una foto de una hoja desde la galería
3. La app detecta el contorno automáticamente
4. Presiona **"Clasificar forma"**
5. La app muestra la especie y los descriptores de Fourier

> **Consejo:** Usa hojas de forma simple sobre fondo liso. Para mejores resultados, el fondo debe contrastar con la hoja.

---

## Estructura del proyecto

```
.
├── shape_signature_app/
│   ├── lib/
│   │   ├── main.dart                          # Punto de entrada
│   │   ├── leaf_classifier_screen.dart         # UI principal (imagen + dibujo + resultados)
│   │   ├── classifier.dart                    # Clasificador k-NN (distancia euclidiana)
│   │   └── native_bridge.dart                 # Comunicación con C++ via MethodChannel
│   ├── android/
│   │   ├── app/
│   │   │   ├── src/main/
│   │   │   │   ├── cpp/
│   │   │   │   │   ├── native_processor.h      # Cabecera C++ (JNI + constantes)
│   │   │   │   │   ├── native_processor.cpp    # Implementación (OpenCV, DFT, Otsu)
│   │   │   │   │   └── CMakeLists.txt          # Build de librería nativa
│   │   │   │   ├── java/com/shapesignature/app/
│   │   │   │   │   └── MainActivity.kt         # MethodChannel + JNI + EXIF rotation
│   │   │   │   └── AndroidManifest.xml
│   │   │   └── build.gradle
│   │   └── settings.gradle
│   ├── assets/
│   │   └── descriptors.json                   # Descriptores de entrenamiento (340)
│   └── pubspec.yaml
├── preprocessing/
│   ├── generate_descriptors.py                # Generar descriptores desde Flavia
│   └── validate_model.py                      # Validación con matriz de confusión
└── Leaves/                                    # Dataset Flavia (no commiteado)
```

---


### Resultados de precisión

| Métrica        | Precision (94.38%) |
|----------------|---------------|
| Clase 3        | 93.3%         |
| Clase 4        | 100%          |
| Clase 5        | 100%          |
| Clase 6        | 92.3%         |
| Clase 7        | 90.9%         |
| Clase 8        | 90.9%         |
| Clase 9        | 83.3%         |
