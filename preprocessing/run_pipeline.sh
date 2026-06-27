#!/bin/bash
set -e

# Config - CHANGE THESE PATHS
FLAVIA_DATASET="/path/to/flavia/images"
FLAVIA_CSV="/path/to/flavia/all.csv"

echo "=== Shape Signature - Pipeline de Procesamiento ==="
echo "Grupo G6 - Clases: 3, 4, 5, 6, 7, 8, 9"
echo ""

# Step 1: Generate descriptors
echo "[1/2] Generando descriptores desde Flavia dataset..."
python3 generate_descriptors.py \
    --dataset "$FLAVIA_DATASET" \
    --csv "$FLAVIA_CSV" \
    --output "../shape_signature_app/assets/descriptors.json"

echo ""
echo "[2/2] Ejecutando validacion..."
python3 validate_model.py \
    "../shape_signature_app/assets/descriptors.json" \
    "$FLAVIA_DATASET"

echo ""
echo "=== Pipeline completado ==="
