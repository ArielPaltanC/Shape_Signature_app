import cv2
import numpy as np
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from generate_descriptors import compute_shape_signature, GROUP_CLASSES, N_DESCRIPTOR_COMPONENTS

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def classify(descriptor, train_data):
    best_dist = float('inf')
    best_class = -1
    for item in train_data:
        dist = euclidean_distance(descriptor, item['descriptor'])
        if dist < best_dist:
            best_dist = dist
            best_class = item['class']
    return best_class, best_dist

def generate_confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t]][class_to_idx[p]] += 1
    return cm

def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_model.py <descriptors.json> <dataset_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    dataset_path = Path(sys.argv[2])

    with open(json_path, 'r') as f:
        data = json.load(f)

    train_data = data['train']
    test_data = data['test']
    classes = data['classes']

    print(f"Testing {len(test_data)} images against {len(train_data)} training descriptors...")

    y_true = []
    y_pred = []
    confusions = []

    for item in test_data:
        img_path = dataset_path / item['image']
        desc = compute_shape_signature(img_path)
        if desc is None:
            print(f"  Warning: could not process {item['image']}")
            continue
        predicted_class, distance = classify(desc, train_data)
        y_true.append(item['class'])
        y_pred.append(predicted_class)
        if predicted_class != item['class']:
            confusions.append({
                'image': item['image'],
                'true_class': item['class'],
                'predicted_class': predicted_class,
                'distance': distance
            })

    cm = generate_confusion_matrix(y_true, y_pred, classes)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Precision global: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*50}\n")

    print("Matriz de Confusion:")
    print(f"{'':>8}", end="")
    for c in classes:
        print(f"{c:>6}", end="")
    print()
    for i, c in enumerate(classes):
        print(f"Real {c:>2}:", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>6}", end="")
        total_real = np.sum(cm[i])
        acc_class = cm[i][i] / total_real * 100 if total_real > 0 else 0
        print(f"  ({acc_class:.1f}%)")

    print(f"\nPrecision por clase:")
    for i, c in enumerate(classes):
        total_real = np.sum(cm[i])
        acc_class = cm[i][i] / total_real * 100 if total_real > 0 else 0
        print(f"  Clase {c}: {acc_class:.1f}% ({cm[i][i]}/{total_real})")

    confusions.sort(key=lambda x: x['distance'], reverse=True)
    print(f"\nEjemplos de formas que confunde (top 10):")
    for c in confusions[:10]:
        print(f"  {c['image']}: real={c['true_class']} -> pred={c['predicted_class']} (dist={c['distance']:.4f})")

if __name__ == '__main__':
    main()
