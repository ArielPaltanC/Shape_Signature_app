import cv2
import numpy as np
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

GROUP_CLASSES = [3, 4, 5, 6, 7, 8, 9]

def compute_full_descriptor(image_path, max_n=100):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    n = len(contour)
    complex_signal = np.zeros(n, dtype=np.complex64)
    for i in range(n):
        complex_signal[i] = complex(contour[i][0] - cx, contour[i][1] - cy)
    fft_result = np.fft.fft(complex_signal)
    fft_mag = np.abs(fft_result)
    if fft_mag[1] == 0:
        return None
    fft_normalized = fft_mag / fft_mag[1]
    max_idx = min(max_n + 1, len(fft_normalized))
    descriptor = fft_normalized[1:max_idx].tolist()
    while len(descriptor) < max_n:
        descriptor.append(0.0)
    return descriptor

def knn_classify(descriptor, train_data, k=3):
    distances = []
    for item in train_data:
        t_desc = item['descriptor']
        min_len = min(len(descriptor), len(t_desc))
        dist = np.linalg.norm(np.array(descriptor[:min_len]) - np.array(t_desc[:min_len]))
        distances.append((dist, item['class']))
    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    classes = [c for _, c in top_k]
    most_common = Counter(classes).most_common(1)
    return most_common[0][0]

def main():
    json_path = "../shape_signature_app/assets/descriptors.json"
    dataset_path = Path("../Leaves")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Recompute full descriptors for training data (100 components)
    print("Recomputing training descriptors with 100 components...")
    train_data = []
    for item in data['train']:
        img_path = dataset_path / item['image']
        desc = compute_full_descriptor(img_path, 100)
        if desc is not None:
            train_data.append({'class': item['class'], 'descriptor': desc})
    
    test_data = data['test']
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    print("\nProbando combinaciones de N componentes y k vecinos:")
    print(f"{'N':>4} {'k=1':>8} {'k=3':>8} {'k=5':>8} {'k=7':>8}")
    print("-" * 40)
    
    results = []
    for n in [5, 8, 10, 12, 15, 20, 30]:
        best_k_acc = 0
        best_k = 1
        for k in [1, 3, 5, 7]:
            correct = 0
            total = 0
            for item in test_data:
                img_path = dataset_path / item['image']
                desc = compute_full_descriptor(img_path, n)
                if desc is None:
                    continue
                pred = knn_classify(desc, [{**t, 'descriptor': t['descriptor'][:n]} for t in train_data], k=k)
                total += 1
                if pred == item['class']:
                    correct += 1
            acc = correct / total * 100 if total > 0 else 0
            if acc > best_k_acc:
                best_k_acc = acc
                best_k = k
            print(f"{n:>4} | {acc:>6.2f}%", end="")
        print()
    
    # Also try weighted k-NN (by inverse distance)
    print("\n--- Weighted k-NN (por distancia inversa) ---")
    for n in [5, 8, 10, 12, 15, 20, 30]:
        for k in [3, 5]:
            correct = 0
            total = 0
            for item in test_data:
                img_path = dataset_path / item['image']
                desc = compute_full_descriptor(img_path, n)
                if desc is None:
                    continue
                
                distances = []
                for t in train_data:
                    t_desc = t['descriptor'][:n]
                    min_len = min(len(desc), len(t_desc))
                    dist = np.linalg.norm(np.array(desc[:min_len]) - np.array(t_desc[:min_len]))
                    distances.append((dist, t['class']))
                distances.sort(key=lambda x: x[0])
                top_k = distances[:k]
                
                weights = {}
                for dist, cls in top_k:
                    w = 1.0 / (dist + 1e-10)
                    weights[cls] = weights.get(cls, 0) + w
                
                pred = max(weights, key=weights.get)
                total += 1
                if pred == item['class']:
                    correct += 1
            
            acc = correct / total * 100 if total > 0 else 0
            print(f"  N={n:>2}, k={k} (weighted): {acc:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
