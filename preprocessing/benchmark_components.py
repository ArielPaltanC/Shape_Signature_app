import cv2
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

GROUP_CLASSES = [3, 4, 5, 6, 7, 8, 9]

def compute_shape_signature(image_path, n_components):
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
    max_idx = min(n_components + 1, len(fft_normalized))
    descriptor = fft_normalized[1:max_idx].tolist()
    while len(descriptor) < n_components:
        descriptor.append(0.0)
    return descriptor

def load_descriptors(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['train'], data['test']

def main():
    json_path = "../shape_signature_app/assets/descriptors.json"
    dataset_path = Path("../Leaves")
    
    train_data, test_data = load_descriptors(json_path)
    
    print(f"Test images: {len(test_data)}")
    print(f"\n{'N':>3} | {'Acc':>7} | {'Correctos':>9}")
    print("-" * 25)
    
    results = []
    for n in [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]:
        correct = 0
        total = 0
        for item in test_data:
            img_path = dataset_path / item['image']
            desc = compute_shape_signature(img_path, n)
            if desc is None:
                continue
            
            best_dist = float('inf')
            best_class = -1
            for t in train_data:
                t_desc = t['descriptor']
                min_len = min(len(desc), len(t_desc))
                dist = np.linalg.norm(np.array(desc[:min_len]) - np.array(t_desc[:min_len]))
                if dist < best_dist:
                    best_dist = dist
                    best_class = t['class']
            
            total += 1
            if best_class == item['class']:
                correct += 1
        
        acc = correct / total * 100 if total > 0 else 0
        results.append((n, acc, correct, total))
        print(f"{n:>3} | {acc:>6.2f}% | {correct:>3}/{total}")
    
    best = max(results, key=lambda r: r[1])
    print(f"\nMejor N: {best[0]} con {best[1]:.2f}% ({best[2]}/{best[3]})")

if __name__ == '__main__':
    main()
