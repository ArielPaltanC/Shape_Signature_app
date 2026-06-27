import cv2
import numpy as np
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

GROUP_CLASSES = [3, 4, 5, 6, 7, 8, 9]

def compute_shape_signature(image_path, n_components, use_canny=False):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    if use_canny:
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
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


def compute_contour_area_ratio(image_path):
    """Compute the ratio of contour area to bounding box area (compactness feature)"""
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
    area = cv2.contourArea(contour)
    if area == 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    if bbox_area == 0:
        return None
    return area / bbox_area


def main():
    json_path = "../shape_signature_app/assets/descriptors.json"
    dataset_path = Path("../Leaves")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("1. Baseline (1-NN, N=12): 49.44%")
    print("\n2. Probando mejoras...\n")
    
    # Test: k-NN + z-score normalization
    print("--- k-NN con Z-score normalization ---")
    for n in [10, 12, 15]:
        # Compute descriptors for train + test
        train_raw = []
        for item in data['train']:
            img_path = dataset_path / item['image']
            desc = compute_shape_signature(img_path, n)
            if desc is not None:
                train_raw.append(np.array(desc))
        
        train_arr = np.array(train_raw)
        mean = np.mean(train_arr, axis=0)
        std = np.std(train_arr, axis=0) + 1e-10
        train_normalized = [(train_arr[i] - mean) / std for i in range(len(train_arr))]
        
        test_items = data['test']
        for k in [3, 5]:
            correct = 0
            total = 0
            for item in test_items:
                img_path = dataset_path / item['image']
                desc = compute_shape_signature(img_path, n)
                if desc is None:
                    continue
                desc_norm = (np.array(desc) - mean) / std
                
                distances = []
                for i, t_desc in enumerate(train_normalized):
                    dist = np.linalg.norm(desc_norm - t_desc)
                    distances.append((dist, data['train'][i]['class']))
                distances.sort(key=lambda x: x[0])
                top_k = distances[:k]
                classes = [c for _, c in top_k]
                pred = Counter(classes).most_common(1)[0][0]
                
                total += 1
                if pred == item['class']:
                    correct += 1
            
            acc = correct / total * 100 if total > 0 else 0
            print(f"  N={n:>2}, k={k}, z-score: {acc:.2f}% ({correct}/{total})")
    
    # Test: Canny edge detection
    print("\n--- Canny edge detection ---")
    for n in [10, 12]:
        train_raw = []
        for item in data['train']:
            img_path = dataset_path / item['image']
            desc = compute_shape_signature(img_path, n, use_canny=True)
            if desc is not None:
                train_raw.append({'desc': np.array(desc), 'cls': item['class']})
        
        test_items = data['test']
        for k in [1, 3]:
            correct = 0
            total = 0
            for item in test_items:
                img_path = dataset_path / item['image']
                desc = compute_shape_signature(img_path, n, use_canny=True)
                if desc is None:
                    continue
                
                distances = []
                for t in train_raw:
                    dist = np.linalg.norm(np.array(desc) - t['desc'])
                    distances.append((dist, t['cls']))
                distances.sort(key=lambda x: x[0])
                top_k = distances[:k]
                classes = [c for _, c in top_k]
                pred = Counter(classes).most_common(1)[0][0]
                
                total += 1
                if pred == item['class']:
                    correct += 1
            
            acc = correct / total * 100 if total > 0 else 0
            print(f"  N={n:>2}, k={k}, Canny: {acc:.2f}% ({correct}/{total})")
    
    # Test: Combine Fourier + compactness feature
    print("\n--- Fourier + Compactness (area/bbox ratio) ---")
    for n in [10, 12]:
        train_data_aug = []
        for item in data['train']:
            img_path = dataset_path / item['image']
            desc = compute_shape_signature(img_path, n)
            comp = compute_contour_area_ratio(img_path)
            if desc is not None and comp is not None:
                train_data_aug.append({'desc': np.array(desc + [comp]), 'cls': item['class']})
        
        test_items = data['test']
        for k in [3, 5]:
            correct = 0
            total = 0
            for item in test_items:
                img_path = dataset_path / item['image']
                desc = compute_shape_signature(img_path, n)
                comp = compute_contour_area_ratio(img_path)
                if desc is None or comp is None:
                    continue
                
                test_vec = np.array(desc + [comp])
                distances = []
                for t in train_data_aug:
                    dist = np.linalg.norm(test_vec - t['desc'])
                    distances.append((dist, t['cls']))
                distances.sort(key=lambda x: x[0])
                top_k = distances[:k]
                classes = [c for _, c in top_k]
                pred = Counter(classes).most_common(1)[0][0]
                
                total += 1
                if pred == item['class']:
                    correct += 1
            
            acc = correct / total * 100 if total > 0 else 0
            print(f"  N={n:>2}, k={k}, +compactness: {acc:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
