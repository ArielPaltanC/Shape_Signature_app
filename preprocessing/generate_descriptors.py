import cv2
import numpy as np
import os
import json
import random
import argparse
from pathlib import Path

GROUP_CLASSES = [3, 4, 5, 6, 7, 8, 9]
N_DESCRIPTOR_COMPONENTS = 13

def load_flavia_csv(csv_path):
    images = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            count, img_id, label = parts[0], parts[1], int(parts[2])
            images.append((img_id, label))
    return images

def compute_shape_signature(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour[:, 0, :]
    M = cv2.moments(contour_pts)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    n = len(contour_pts)
    complex_signal = np.zeros(n, dtype=np.complex64)
    for i in range(n):
        complex_signal[i] = complex(contour_pts[i][0] - cx, contour_pts[i][1] - cy)
    fft_result = np.fft.fft(complex_signal)
    fft_mag = np.abs(fft_result)
    if fft_mag[1] == 0:
        return None
    fft_normalized = fft_mag / fft_mag[1]
    descriptor = fft_normalized[1:N_DESCRIPTOR_COMPONENTS].tolist()
    compactness = cv2.contourArea(contour) / (cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3])
    if cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3] == 0:
        compactness = 0.0
    descriptor.append(compactness)
    return descriptor

def main():
    parser = argparse.ArgumentParser(description='Generate Shape Signature descriptors for Flavia dataset')
    parser.add_argument('--dataset', required=True, help='Path to Flavia dataset root directory')
    parser.add_argument('--csv', required=True, help='Path to all.csv')
    parser.add_argument('--output', default='descriptors.json', help='Output JSON file path')
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    all_images = load_flavia_csv(args.csv)
    class_images = {c: [] for c in GROUP_CLASSES}
    for img_id, label in all_images:
        if label in GROUP_CLASSES:
            class_images[label].append(img_id)

    random.seed(42)
    train_data = []
    test_data = []
    for cls in GROUP_CLASSES:
        imgs = class_images[cls]
        random.shuffle(imgs)
        split_idx = int(len(imgs) * 0.8)
        train_imgs = imgs[:split_idx]
        test_imgs = imgs[split_idx:]
        for img_id in train_imgs:
            img_path = dataset_root / img_id
            desc = compute_shape_signature(img_path)
            if desc is not None:
                train_data.append({'image': img_id, 'class': cls, 'descriptor': desc})
        for img_id in test_imgs:
            test_data.append({'image': img_id, 'class': cls})

    output = {
        'classes': GROUP_CLASSES,
        'n_components': N_DESCRIPTOR_COMPONENTS,
        'train': train_data,
        'test': test_data
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(train_data)} training descriptors and {len(test_data)} test images to {args.output}")
    print(f"Descriptor dimension: {N_DESCRIPTOR_COMPONENTS}")

if __name__ == '__main__':
    main()
