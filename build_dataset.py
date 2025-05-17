# build_dataset.py
# 用于从标注 JSON + 图片中构建颜色主色训练数据集（掩码区域图像块 + RGB 标签）

import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# 配置路径
ANNOTATION_DIR = "src/color_annotator/annotations"
IMAGE_DIR = "src/color_annotator/images"
OUTPUT_FILE = "data/dataset_rgb_region.npz"

# 掩码太小的区域丢弃
MIN_MASK_AREA = 10 * 10

# 存储数据
samples = []
labels = []


def decode_rle(rle, shape):
    """RLE 解码为掩码"""
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    for start, length in rle:
        flat[start:start + length] = True
    return flat.reshape(shape)


for file in tqdm(os.listdir(ANNOTATION_DIR)):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(ANNOTATION_DIR, file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rel_img_path = data.get("image_path")
    img_path = os.path.join(IMAGE_DIR, os.path.basename(rel_img_path))

    if not os.path.exists(img_path):
        print(f"[跳过] 图片不存在: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[跳过] 图像读取失败: {img_path}")
        continue

    for ann in data.get("annotations", []):
        rle = ann["rle"]
        size = tuple(ann["size"])
        color = ann["main_color"]

        mask = decode_rle(rle, size)
        if np.sum(mask) < MIN_MASK_AREA:
            continue  # 掩码区域太小，跳过

        # 裁剪掩码区域（含上下边界）
        y_indices, x_indices = np.where(mask)
        top, bottom = np.min(y_indices), np.max(y_indices)
        left, right = np.min(x_indices), np.max(x_indices)

        cropped = img[top:bottom + 1, left:right + 1]
        resized = cv2.resize(cropped, (64, 64))  # 统一大小，利于训练

        samples.append(resized)
        labels.append(color)

samples = np.array(samples, dtype=np.uint8)  # (N, 64, 64, 3)
labels = np.array(labels, dtype=np.uint8)    # (N, 3)

print(f"[完成] 样本数：{len(samples)}")
np.savez_compressed(OUTPUT_FILE, images=samples, labels=labels)
print(f"[保存] 已保存至：{OUTPUT_FILE}")