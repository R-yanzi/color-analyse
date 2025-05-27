# 位置：src/utils/generate_masks.py

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def decode_rle(rle, shape):
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in rle:
        flat[start:start + length] = 1
    return flat.reshape(shape)

def generate_all_masks(anno_dir="annotations", image_dir="images", output_dir="datasets"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)
    os.makedirs(f"{output_dir}/masks_colored", exist_ok=True)

    for file in tqdm(os.listdir(anno_dir)):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(anno_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = os.path.join(image_dir, os.path.basename(data["image_path"]))
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)         # 类别图（训练用）
        mask_colored = np.zeros((h, w, 3), dtype=np.uint8)  # 彩色可视化图

        for idx, ann in enumerate(data["annotations"], start=1):
            decoded = decode_rle(ann["rle"], tuple(ann["size"]))
            if decoded.shape != (h, w):
                decoded = cv2.resize(decoded.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            binary_mask = decoded.astype(bool)
            mask[binary_mask] = idx

            # 可视化颜色填主色
            color = ann.get("main_color", [0, 255, 0])
            mask_colored[binary_mask] = (color[2], color[1], color[0])  # 转 BGR

        base_name = os.path.splitext(file)[0]
        cv2.imwrite(f"{output_dir}/images/{base_name}.png", image)
        cv2.imwrite(f"{output_dir}/masks/{base_name}.png", mask)
        cv2.imwrite(f"{output_dir}/masks_colored/{base_name}.png", mask_colored)

        print(f"[✓] {base_name}.png 掩码生成完毕（类别: {np.unique(mask)})")

    print("✅ 所有掩码图像已保存完毕")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    anno_dir = os.path.join(project_root, "src", "color_annotator", "annotations")
    image_dir = os.path.join(project_root, "src", "color_annotator", "images")
    output_dir = os.path.join(project_root, "src", "datasets")

    generate_all_masks(anno_dir, image_dir, output_dir)
