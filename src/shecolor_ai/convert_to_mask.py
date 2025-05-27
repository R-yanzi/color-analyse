import os
import json
import numpy as np
import cv2
from tqdm import tqdm

from src.shecolor_ai.convert_rle_to_coco import rle_decode


def convert_json_to_masks():
    root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    ann_dir = os.path.join(root, "datasets", "shecolor", "annotations")
    image_dir = os.path.join(root, "datasets", "shecolor", "images")
    mask_dir = os.path.join(root, "datasets", "shecolor", "masks")
    os.makedirs(mask_dir, exist_ok=True)

    colored_dir = os.path.join(root, "datasets", "shecolor", "masks_colored")
    os.makedirs(colored_dir, exist_ok=True)

    print("[生成中] 每张图的主色掩码标签图")
    for filename in tqdm(os.listdir(ann_dir)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ann_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = os.path.join(image_dir, os.path.basename(data["image_path"]))
        if not os.path.exists(image_path):
            print(f"[跳过] 找不到图像文件 {image_path}")
            continue

        h, w = data["annotations"][0]["size"]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)  # 彩色掩码初始化

        for idx, ann in enumerate(data["annotations"], start=1):
            decoded = rle_decode(ann["rle"], ann["size"])
            mask[decoded == 1] = idx

            color = ann.get("main_color", [0, 0, 0])
            if len(color) == 4:
                color = color[:3]  # 去掉 alpha
            color_bgr = color[::-1]  # 把 [R, G, B] 变成 [B, G, R]
            mask_color[decoded == 1] = color_bgr

        save_name = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(mask_dir, save_name), mask)
        cv2.imwrite(os.path.join(colored_dir, save_name), mask_color)


    print("[完成] 所有掩码图已生成完毕")

if __name__ == "__main__":
    convert_json_to_masks()
