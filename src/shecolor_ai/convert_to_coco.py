import os
import json
import cv2
import numpy as np
from tqdm import tqdm

from pycocotools import mask as mask_utils

def convert():
    # ==== 目录配置 ====
    PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    ANNOT_DIR = os.path.join(PROJECT_ROOT, "datasets", "shecolor", "annotations")
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets", "shecolor", "images")
    COCO_OUTPUT = os.path.join(PROJECT_ROOT, "datasets", "shecolor", "coco")
    os.makedirs(COCO_OUTPUT, exist_ok=True)

    # ==== COCO 初始化 ====
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "main_color"}]
    }

    ann_id = 1
    image_id = 1

    print("[转换中] 原始标注 -> COCO 格式")

    for filename in tqdm(sorted(os.listdir(ANNOT_DIR))):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(ANNOT_DIR, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_filename = data.get("imagePath", filename.replace(".json", ".png"))
        image_path = os.path.join(IMAGE_DIR, image_filename)
        height = data.get("imageHeight", 512)
        width = data.get("imageWidth", 512)

        coco["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        print(f"[调试] 当前文件：{filename}")

        for shape in data.get("shapes", []):


            print(f"{filename} shape points:", shape.get("points", []))
            points = shape.get("points", [])
            if len(points) < 3:
                print(f"[跳过] {filename} 中有非法 shape（少于3点）")
                continue

            polygon = np.array(points, dtype=np.float32)
            segmentation = [polygon.flatten().tolist()]

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)

            encoded = mask_utils.encode(np.asfortranarray(mask))
            area = float(mask_utils.area(encoded))
            bbox = mask_utils.toBbox(encoded).tolist()

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "color": shape.get("color", [0, 0, 0, 255])  # 可选保留颜色信息
            })
            ann_id += 1

        image_id += 1

    # ==== 保存 COCO JSON ====
    with open(os.path.join(COCO_OUTPUT, "instances_train.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print("[完成] 已生成 COCO 格式标注：instances_train.json")

if __name__ == "__main__":
    convert()