import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils

def rle_decode(rle, size):
    """解码自定义的 RLE 为二值掩码"""
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    for start, length in rle:
        mask[start:start+length] = 1
    return mask.reshape(size)

def convert_rle_to_coco():
    PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    ANNOT_DIR = os.path.join(PROJECT_ROOT, "datasets", "shecolor", "annotations")
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets", "shecolor", "read_images")
    COCO_OUTPUT = os.path.join(PROJECT_ROOT, "datasets", "shecolor", "coco")
    os.makedirs(COCO_OUTPUT, exist_ok=True)

    coco = {
        "read_images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "main_color"}]
    }

    ann_id = 1
    image_id = 1

    print("[转换中] RLE 标注 → COCO 格式")

    for filename in tqdm(sorted(os.listdir(ANNOT_DIR))):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ANNOT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = os.path.basename(data["image_path"])
        image_full = os.path.join(IMAGE_DIR, image_path)
        if not os.path.exists(image_full):
            print(f"[跳过] 未找到图片文件 {image_path}")
            continue

        height, width = data["annotations"][0]["size"]
        coco["read_images"].append({
            "id": image_id,
            "file_name": image_path,
            "width": width,
            "height": height
        })

        for ann in data["annotations"]:
            mask = rle_decode(ann["rle"], ann["size"])
            encoded = mask_utils.encode(np.asfortranarray(mask))
            area = float(mask_utils.area(encoded))
            bbox = mask_utils.toBbox(encoded).tolist()

            # 从 mask 中提取轮廓（polygon）
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                if len(contour) >= 3:
                    seg = contour.flatten().tolist()
                    segmentation.append(seg)

            if not segmentation:
                continue  # 跳过无效区域

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "main_color": ann.get("main_color", [0, 0, 0])
            })
            ann_id += 1

        image_id += 1

    with open(os.path.join(COCO_OUTPUT, "instances_train.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[完成] 共导出 {len(coco['annotations'])} 个区域，保存至 instances_train.json")

if __name__ == "__main__":
    convert_rle_to_coco()
