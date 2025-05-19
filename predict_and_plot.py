import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from train_color_model import ColorCNN

# 配置路径
MODEL_PATH = "color_model.pt"
DEVICE = torch.device("cpu")
SAM_CHECKPOINT = "src/color_annotator/checkpoints/sam_vit_b.pth"
MODEL_TYPE = "vit_b"
CROP_SIZE = 64
OUTPUT_DIR = "outputs"
TEST_IMAGE_DIR = "src/color_annotator/test_images"

def load_model():
    model = ColorCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def set_up_sam():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    return predictor

def predict_color(model, crop):
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    x = torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x.to(DEVICE)).squeeze().cpu().numpy()
        rgb = (out * 255).clip(0, 255).astype(int)
    return rgb.tolist()

def main(image_path):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"[错误] 无法读取图像 {image_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    predictor = set_up_sam()

    h, w = orig_img.shape[:2]
    scale = 1024 / max(h, w)
    resized_img = cv2.resize(orig_img, (int(w * scale), int(h * scale)))
    predictor.set_image(resized_img)

    input_points = []
    input_labels = []
    for y in range(100, resized_img.shape[0] - 100, 200):
        for x in range(100, resized_img.shape[1] - 100, 200):
            input_points.append([x, y])
            input_labels.append(1)

    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    color_stats = []
    total_area = 0
    for i, mask in enumerate(masks):
        filename = os.path.basename(image_path).rsplit(".", 1)[0]
        full_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if np.sum(full_mask) < 500:
            continue

        # 可视化 1：叠加原图轮廓
        overlay = orig_img.copy()
        mask_vis = full_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{OUTPUT_DIR}/region_overlay_{filename}_{i}.jpg", overlay)

        # 可视化 2：裁剪区域
        crop = orig_img.copy()
        crop[~full_mask] = 0
        x, y, w_, h_ = cv2.boundingRect(full_mask.astype(np.uint8))
        region = crop[y:y + h_, x:x + w_]
        cv2.imwrite(f"{OUTPUT_DIR}/region_crop_{filename}_{i}.jpg", region)

        # 主色预测
        rgb = predict_color(model, region)
        area = np.sum(full_mask)
        color_stats.append((rgb, area))
        total_area += area

        # 可视化 3：区域 + 主色合图
        region_resized = cv2.resize(region, (64, 64))
        color_patch = np.full((64, 64, 3), rgb, dtype=np.uint8)
        combo = np.hstack([region_resized, color_patch])
        cv2.imwrite(f"{OUTPUT_DIR}/region_and_color_{filename}_{i}.jpg", combo)

        print(f"区域 {i} 主色预测：{rgb}，面积：{area}")

    if not color_stats:
        print("[提示] 无有效区域被处理")
        return

    # 饼图可视化
    labels, sizes, colors = [], [], []
    for i, (rgb, area) in enumerate(color_stats):
        ratio = area / total_area
        labels.append(f"{rgb} ({ratio:.1%})")
        sizes.append(ratio)
        colors.append(np.array(rgb) / 255.0)

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, counterclock=False)
    ax.axis('equal')
    plt.title("主色提取饼图")
    save_path = f"{OUTPUT_DIR}/debug_pie_{os.path.basename(image_path)}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[完成] 饼图保存于：{save_path}")

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"[错误] 测试目录不存在：{TEST_IMAGE_DIR}")
        exit(1)

    test_images = [f for f in os.listdir(TEST_IMAGE_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

    if not test_images:
        print("[提示] 测试目录中没有图片")
    else:
        print(f"[开始] 共检测到 {len(test_images)} 张图片")
        for img in test_images:
            print(f"\n🖼️ 处理图像：{img}")
            main(os.path.join(TEST_IMAGE_DIR, img))
