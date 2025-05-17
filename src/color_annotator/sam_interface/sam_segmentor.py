# sam_interface/sam_segmentor.py

import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

class SAMSegmentor:
    def __init__(self, model_type="vit_b", checkpoint_path="checkpoints/sam_vit_b.pth", device="cuda"):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)

        # 初始化属性
        self.original_shape = None
        self.resized_image = None
        self.scale_factor = 1.0

    def set_image(self, image: np.ndarray, max_size=1024):
        """设置图像给 SAM，并自动 resize 到合适大小"""
        self.original_shape = image.shape[:2]  # (h, w)
        h, w = self.original_shape

        if max(h, w) <= max_size:
            self.resized_image = image
            self.scale_factor = 1.0
        else:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            self.scale_factor = 1.0 / scale  # 注意这里是：原图坐标 = 缩放后坐标 * scale_factor

        self.predictor.set_image(self.resized_image)

    def predict_from_point(self, point: tuple):
        """处理单点"""
        scaled_point = (point[0] / self.scale_factor, point[1] / self.scale_factor)
        input_point = np.array([scaled_point])
        input_label = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        return self._resize_mask_back(masks[0])

    def predict_from_points(self, points: np.ndarray, labels: np.ndarray):
        """处理多点"""
        # 💡 点也要缩放到 resized 尺寸
        scaled_points = points / self.scale_factor

        masks, scores, logits = self.predictor.predict(
            point_coords=scaled_points,
            point_labels=labels,
            multimask_output=False
        )
        return self._resize_mask_back(masks[0])

    def _resize_mask_back(self, mask):
        """将分割结果 resize 回原图大小"""
        if self.scale_factor != 1.0:
            h, w = self.original_shape
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return mask.astype(bool)
        return mask
