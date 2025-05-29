# sam_interface/sam_segmentor.py

import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path

class SAMSegmentor:
    def __init__(self, model_type="vit_b", checkpoint_path=None):
        if checkpoint_path is None:
            # 获取当前文件所在目录
            current_dir = Path(__file__).resolve().parent.parent
            checkpoint_path = str(current_dir / "checkpoints" / "sam_vit_b.pth")
            print(f"[SAM] 使用默认模型路径: {checkpoint_path}")
        
        try:
            print(f"[SAM] 加载模型: {model_type} from {checkpoint_path}")
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print("[SAM] 模型加载成功")
        except Exception as e:
            print(f"[SAM] 加载模型失败: {str(e)}")
            raise

        self.predictor = SamPredictor(self.sam)

        # 初始化属性
        self.original_shape = None
        self.resized_image = None
        self.scale_factor = 1.0

    def set_image(self, image: np.ndarray, max_size=1024):
        """设置图像给 SAM，并自动 resize 到合适大小"""
        try:
            if image is None:
                raise ValueError("输入图像为空")

            if len(image.shape) != 3:
                raise ValueError(f"图像维度不正确，期望3维，实际{len(image.shape)}维")

            print(f"[SAM] 原始图像尺寸: {image.shape}")
            self.original_shape = image.shape[:2]  # (h, w)
            h, w = self.original_shape

            if max(h, w) <= max_size:
                self.resized_image = image
                self.scale_factor = 1.0
                print("[SAM] 图像尺寸合适，无需缩放")
            else:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                self.resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self.scale_factor = 1.0 / scale  # 注意这里是：原图坐标 = 缩放后坐标 * scale_factor
                print(f"[SAM] 图像已缩放至 {new_w}x{new_h}，缩放因子: {scale:.2f}")

            # 确保图像是RGB格式
            if len(self.resized_image.shape) == 2:
                self.resized_image = cv2.cvtColor(self.resized_image, cv2.COLOR_GRAY2RGB)
            elif self.resized_image.shape[2] == 4:
                self.resized_image = self.resized_image[:, :, :3]

            print("[SAM] 设置图像到预测器...")
            self.predictor.set_image(self.resized_image)
            print("[SAM] 图像设置完成")

        except Exception as e:
            print(f"[SAM] 设置图像时出错: {str(e)}")
            raise

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

    def predict_from_points(self, points: np.ndarray, labels: np.ndarray, multimask_output=False):
        """处理多点"""
        # 💡 点也要缩放到 resized 尺寸
        scaled_points = points / self.scale_factor

        masks, scores, logits = self.predictor.predict(
            point_coords=scaled_points,
            point_labels=labels,
            multimask_output=multimask_output
        )
        return self._resize_mask_back(masks[0])

    def _resize_mask_back(self, mask):
        """将分割结果 resize 回原图大小"""
        if self.scale_factor != 1.0:
            h, w = self.original_shape
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return mask.astype(bool)
        return mask

    def segment_with_points(self, image, input_points, input_labels):
        """使用点提示进行分割"""
        try:
            # 确保图像是RGB格式
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            # 转换为torch张量
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            input_points_tensor = torch.from_numpy(input_points).unsqueeze(0)
            input_labels_tensor = torch.from_numpy(input_labels).unsqueeze(0)
            
            # 移动到正确的设备
            device = next(self.sam.parameters()).device
            image_tensor = image_tensor.to(device)
            input_points_tensor = input_points_tensor.to(device)
            input_labels_tensor = input_labels_tensor.to(device)
            
            # 进行分割
            with torch.no_grad():
                image_embedding = self.sam.image_encoder(image_tensor.unsqueeze(0))
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=(input_points_tensor, input_labels_tensor),
                    boxes=None,
                    masks=None,
                )
                mask_predictions, _ = self.sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            
            # 处理预测结果
            mask_predictions = torch.sigmoid(mask_predictions)
            mask_predictions = mask_predictions > 0.5
            mask = mask_predictions[0, 0].cpu().numpy()
            
            return mask
            
        except Exception as e:
            print(f"[SAM] 分割失败: {str(e)}")
            raise
