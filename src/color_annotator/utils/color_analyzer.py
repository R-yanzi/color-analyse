import numpy as np
import cv2
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ColorInfo:
    rgb: Tuple[int, int, int]
    percentage: float
    
    def __init__(self, rgb, percentage):
        # 确保RGB值是标准整数元组，而不是np.int64
        self.rgb = tuple(int(c) for c in rgb)
        self.percentage = percentage
    
    def __str__(self):
        r, g, b = self.rgb
        return f"RGB({r},{g},{b}): {self.percentage:.1%}"
    
    def is_white_or_black(self, threshold: int = 240) -> bool:
        """判断是否为白色或黑色"""
        r, g, b = self.rgb
        return (r > threshold and g > threshold and b > threshold) or \
               (r < 30 and g < 30 and b < 30)
    
    def get_brightness(self) -> float:
        """获取颜色的亮度"""
        r, g, b = self.rgb
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

class ColorAnalyzer:
    def __init__(self):
        self.min_color_area = 0.003  # 降低最小颜色区域占比到0.3%
        
    def extract_main_colors(self, image: np.ndarray, mask: np.ndarray, k: int = 12) -> List[ColorInfo]:
        """
        从掩码区域提取主要颜色
        Args:
            image: BGR格式的图像
            mask: 二值掩码
            k: 期望提取的主色数量
        Returns:
            List[ColorInfo]: 主色列表，按占比降序排列
        """
        # 转换为RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取掩码区域的像素
        pixels = img_rgb[mask.astype(bool)]
        if pixels.size == 0:
            return []
            
        # 转换为浮点数进行聚类
        pixels_float = pixels.astype(np.float32)
        
        # K-means聚类参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)  # 增加迭代次数和精度
        
        # 执行K-means聚类
        _, labels, centers = cv2.kmeans(
            pixels_float, 
            k, 
            None, 
            criteria, 
            30,  # 增加尝试次数
            cv2.KMEANS_PP_CENTERS  # 使用K-means++初始化
        )
        
        # 计算每个聚类的像素占比
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / counts.sum()
        
        # 转换结果
        colors = []
        for i in range(len(centers)):
            # 只保留超过最小面积阈值的颜色
            if percentages[i] >= self.min_color_area:
                color = centers[i].astype(int)
                # 确保转换为标准整数元组
                rgb_tuple = (int(color[0]), int(color[1]), int(color[2]))
                colors.append(ColorInfo(
                    rgb=rgb_tuple,
                    percentage=percentages[i]
                ))
        
        # 过滤掉白色和黑色
        colors = [c for c in colors if not c.is_white_or_black()]
        
        if not colors:  # 如果过滤后没有颜色，返回原始颜色
            return sorted(colors, key=lambda x: x.percentage, reverse=True)
        
        # 重新计算百分比
        total_percentage = sum(c.percentage for c in colors)
        for c in colors:
            c.percentage = c.percentage / total_percentage
        
        # 按占比降序排序
        return sorted(colors, key=lambda x: x.percentage, reverse=True)
    
    def optimize_colors(self, colors: List[ColorInfo], 
                       merge_threshold: float = 25.0) -> List[ColorInfo]:
        """
        优化颜色列表，合并相似颜色
        Args:
            colors: 原始颜色列表
            merge_threshold: 颜色相似度阈值（欧氏距离）
        Returns:
            List[ColorInfo]: 优化后的颜色列表
        """
        if not colors:
            return []
            
        # 复制一份进行处理
        processed = colors.copy()
        i = 0
        
        while i < len(processed):
            j = i + 1
            while j < len(processed):
                # 计算两个颜色的欧氏距离
                color1 = np.array(processed[i].rgb)
                color2 = np.array(processed[j].rgb)
                dist = np.sqrt(np.sum((color1 - color2) ** 2))
                
                # 如果颜色相似，合并
                if dist < merge_threshold:
                    # 按占比加权平均
                    w1 = processed[i].percentage
                    w2 = processed[j].percentage
                    total_weight = w1 + w2
                    
                    merged_color = (
                        (color1 * w1 + color2 * w2) / total_weight
                    ).astype(int)
                    
                    # 更新第一个颜色，确保使用标准整数元组
                    rgb_tuple = (int(merged_color[0]), int(merged_color[1]), int(merged_color[2]))
                    processed[i] = ColorInfo(
                        rgb=rgb_tuple,
                        percentage=total_weight
                    )
                    
                    # 删除第二个颜色
                    processed.pop(j)
                else:
                    j += 1
            i += 1
            
        return processed
    
    def analyze_image_colors(self, image: np.ndarray, mask: np.ndarray, 
                           k: int = 8) -> List[ColorInfo]:  # 减少默认聚类数
        """
        分析图像颜色的完整流程
        """
        # 1. 提取初始主色
        main_colors = self.extract_main_colors(image, mask, k)
        
        # 2. 优化相似颜色，增大合并阈值
        optimized_colors = self.optimize_colors(main_colors, merge_threshold=60.0)  # 增大合并阈值
        
        # 3. 过滤掉占比过小的颜色
        filtered_colors = [c for c in optimized_colors if c.percentage > 0.05]  # 只保留占比超过5%的颜色
        
        # 4. 重新计算百分比
        if filtered_colors:
            total = sum(c.percentage for c in filtered_colors)
            for c in filtered_colors:
                c.percentage = c.percentage / total
        
        return filtered_colors 