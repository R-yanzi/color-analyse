import numpy as np
import cv2
from typing import List, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans

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
    """颜色分析工具，用于提取图像中的主要颜色"""
    
    def __init__(self):
        """初始化颜色分析器"""
        self.min_color_area = 0.003  # 降低最小颜色区域占比到0.3%
        
    def create_color_info(self, rgb, percentage):
        """
        从已知的RGB值和占比创建ColorInfo对象
        
        Args:
            rgb: RGB值元组 (r, g, b)
            percentage: 该颜色的占比
            
        Returns:
            ColorInfo对象
        """
        return ColorInfo(rgb=rgb, percentage=percentage)
    
    def analyze_image_colors(self, image, mask=None, k=3, min_count=100):
        """
        分析图像中的主要颜色
        
        Args:
            image: 输入图像，BGR格式
            mask: 可选的掩码，用于限制分析区域
            k: 要提取的颜色数量
            min_count: 颜色聚类的最小像素数
            
        Returns:
            ColorInfo对象列表，按占比降序排列
        """
        try:
            # 确保图像是BGR格式
            img = image.copy()
            
            # 使用双边滤波减少纹理影响，保留边缘
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # 增强对比度，使颜色更鲜明
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 转换为RGB格式进行处理
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 获取图像尺寸
            height, width, _ = img_rgb.shape
            total_pixels = height * width
            
            # 如果有掩码，只分析掩码区域
            if mask is not None:
                # 确保掩码与图像尺寸一致
                if mask.shape[:2] != (height, width):
                    print(f"掩码尺寸 {mask.shape[:2]} 与图像尺寸 {(height, width)} 不匹配")
                    return []
                
                # 应用掩码
                masked_img = img_rgb.copy()
                masked_img[~mask] = [0, 0, 0]
                img_rgb = masked_img
                
                # 更新总像素数为掩码内的像素数
                total_pixels = np.sum(mask)
                if total_pixels == 0:
                    print("掩码内无有效像素")
                    return []
            
            # 展平图像数组
            pixels = img_rgb.reshape(-1, 3).astype(np.float32)
            
            # 排除黑色像素（背景）
            non_black = np.any(pixels > 5, axis=1)
            pixels_filtered = pixels[non_black]
            
            # 如果没有足够的非黑色像素，返回空列表
            if len(pixels_filtered) < min_count:
                print(f"有效像素数 {len(pixels_filtered)} 小于最小要求 {min_count}")
                return []
            
            # 对RGB数据进行加权处理，使聚类更关注人眼敏感的颜色区域
            # 人眼对绿色最敏感，其次是红色，最后是蓝色
            # 使用权重比例大约为G:R:B = 6:3:1
            weighted_pixels = pixels_filtered.copy()
            weighted_pixels[:, 0] *= 1.2  # R
            weighted_pixels[:, 1] *= 1.8  # G
            weighted_pixels[:, 2] *= 0.6  # B
            
            # 使用K-means聚类提取主要颜色
            k = min(k, len(pixels_filtered) // min_count)
            if k < 1:
                k = 1
                
            # 使用改进的聚类算法提高准确性
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(weighted_pixels)  # 使用加权像素进行聚类
            
            # 获取聚类中心（颜色）但使用原始RGB值
            cluster_centers = kmeans.cluster_centers_.copy()
            # 反向调整权重以获得真实RGB值
            cluster_centers[:, 0] /= 1.2  # R
            cluster_centers[:, 1] /= 1.8  # G
            cluster_centers[:, 2] /= 0.6  # B
            # 确保RGB值在0-255范围内
            colors = np.clip(cluster_centers, 0, 255).astype(np.uint8)
            
            # 计算每个颜色的像素数量
            labels = kmeans.labels_
            counts = np.bincount(labels, minlength=k)
            
            # 创建颜色信息列表
            color_infos = []
            for i in range(k):
                # 计算占比
                percentage = counts[i] / total_pixels
                
                # 颜色RGB值
                rgb = tuple(colors[i])
                
                # 创建颜色信息对象
                color_info = ColorInfo(rgb=rgb, percentage=percentage)
                color_infos.append(color_info)
            
            # 按占比降序排序
            color_infos.sort(key=lambda x: x.percentage, reverse=True)
            
            # 过滤掉过暗或过亮的颜色（接近黑色或白色）
            filtered_colors = []
            for color_info in color_infos:
                r, g, b = color_info.rgb
                # 跳过接近黑色或白色的颜色
                if (r < 30 and g < 30 and b < 30) or (r > 230 and g > 230 and b > 230):
                    continue
                # 确保最明显的几种颜色被保留
                if len(filtered_colors) < 2 or color_info.percentage > 0.05:
                    filtered_colors.append(color_info)
            
            # 如果所有颜色都被过滤掉了，返回原始颜色列表
            if not filtered_colors and color_infos:
                return color_infos
            
            return filtered_colors
            
        except Exception as e:
            print(f"分析图像颜色时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
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