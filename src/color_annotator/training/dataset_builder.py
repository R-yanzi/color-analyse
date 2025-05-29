import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

class SegmentationDatasetBuilder:
    def __init__(self, annotations_dir, output_dir, train_ratio=0.8):
        """
        初始化数据集构建器
        
        Args:
            annotations_dir: 标注文件目录
            output_dir: 输出目录
            train_ratio: 训练集比例
        """
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        
        # 创建必要的目录
        self.train_img_dir = self.output_dir / "train" / "read_images"
        self.train_mask_dir = self.output_dir / "train" / "masks"
        self.val_img_dir = self.output_dir / "val" / "read_images"
        self.val_mask_dir = self.output_dir / "val" / "masks"
        
        for dir_path in [self.train_img_dir, self.train_mask_dir, 
                        self.val_img_dir, self.val_mask_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def decode_rle(self, rle, shape):
        """解码RLE数据为二值掩码"""
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, length in rle:
            mask[start:start + length] = 1
        return mask.reshape(shape)

    def process_single_annotation(self, json_path):
        """处理单个标注文件"""
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像路径
            img_path = Path(data['image_path'])
            if not img_path.is_absolute():
                # 尝试多个可能的图片位置
                possible_paths = [
                    Path(self.annotations_dir).parent / img_path,  # 相对于annotations目录
                    Path(self.annotations_dir).parent / 'read_images' / img_path.name,  # read_images目录
                    Path(self.annotations_dir).parent / 'images' / img_path.name,  # images目录
                ]
                
                # 检查每个可能的路径
                found = False
                for p in possible_paths:
                    if p.exists():
                        img_path = p
                        found = True
                        break
                
                if not found:
                    print(f"错误: 找不到图片文件 {img_path.name}")
                    print("尝试过以下路径:")
                    for p in possible_paths:
                        print(f"  - {p}")
                    return None
            
            print(f"处理图片: {img_path}")
            
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"错误: 无法读取图像: {img_path}")
                return None
            
            # 创建组合掩码
            height, width = image.shape[:2]
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 处理每个标注
            for annotation in data['annotations']:
                mask_shape = tuple(annotation['size'])
                rle = annotation['rle']
                
                try:
                    mask = self.decode_rle(rle, mask_shape)
                    
                    # 检查掩码尺寸是否与图像匹配
                    if mask.shape != (height, width):
                        print(f"警告: 掩码尺寸 {mask.shape} 与图像尺寸 {(height, width)} 不匹配")
                        print(f"正在调整掩码尺寸...")
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    mask = (mask > 0).astype(np.uint8)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
                except Exception as e:
                    print(f"处理掩码时出错: {e}")
                    continue
            
            if np.sum(combined_mask) == 0:
                print(f"警告: {json_path.name} 生成的掩码为空")
                return None
            
            print(f"成功处理: {json_path.name}")
            print(f"  - 图像路径: {img_path}")
            print(f"  - 图像尺寸: {width}x{height}")
            print(f"  - 掩码尺寸: {combined_mask.shape}")
            print(f"  - 掩码像素数: {np.sum(combined_mask)}")
            
            return {
                'image': image,
                'mask': combined_mask,
                'filename': img_path.name
            }
            
        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            print(f"错误详情:\n{traceback.format_exc()}")
            return None

    def build_dataset(self):
        """构建数据集"""
        print("\n=== 开始构建数据集 ===")
        print(f"工作目录: {os.getcwd()}")
        print(f"标注目录: {self.annotations_dir}")
        
        # 检查目录是否存在
        if not self.annotations_dir.exists():
            raise ValueError(f"标注目录不存在: {self.annotations_dir}")
            
        # 获取所有JSON文件
        json_files = list(self.annotations_dir.glob('*.json'))
        if not json_files:
            raise ValueError(f"在 {self.annotations_dir} 中未找到JSON文件")

        print(f"\n找到 {len(json_files)} 个标注文件:")
        for json_file in json_files:
            print(f"  - {json_file.name}")

        # 处理所有标注文件
        dataset = []
        for json_path in json_files:
            print(f"\n处理文件: {json_path.name}")
            result = self.process_single_annotation(json_path)
            if result:
                dataset.append(result)
                print(f"✓ 成功添加到数据集: {json_path.name}")
            else:
                print(f"✗ 跳过文件: {json_path.name}")

        if not dataset:
            print("\n错误: 没有成功处理任何标注文件")
            return

        # 划分训练集和验证集
        train_data, val_data = train_test_split(
            dataset, 
            train_size=self.train_ratio,
            random_state=42
        )

        # 保存数据集
        print("\n=== 保存数据集 ===")
        print("保存训练集...")
        self._save_dataset_split(train_data, self.train_img_dir, self.train_mask_dir)
        
        print("\n保存验证集...")
        self._save_dataset_split(val_data, self.val_img_dir, self.val_mask_dir)

        print("\n=== 数据集构建完成 ===")
        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        print(f"\n数据集保存在: {self.output_dir}")

    def _save_dataset_split(self, data_split, img_dir, mask_dir):
        """保存数据集划分"""
        for idx, item in enumerate(data_split):
            # 保存图像
            img_path = img_dir / item['filename']
            cv2.imwrite(str(img_path), item['image'])
            print(f"✓ 保存图像: {img_path.name}")
            
            # 保存掩码
            mask_path = mask_dir / item['filename'].replace(
                Path(item['filename']).suffix, '_mask.png'
            )
            cv2.imwrite(str(mask_path), item['mask'] * 255)
            print(f"✓ 保存掩码: {mask_path.name}")

def main():
    """主函数"""
    try:
        # 获取脚本所在目录的根目录路径
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent.parent
        
        # 设置路径
        annotations_dir = project_root / "annotations"
        output_dir = project_root / "segmentation_dataset"
        
        print("=== 配置信息 ===")
        print(f"项目根目录: {project_root}")
        print(f"标注文件目录: {annotations_dir}")
        print(f"输出目录: {output_dir}")
        
        # 构建数据集
        builder = SegmentationDatasetBuilder(annotations_dir, output_dir)
        builder.build_dataset()
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        
if __name__ == "__main__":
    main() 