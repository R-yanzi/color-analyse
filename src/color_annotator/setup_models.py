import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_models():
    """设置所需的模型文件"""
    try:
        # 获取当前文件所在目录
        current_dir = Path(__file__).resolve().parent
        
        # 创建checkpoints目录
        checkpoints_dir = current_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # SAM模型文件路径
        sam_checkpoint = checkpoints_dir / "sam_vit_b.pth"
        
        # 如果SAM模型不存在，下载它
        if not sam_checkpoint.exists():
            print("正在下载SAM模型...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            download_file(url, sam_checkpoint)
            print("SAM模型下载完成！")
        else:
            print("SAM模型已存在，跳过下载。")
        
        # 创建其他必要的目录
        (current_dir / "images").mkdir(exist_ok=True)
        (current_dir / "annotations").mkdir(exist_ok=True)
        (current_dir / "reference_images").mkdir(exist_ok=True)
        
        print("所有必要的目录和文件已准备就绪！")
        return True
        
    except Exception as e:
        print(f"设置模型时出错: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_models():
        print("初始化成功！")
        sys.exit(0)
    else:
        print("初始化失败！")
        sys.exit(1) 