import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from .gui.main_window import MainWindow
import traceback

def run_automated_test():
    try:
        # 创建QApplication实例
        app = QApplication(sys.argv)
        
        # 创建主窗口
        print("[测试] 创建主窗口...")
        window = MainWindow()
        window.show()
        
        # 获取测试图像路径
        current_dir = Path(__file__).resolve().parent
        test_image = current_dir / "images" / "test.jpg"
        
        print(f"[测试] 图像路径: {test_image}")
        if not test_image.exists():
            print(f"[错误] 测试图像不存在: {test_image}")
            return 1
        
        def execute_test():
            try:
                print("[测试] 开始自动化测试...")
                
                # 1. 加载图像
                print("[测试] 加载图像...")
                img = cv2.imread(str(test_image))
                if img is None:
                    print("[错误] 无法加载图像")
                    return
                
                print(f"[测试] 图像尺寸: {img.shape}")
                window.viewer.set_image(img)
                window.viewer.image_path = str(test_image)
                
                # 2. 执行分割
                print("[测试] 执行分割...")
                window.run_segmentation()
                
                # 3. 设置一个定时器来检查结果
                def check_results():
                    try:
                        if hasattr(window.viewer, 'masks') and window.viewer.masks:
                            print(f"[测试] 检测到 {len(window.viewer.masks)} 个掩码")
                            for mask_id, mask_data in window.viewer.masks.items():
                                print(f"[测试] 掩码 {mask_id}:")
                                print(f"  - 颜色: {mask_data.get('color', 'unknown')}")
                                if 'mask' in mask_data:
                                    mask = mask_data['mask']
                                    print(f"  - 掩码尺寸: {mask.shape}")
                                    print(f"  - 掩码像素数: {np.sum(mask)}")
                        else:
                            print("[测试] 未检测到掩码")
                    except Exception as e:
                        print(f"[错误] 检查结果时出错: {str(e)}")
                        print(traceback.format_exc())
                
                # 3秒后检查结果
                QTimer.singleShot(3000, check_results)
                
            except Exception as e:
                print(f"[错误] 测试过程中出错: {str(e)}")
                print(traceback.format_exc())
        
        # 使用QTimer延迟执行测试
        QTimer.singleShot(1000, execute_test)
        
        # 运行应用
        return app.exec_()
        
    except Exception as e:
        print(f"[错误] 初始化测试时出错: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(run_automated_test()) 