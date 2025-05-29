import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from .utils.auto_tester import AutoTester
from .utils.param_optimizer import ParamOptimizer
from .gui.ai_segmentation import AISegmentationWidget
from .gui.main_window import MainWindow

def run_auto_test(test_data_dir=None, optimize=True, n_trials=10):
    """运行自动化测试"""
    try:
        # 初始化QApplication
        app = QApplication(sys.argv)
        
        # 创建主窗口（但不显示）
        window = MainWindow()
        
        # 创建分割器
        segmentor = AISegmentationWidget()
        segmentor.setViewer(window.viewer)
        
        # 创建测试器
        tester = AutoTester()
        
        # 准备测试数据
        if test_data_dir:
            print(f"[准备] 从 {test_data_dir} 加载测试数据")
            tester.prepare_test_data(test_data_dir)
        
        if optimize:
            # 创建参数优化器
            print("\n[优化] 开始参数优化...")
            optimizer = ParamOptimizer()
            best_params, best_score = optimizer.optimize(tester, segmentor, n_trials)
            
            # 绘制优化进度
            optimizer.plot_optimization_progress()
            
            print("\n=== 优化结果 ===")
            print(f"最佳得分: {best_score:.3f}")
            print("\n最佳参数:")
            for param, value in best_params.items():
                print(f"{param}: {value:.3f}")
        else:
            # 直接运行测试
            print("\n[开始] 执行自动化测试...")
            results = tester.run_test(segmentor)
            
            # 分析结果
            print("\n[分析] 分析测试结果...")
            tester.analyze_results()
        
        return 0
        
    except Exception as e:
        print(f"[错误] 自动化测试失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自动化测试和优化工具")
    parser.add_argument("--test-data", type=str, help="测试数据目录路径")
    parser.add_argument("--no-optimize", action="store_true", help="禁用参数优化")
    parser.add_argument("--trials", type=int, default=10, help="优化试验次数")
    
    args = parser.parse_args()
    
    # 获取测试数据目录
    test_data_dir = args.test_data
    if not test_data_dir:
        current_dir = Path(__file__).resolve().parent
        test_data_dir = current_dir / "test_data"
    
    # 运行测试
    return run_auto_test(
        test_data_dir=test_data_dir,
        optimize=not args.no_optimize,
        n_trials=args.trials
    )

if __name__ == "__main__":
    sys.exit(main()) 