import argparse
import os
from pathlib import Path
import torch
import time
import datetime
import shutil
import sys

def main():
    parser = argparse.ArgumentParser(description="畲族服饰分割模型训练与评估工具")
    parser.add_argument("--mode", type=str, choices=['train', 'auto-train', 'evaluate', 'full'], 
                       default='train', help="运行模式: train(常规训练), auto-train(自动调参), evaluate(评估), full(全流程)")
    parser.add_argument("--train-dir", type=str, default="segmentation_dataset/train", help="训练数据目录")
    parser.add_argument("--val-dir", type=str, default="segmentation_dataset/val", help="验证数据目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--mixed-precision", action="store_true", help="使用混合精度训练")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--model-path", type=str, help="评估模式下的模型路径")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="评估结果输出目录")
    parser.add_argument("--trials", type=int, default=5, help="自动调参模式下的尝试次数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    
    args = parser.parse_args()
    
    # 创建时间戳目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/run_{timestamp}")
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存运行配置
    with open(run_dir / "config.txt", "w") as f:
        f.write(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"运行模式: {args.mode}\n")
        f.write(f"命令行参数: {' '.join(sys.argv)}\n\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # 根据模式执行不同操作
    if args.mode == 'train' or args.mode == 'full':
        print("\n=== 开始常规训练 ===")
        train_save_dir = run_dir / "train_results"
        train_save_dir.mkdir(exist_ok=True, parents=True)
        
        # 构建训练命令
        train_cmd = [
            "python", "-m", "src.color_annotator.improved_train",
            "--train-dir", args.train_dir,
            "--val-dir", args.val_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--device", args.device,
            "--save-dir", str(train_save_dir)
        ]
        
        if args.mixed_precision:
            train_cmd.append("--mixed-precision")
            
        if args.resume:
            train_cmd.extend(["--resume", args.resume])
        
        # 执行训练
        import subprocess
        print(f"执行命令: {' '.join(train_cmd)}")
        train_process = subprocess.run(train_cmd)
        
        if train_process.returncode != 0:
            print(f"训练失败，返回代码: {train_process.returncode}")
            return
        
        # 找到最佳模型路径
        best_model_path = train_save_dir / "best_model.pth"
        
    if args.mode == 'auto-train' or args.mode == 'full':
        print("\n=== 开始自动调参训练 ===")
        auto_save_dir = run_dir / "auto_train_results"
        auto_save_dir.mkdir(exist_ok=True, parents=True)
        
        # 构建自动训练命令
        auto_cmd = [
            "python", "-m", "src.color_annotator.auto_train",
            "--train-dir", args.train_dir,
            "--val-dir", args.val_dir,
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--trials", str(args.trials)
        ]
        
        # 执行自动训练
        import subprocess
        print(f"执行命令: {' '.join(auto_cmd)}")
        auto_process = subprocess.run(auto_cmd)
        
        if auto_process.returncode != 0:
            print(f"自动训练失败，返回代码: {auto_process.returncode}")
        else:
            # 复制最佳模型到运行目录
            if os.path.exists("best_auto_model.pth"):
                shutil.copy("best_auto_model.pth", auto_save_dir / "best_auto_model.pth")
                best_model_path = auto_save_dir / "best_auto_model.pth"
    
    if args.mode == 'evaluate' or args.mode == 'full':
        print("\n=== 开始模型评估 ===")
        eval_output_dir = run_dir / "evaluation_results"
        eval_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 确定要评估的模型路径
        if args.mode == 'evaluate':
            if not args.model_path:
                print("评估模式需要指定 --model-path 参数")
                return
            eval_model_path = args.model_path
        else:  # full模式
            eval_model_path = best_model_path
        
        # 构建评估命令
        eval_cmd = [
            "python", "-m", "src.color_annotator.evaluate_model",
            "--model-path", str(eval_model_path),
            "--data-dir", args.val_dir,
            "--output-dir", str(eval_output_dir),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        
        # 执行评估
        import subprocess
        print(f"执行命令: {' '.join(eval_cmd)}")
        eval_process = subprocess.run(eval_cmd)
        
        if eval_process.returncode != 0:
            print(f"评估失败，返回代码: {eval_process.returncode}")
    
    print(f"\n=== 运行完成 ===")
    print(f"所有结果已保存到: {run_dir}")

if __name__ == "__main__":
    main() 