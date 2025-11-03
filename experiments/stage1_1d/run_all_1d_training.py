#!/usr/bin/env python
"""
一键运行第一阶段（1D特征）完整训练流程
====================================
自动执行：
1. 数据集划分
2. 传统机器学习模型训练
3. 神经网络训练
4. 模型对比与可视化

用法：
    python experiments/stage1_1d/run_all_1d_training.py
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, log_section


class Stage1DPipeline:
    """第一阶段完整训练流程"""
    
    def __init__(self):
        self.logger = setup_logger("Stage1D_Pipeline")
        self.experiments_dir = Path(__file__).parent
        self.start_time = None
    
    def run_script(self, script_name, description):
        """运行单个脚本"""
        log_section(self.logger, description)
        
        script_path = self.experiments_dir / script_name
        
        if not script_path.exists():
            self.logger.error(f"脚本不存在: {script_path}")
            return False
        
        self.logger.info(f"运行: {script_name}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )
            
            # 打印输出
            if result.stdout:
                print(result.stdout)
            
            self.logger.info(f"✓ {description} 完成")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ {description} 失败")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            return False
    
    def run(self):
        """运行完整流程"""
        self.start_time = time.time()
        
        print("="*70)
        print("NLRP3 筛选 - 第一阶段（1D特征）完整训练流程")
        print("="*70)
        print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 流程步骤
        steps = [
            ("01_split_dataset.py", "步骤1: 数据集划分"),
            ("02_train_traditional_ml.py", "步骤2: 传统机器学习训练"),
            ("03_train_neural_network.py", "步骤3: 神经网络训练"),
            ("04_compare_all_models.py", "步骤4: 模型对比与可视化"),
        ]
        
        # 执行所有步骤
        success_count = 0
        for script, description in steps:
            if self.run_script(script, description):
                success_count += 1
                print()
            else:
                self.logger.error(f"流程在 {description} 处中断")
                break
        
        # 总结
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print()
        print("="*70)
        if success_count == len(steps):
            print("[SUCCESS] 完整训练流程成功完成！")
        else:
            print(f"[WARNING] 流程部分完成 ({success_count}/{len(steps)})")
        print("="*70)
        print(f"\n总耗时: {minutes}分{seconds}秒")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success_count == len(steps):
            print(f"\n[OUTPUT] 查看结果:")
            print(f"  - 对比报告: results/stage1_1d/comparison/SUMMARY_REPORT.md")
            print(f"  - 可视化: results/stage1_1d/comparison/*.png")
            print(f"  - 模型文件: results/stage1_1d/")
            
            print(f"\n[NEXT] 下一步:")
            print(f"  - 查看模型性能报告")
            print(f"  - 选择最佳模型进行虚拟筛选")
            print(f"  - 或继续训练2D/3D模型")


def main():
    """主函数"""
    pipeline = Stage1DPipeline()
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
