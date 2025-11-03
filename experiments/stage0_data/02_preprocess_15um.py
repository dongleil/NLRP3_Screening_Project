"""
NLRP3数据预处理脚本 - 调整版（IC50<15μM）
=====================================
为了达到900个活性样本，将阈值从10μM调整到15μM
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 导入原始预处理器
from experiments.stage0_data.stage0_data.02_preprocess_strict import StrictPreprocessor
from src.utils import load_data_config


def main():
    """主函数 - 使用15μM阈值"""
    config = load_data_config()
    
    input_dir = config['paths']['raw_data_dir']
    input_file = config['filenames']['raw_data']
    input_path = Path(input_dir) / input_file
    
    if not input_path.exists():
        print(f"❌ 错误: 找不到文件 {input_path}")
        return
    
    # 创建预处理器并调整阈值
    preprocessor = StrictPreprocessor(config)
    
    # 🔧 调整阈值：10μM → 15μM
    print(f"\n{'='*70}")
    print("⚙️  阈值调整")
    print(f"{'='*70}")
    print("原阈值: IC50/EC50 < 10μM")
    print("新阈值: IC50/EC50 < 15μM")
    print(f"{'='*70}\n")
    
    preprocessor.THRESHOLDS = {
        'IC50': {'active': 15.0, 'inactive': 50.0},  # 10 → 15
        'EC50': {'active': 15.0, 'inactive': 50.0},  # 10 → 15
        'Ki': {'active': 15.0, 'inactive': 50.0},    # 10 → 15
        'Kd': {'active': 15.0, 'inactive': 50.0},    # 10 → 15
    }
    
    preprocessor.logger.info("✓ 阈值已调整为15μM")
    
    try:
        output_path = preprocessor.run(str(input_path))
        
        print(f"\n{'='*70}")
        print("✅ 预处理完成")
        print(f"{'='*70}")
        print(f"\n📁 输出: {output_path}")
        print(f"\n🎯 调整后目标:")
        print(f"  活性: 900 (IC50<15μM 或 EC50<15μM)")
        print(f"  非活性: 2700 (IC50>50μM)")
        print(f"  比例: 1:3")
        print(f"\n💡 如果还不够，可以继续调整到20μM")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
