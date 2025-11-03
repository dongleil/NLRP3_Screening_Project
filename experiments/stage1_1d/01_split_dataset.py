"""
数据集划分脚本
==============
将处理好的数据划分为训练集、验证集、测试集
- 训练集: 70%
- 验证集: 15%
- 测试集: 15%
- 分层采样保证比例一致
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, load_data_config, log_section


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
):
    """
    分层划分数据集
    
    Args:
        df: 输入数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        
    Returns:
        train_df, val_df, test_df
    """
    # 首先划分出测试集
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df['label']
    )
    
    # 再从训练+验证集中划分出验证集
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df['label']
    )
    
    return train_df, val_df, test_df


def main():
    """主函数"""
    logger = setup_logger("Dataset_Splitter")
    config = load_data_config()
    
    log_section(logger, "数据集划分")
    
    # 加载处理后的数据
    input_dir = config['paths']['processed_data_dir']
    input_file = config['filenames']['processed_data']
    input_path = Path(input_dir) / input_file
    
    if not input_path.exists():
        logger.error(f"找不到文件: {input_path}")
        logger.error("请先运行: python experiments/stage0_data/02_preprocess_strict.py")
        return
    
    logger.info(f"加载数据: {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"总样本数: {len(df)}")
    logger.info(f"  活性: {(df['label']==1).sum()}")
    logger.info(f"  非活性: {(df['label']==0).sum()}")
    
    # 划分数据集
    logger.info("\n划分比例: 训练70% / 验证15% / 测试15%")
    
    train_df, val_df, test_df = split_dataset(df)
    
    # 统计信息
    logger.info(f"\n{'='*70}")
    logger.info("划分结果")
    logger.info(f"{'='*70}")
    
    for name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        n_active = (split_df['label'] == 1).sum()
        n_inactive = (split_df['label'] == 0).sum()
        ratio = n_inactive / max(n_active, 1)
        
        logger.info(f"\n{name}:")
        logger.info(f"  总数: {len(split_df)}")
        logger.info(f"  活性: {n_active} ({n_active/len(split_df)*100:.1f}%)")
        logger.info(f"  非活性: {n_inactive} ({n_inactive/len(split_df)*100:.1f}%)")
        logger.info(f"  比例: 1:{ratio:.2f}")
    
    # 保存
    output_dir = Path(config['paths']['processed_data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"\n{'='*70}")
    logger.info("保存完成")
    logger.info(f"{'='*70}")
    logger.info(f"训练集: {train_path}")
    logger.info(f"验证集: {val_path}")
    logger.info(f"测试集: {test_path}")
    
    # 保存划分信息
    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_active': int((train_df['label']==1).sum()),
        'val_active': int((val_df['label']==1).sum()),
        'test_active': int((test_df['label']==1).sum()),
        'random_state': 42
    }
    
    import json
    info_path = output_dir / 'split_info.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"划分信息: {info_path}")
    
    log_section(logger, "划分完成")
    
    print(f"\n{'='*70}")
    print("[OK] 数据集划分完成")
    print(f"{'='*70}")
    print(f"\n下一步:")
    print(f"  python experiments/stage1_1d/02_train_traditional_ml.py")


if __name__ == "__main__":
    main()
