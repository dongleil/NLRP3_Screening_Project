"""
数据预处理脚本
对ChEMBL下载的数据进行清洗、标准化和标注
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logger, load_data_config, log_section, log_dict,
    MoleculeProcessor, MoleculeValidator, get_inchi_key,
    calculate_descriptors
)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: dict):
        """
        初始化预处理器
        
        Args:
            config: 数据配置字典
        """
        self.config = config
        self.logger = setup_logger("Data_Preprocessor")
        self.mol_processor = MoleculeProcessor()
        self.mol_validator = MoleculeValidator(
            mw_range=tuple(config['filtering']['molecular_weight_range']),
            heavy_atom_range=tuple(config['filtering']['heavy_atom_count_range'])
        )
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据DataFrame
        """
        self.logger.info(f"加载原始数据: {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info(f"  加载了 {len(df)} 条记录")
        return df
    
    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一单位转换为 μM
        
        Args:
            df: 输入DataFrame
            
        Returns:
            转换后的DataFrame
        """
        self.logger.info("统一单位转换...")
        
        df = df.copy()
        df['value_um'] = np.nan
        
        # nM -> μM
        mask_nm = df['standard_units'] == 'nM'
        df.loc[mask_nm, 'value_um'] = df.loc[mask_nm, 'standard_value'] / 1000
        
        # μM -> μM
        mask_um = df['standard_units'].isin(['uM', 'μM'])
        df.loc[mask_um, 'value_um'] = df.loc[mask_um, 'standard_value']
        
        # 删除无法转换的数据
        df = df.dropna(subset=['value_um'])
        
        self.logger.info(f"  单位转换完成，剩余 {len(df)} 条记录")
        return df
    
    def assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据活性阈值分配标签
        
        Args:
            df: 输入DataFrame
            
        Returns:
            带标签的DataFrame
        """
        active_threshold = self.config['filtering']['active_threshold']
        inactive_threshold = self.config['filtering']['inactive_threshold']
        
        self.logger.info(f"分配标签...")
        self.logger.info(f"  活性阈值: IC50 < {active_threshold} μM")
        self.logger.info(f"  非活性阈值: IC50 > {inactive_threshold} μM")
        
        df = df.copy()
        df['label'] = -1  # 初始化为-1（未分类）
        
        # 活性化合物
        df.loc[df['value_um'] < active_threshold, 'label'] = 1
        
        # 非活性化合物
        df.loc[df['value_um'] > inactive_threshold, 'label'] = 0
        
        # 统计
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        n_uncertain = (df['label'] == -1).sum()
        
        self.logger.info(f"  活性化合物: {n_active}")
        self.logger.info(f"  非活性化合物: {n_inactive}")
        self.logger.info(f"  不确定区域: {n_uncertain} (将被移除)")
        
        # 移除不确定区域
        df = df[df['label'] != -1]
        
        return df
    
    def standardize_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化分子结构
        
        Args:
            df: 输入DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        self.logger.info("标准化分子结构...")
        
        results = []
        failed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="标准化分子"):
            smiles = row['canonical_smiles']
            
            # 处理SMILES
            canonical_smiles, mol = self.mol_processor.process_smiles(smiles)
            
            if canonical_smiles is None:
                failed_count += 1
                continue
            
            # 验证分子
            if not self.mol_validator.is_valid(mol):
                failed_count += 1
                continue
            
            # 获取InChI Key（用于去重）
            inchi_key = get_inchi_key(mol)
            if inchi_key is None:
                failed_count += 1
                continue
            
            # 计算描述符
            descriptors = calculate_descriptors(mol)
            
            # 保存结果
            result = row.to_dict()
            result['smiles_standardized'] = canonical_smiles
            result['inchi_key'] = inchi_key
            result.update(descriptors)
            
            results.append(result)
        
        df_clean = pd.DataFrame(results)
        
        self.logger.info(f"  标准化完成")
        self.logger.info(f"  成功: {len(df_clean)}")
        self.logger.info(f"  失败: {failed_count}")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据InChI Key去重
        
        Args:
            df: 输入DataFrame
            
        Returns:
            去重后的DataFrame
        """
        self.logger.info("去重...")
        
        initial_count = len(df)
        
        # 按InChI Key分组，取活性最强的（value_um最小）
        df = df.sort_values('value_um')
        df = df.drop_duplicates(subset=['inchi_key'], keep='first')
        
        final_count = len(df)
        removed = initial_count - final_count
        
        self.logger.info(f"  移除了 {removed} 个重复分子")
        self.logger.info(f"  剩余 {final_count} 个唯一分子")
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        平衡数据集
        
        Args:
            df: 输入DataFrame
            
        Returns:
            平衡后的DataFrame
        """
        method = self.config['balancing']['method']
        target_ratio = self.config['balancing']['target_ratio']
        
        self.logger.info(f"平衡数据集 (方法: {method}, 目标比例: {target_ratio})")
        
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        
        self.logger.info(f"  原始分布: 活性={n_active}, 非活性={n_inactive}")
        
        if method == "undersample":
            # 欠采样多数类
            target_inactive = int(n_active * target_ratio)
            
            if n_inactive > target_inactive:
                df_active = df[df['label'] == 1]
                df_inactive = df[df['label'] == 0].sample(
                    n=target_inactive, random_state=42
                )
                df = pd.concat([df_active, df_inactive])
                
                self.logger.info(f"  欠采样后: 活性={len(df_active)}, 非活性={len(df_inactive)}")
        
        elif method == "none":
            self.logger.info("  不进行平衡")
        
        return df.reset_index(drop=True)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        保存处理后的数据
        
        Args:
            df: 数据DataFrame
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 选择要保存的列
        columns_to_save = [
            'molecule_chembl_id',
            'smiles_standardized',
            'inchi_key',
            'standard_type',
            'value_um',
            'label',
            'MW', 'LogP', 'TPSA', 'HBA', 'HBD',
            'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
        ]
        
        df_save = df[columns_to_save].copy()
        df_save.to_csv(output_path, index=False)
        
        self.logger.info(f"数据已保存到: {output_path}")
    
    def save_statistics(self, df: pd.DataFrame, output_path: str):
        """
        保存数据统计信息
        
        Args:
            df: 数据DataFrame
            output_path: 输出路径
        """
        import json
        
        stats = {
            "total_compounds": len(df),
            "active_compounds": int((df['label'] == 1).sum()),
            "inactive_compounds": int((df['label'] == 0).sum()),
            "activity_distribution": df['label'].value_counts().to_dict(),
            "molecular_properties": {
                "MW": {
                    "mean": float(df['MW'].mean()),
                    "std": float(df['MW'].std()),
                    "min": float(df['MW'].min()),
                    "max": float(df['MW'].max())
                },
                "LogP": {
                    "mean": float(df['LogP'].mean()),
                    "std": float(df['LogP'].std()),
                    "min": float(df['LogP'].min()),
                    "max": float(df['LogP'].max())
                },
                "TPSA": {
                    "mean": float(df['TPSA'].mean()),
                    "std": float(df['TPSA'].std())
                }
            }
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"统计信息已保存到: {output_path}")
    
    def run(self, input_path: str) -> str:
        """
        运行完整的预处理流程
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            输出文件路径
        """
        log_section(self.logger, "数据预处理")
        
        # 1. 加载原始数据
        df = self.load_raw_data(input_path)
        
        # 2. 统一单位
        df = self.convert_units(df)
        
        # 3. 分配标签
        df = self.assign_labels(df)
        
        # 4. 标准化分子
        df = self.standardize_molecules(df)
        
        # 5. 去重
        df = self.remove_duplicates(df)
        
        # 6. 平衡数据集
        df = self.balance_dataset(df)
        
        # 7. 保存数据
        output_dir = self.config['paths']['processed_data_dir']
        output_file = self.config['filenames']['processed_data']
        output_path = Path(output_dir) / output_file
        
        self.save_processed_data(df, output_path)
        
        # 8. 保存统计信息
        stats_file = self.config['filenames']['data_statistics']
        stats_path = Path(output_dir) / stats_file
        self.save_statistics(df, stats_path)
        
        # 显示最终统计
        log_section(self.logger, "预处理完成")
        self.logger.info(f"最终数据集:")
        self.logger.info(f"  总化合物数: {len(df)}")
        self.logger.info(f"  活性化合物: {(df['label'] == 1).sum()}")
        self.logger.info(f"  非活性化合物: {(df['label'] == 0).sum()}")
        self.logger.info(f"  活性/非活性比例: {(df['label'] == 1).sum() / (df['label'] == 0).sum():.2f}")
        
        return str(output_path)


def main():
    """主函数"""
    # 加载配置
    config = load_data_config()
    
    # 构建输入路径
    input_dir = config['paths']['raw_data_dir']
    input_file = config['filenames']['raw_data']
    input_path = Path(input_dir) / input_file
    
    if not input_path.exists():
        print(f"错误: 找不到原始数据文件: {input_path}")
        print("请先运行: python experiments/stage0_data/01_download_chembl.py")
        return
    
    # 创建预处理器
    preprocessor = DataPreprocessor(config)
    
    # 运行预处理
    output_path = preprocessor.run(str(input_path))
    
    print(f"\n✓ 预处理完成")
    print(f"✓ 数据已保存到: {output_path}")
    print("\n下一步: 运行数据集划分")
    print("  python experiments/stage0_data/03_split_dataset.py")


if __name__ == "__main__":
    main()
