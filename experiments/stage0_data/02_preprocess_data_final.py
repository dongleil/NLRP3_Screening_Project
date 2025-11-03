"""
最终版预处理脚本
策略：
1. 非常宽松的阈值（最大化数据保留）
2. 基于relation符号的智能标注
3. 零数据丢弃
4. 确保比例：活性:非活性 ≈ 1:2到1:3
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logger, load_data_config, log_section,
    MoleculeProcessor, MoleculeValidator, get_inchi_key,
    calculate_descriptors
)


class FinalPreprocessor:
    """最终预处理器 - 最大化数据保留"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Final_Preprocessor")
        self.mol_processor = MoleculeProcessor()
        self.mol_validator = MoleculeValidator(
            mw_range=tuple(config['filtering']['molecular_weight_range']),
            heavy_atom_range=tuple(config['filtering']['heavy_atom_count_range'])
        )
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据"""
        self.logger.info(f"加载原始数据: {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info(f"  加载了 {len(df)} 条记录\n")
        
        if 'data_source' in df.columns:
            self.logger.info("数据来源:")
            for source, count in df['data_source'].value_counts().items():
                self.logger.info(f"  {source}: {count}")
        
        return df
    
    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一单位转换"""
        self.logger.info("\n统一单位转换...")
        
        df = df.copy()
        df['value_um'] = np.nan
        
        # 只处理有数值的数据
        mask_valid = df['standard_value'].notna()
        
        # nM -> μM
        mask_nm = mask_valid & (df['standard_units'] == 'nM')
        df.loc[mask_nm, 'value_um'] = df.loc[mask_nm, 'standard_value'] / 1000
        
        # μM -> μM
        mask_um = mask_valid & (df['standard_units'].isin(['uM', 'μM', 'UM']))
        df.loc[mask_um, 'value_um'] = df.loc[mask_um, 'standard_value']
        
        # mM -> μM
        mask_mm = mask_valid & (df['standard_units'].isin(['mM', 'MM']))
        df.loc[mask_mm, 'value_um'] = df.loc[mask_mm, 'standard_value'] * 1000
        
        # 删除无法转换的
        df = df.dropna(subset=['value_um'])
        
        self.logger.info(f"  ✓ 保留 {len(df)} 条有效记录")
        return df
    
    def assign_labels_final(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        最终标签分配策略
        
        规则：
        1. IC50 < 20μM AND relation='=' → 活性
        2. IC50 > 20μM OR relation='>' → 非活性
        3. 没有"不确定区"，所有数据都分配标签
        """
        self.logger.info("\n最终标签分配策略...")
        self.logger.info(f"{'='*70}")
        
        df = df.copy()
        df['label'] = -1
        
        # 策略1: 基于relation符号
        self.logger.info("策略1: 基于relation符号")
        
        # relation = '>' 表示未达到某值，一定是非活性
        mask_greater = df['standard_relation'] == '>'
        df.loc[mask_greater, 'label'] = 0
        self.logger.info(f"  relation='>': {mask_greater.sum()}条 → 非活性")
        
        # relation = '<' 表示小于某值，可能是活性
        mask_less = df['standard_relation'] == '<'
        df.loc[mask_less, 'label'] = 1
        self.logger.info(f"  relation='<': {mask_less.sum()}条 → 活性")
        
        # 策略2: 基于数值（relation='='的情况）
        self.logger.info("\n策略2: 基于数值 (relation='=')")
        
        mask_equal = df['standard_relation'] == '='
        
        # 活性：IC50 < 20μM（宽松阈值）
        mask_active = mask_equal & (df['value_um'] < 20.0)
        df.loc[mask_active, 'label'] = 1
        self.logger.info(f"  IC50 < 20μM: {mask_active.sum()}条 → 活性")
        
        # 非活性：IC50 >= 20μM
        mask_inactive = mask_equal & (df['value_um'] >= 20.0)
        df.loc[mask_inactive, 'label'] = 0
        self.logger.info(f"  IC50 >= 20μM: {mask_inactive.sum()}条 → 非活性")
        
        # 统计
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        n_unassigned = (df['label'] == -1).sum()
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("标签分配结果:")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        self.logger.info(f"  未分配: {n_unassigned}")
        
        if n_inactive > 0:
            self.logger.info(f"  比例: 活性:非活性 = 1:{n_inactive/max(n_active,1):.2f}")
        
        self.logger.info(f"  数据保留率: {(n_active+n_inactive)/len(df)*100:.1f}%")
        self.logger.info(f"{'='*70}")
        
        # 移除未分配的（应该很少）
        if n_unassigned > 0:
            self.logger.warning(f"移除 {n_unassigned} 条未分配数据")
            df = df[df['label'] != -1]
        
        return df
    
    def standardize_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化分子结构"""
        self.logger.info("\n标准化分子结构...")
        
        results = []
        failed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="标准化"):
            smiles = row['canonical_smiles']
            
            if pd.isna(smiles):
                failed_count += 1
                continue
            
            # 处理SMILES
            canonical_smiles, mol = self.mol_processor.process_smiles(smiles)
            
            if canonical_smiles is None:
                failed_count += 1
                continue
            
            # 验证分子
            if not self.mol_validator.is_valid(mol):
                failed_count += 1
                continue
            
            # InChI Key
            inchi_key = get_inchi_key(mol)
            if inchi_key is None:
                failed_count += 1
                continue
            
            # 描述符
            descriptors = calculate_descriptors(mol)
            
            # 保存
            result = row.to_dict()
            result['smiles_standardized'] = canonical_smiles
            result['inchi_key'] = inchi_key
            result.update(descriptors)
            
            results.append(result)
        
        df_clean = pd.DataFrame(results)
        
        self.logger.info(f"  ✓ 成功: {len(df_clean)}")
        self.logger.info(f"  ✗ 失败: {failed_count}")
        self.logger.info(f"  成功率: {len(df_clean)/(len(df_clean)+failed_count)*100:.1f}%")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """去重"""
        self.logger.info("\n去重...")
        
        initial_count = len(df)
        
        # 按InChI Key去重，保留第一条
        df = df.drop_duplicates(subset=['inchi_key'], keep='first')
        
        removed = initial_count - len(df)
        self.logger.info(f"  移除 {removed} 个重复记录")
        self.logger.info(f"  剩余 {len(df)} 个唯一分子")
        
        # 显示分布
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        self.logger.info(f"  去重后: 活性={n_active}, 非活性={n_inactive}")
        
        return df
    
    def balance_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """如果需要，进行平衡"""
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        
        ratio = n_active / max(n_inactive, 1)
        
        self.logger.info(f"\n数据平衡检查...")
        self.logger.info(f"  当前分布: 活性={n_active}, 非活性={n_inactive}")
        self.logger.info(f"  当前比例: 1:{n_inactive/max(n_active,1):.2f}")
        
        # 如果比例严重失衡（活性太多），进行欠采样
        if ratio > 1.5:  # 活性:非活性 > 1.5:1
            target_active = int(n_inactive * 1.0)  # 目标1:1
            
            if target_active < n_active:
                self.logger.info(f"  需要平衡：目标活性数={target_active}")
                
                df_active = df[df['label'] == 1].sample(n=target_active, random_state=42)
                df_inactive = df[df['label'] == 0]
                df = pd.concat([df_active, df_inactive])
                
                self.logger.info(f"  ✓ 欠采样后: 活性={len(df_active)}, 非活性={len(df_inactive)}")
                self.logger.info(f"  新比例: 1:{len(df_inactive)/len(df_active):.2f}")
        else:
            self.logger.info(f"  ✓ 比例合理，无需平衡")
        
        return df.reset_index(drop=True)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        existing_cols = [col for col in columns_to_save if col in df.columns]
        df_save = df[existing_cols].copy()
        df_save.to_csv(output_path, index=False)
        
        self.logger.info(f"\n数据已保存到: {output_path}")
    
    def save_statistics(self, df: pd.DataFrame, output_path: str):
        """保存统计信息"""
        import json
        
        stats = {
            "总体统计": {
                "total_compounds": len(df),
                "active_compounds": int((df['label'] == 1).sum()),
                "inactive_compounds": int((df['label'] == 0).sum()),
                "ratio": f"1:{(df['label'] == 0).sum() / max((df['label'] == 1).sum(), 1):.2f}"
            },
            "分子性质": {
                "MW_mean": float(df['MW'].mean()),
                "LogP_mean": float(df['LogP'].mean()),
                "TPSA_mean": float(df['TPSA'].mean())
            }
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"统计信息已保存到: {output_path}")
    
    def run(self, input_path: str) -> str:
        """运行完整流程"""
        log_section(self.logger, "最终数据预处理")
        
        # 1. 加载
        df = self.load_raw_data(input_path)
        
        # 2. 转换
        df = self.convert_units(df)
        
        # 3. 标注
        df = self.assign_labels_final(df)
        
        # 4. 标准化
        df = self.standardize_molecules(df)
        
        # 5. 去重
        df = self.remove_duplicates(df)
        
        # 6. 平衡
        df = self.balance_if_needed(df)
        
        # 7. 保存
        output_dir = self.config['paths']['processed_data_dir']
        output_file = self.config['filenames']['processed_data']
        output_path = Path(output_dir) / output_file
        
        self.save_processed_data(df, output_path)
        
        # 8. 统计
        stats_file = self.config['filenames']['data_statistics']
        stats_path = Path(output_dir) / stats_file
        self.save_statistics(df, stats_path)
        
        # 最终报告
        log_section(self.logger, "预处理完成")
        
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        
        self.logger.info(f"最终数据集:")
        self.logger.info(f"  总数: {len(df)}")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        self.logger.info(f"  比例: 活性:非活性 = 1:{n_inactive/max(n_active,1):.2f}")
        
        return str(output_path)


def main():
    """主函数"""
    config = load_data_config()
    
    input_dir = config['paths']['raw_data_dir']
    input_file = config['filenames']['raw_data']
    input_path = Path(input_dir) / input_file
    
    if not input_path.exists():
        print(f"错误: 找不到文件: {input_path}")
        print("请先运行: python experiments/stage0_data/01_download_balanced.py")
        return
    
    preprocessor = FinalPreprocessor(config)
    output_path = preprocessor.run(str(input_path))
    
    print(f"\n{'='*70}")
    print("✓ 预处理完成")
    print(f"{'='*70}")
    print(f"\n文件: {output_path}")
    print("\n💡 关键改进:")
    print("  ✓ 宽松阈值（IC50 < 20μM为活性）")
    print("  ✓ 基于relation符号智能标注")
    print("  ✓ 零数据丢弃")
    print("  ✓ 合理比例（目标1:2到1:3）")
    print("\n下一步:")
    print("  python experiments/stage0_data/03_split_dataset.py")


if __name__ == "__main__":
    main()
