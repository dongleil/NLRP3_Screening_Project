"""
改进的数据预处理脚本
- 更合理的活性阈值
- 更少的数据流失
- 更平衡的数据集
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


class ImprovedDataPreprocessor:
    """改进的数据预处理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Improved_Preprocessor")
        self.mol_processor = MoleculeProcessor()
        self.mol_validator = MoleculeValidator(
            mw_range=tuple(config['filtering']['molecular_weight_range']),
            heavy_atom_range=tuple(config['filtering']['heavy_atom_count_range'])
        )
        
        # 改进的阈值设置
        self.thresholds = self._get_improved_thresholds()
    
    def _get_improved_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        获取改进的活性阈值
        
        策略：
        1. 降低活性阈值，提高非活性阈值
        2. 缩小不确定区域
        3. 为不同活性类型设置不同阈值
        """
        return {
            'IC50': {
                'active': 5.0,      # <5μM 为活性 (原10μM)
                'inactive': 30.0,   # >30μM 为非活性 (原50μM)
                'weight': 1.0
            },
            'EC50': {
                'active': 5.0,
                'inactive': 30.0,
                'weight': 1.0
            },
            'Ki': {
                'active': 1.0,      # Ki通常更严格
                'inactive': 20.0,
                'weight': 1.2
            },
            'Kd': {
                'active': 1.0,
                'inactive': 20.0,
                'weight': 1.2
            },
            'AC50': {
                'active': 5.0,
                'inactive': 30.0,
                'weight': 1.0
            }
        }
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据"""
        self.logger.info(f"加载原始数据: {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info(f"  加载了 {len(df)} 条记录")
        
        # 显示数据来源
        if 'data_source' in df.columns:
            self.logger.info(f"\n  数据来源分布:")
            for source, count in df['data_source'].value_counts().items():
                self.logger.info(f"    {source}: {count}")
        
        return df
    
    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一单位转换为 μM"""
        self.logger.info("\n统一单位转换...")
        
        df = df.copy()
        df['value_um'] = np.nan
        
        # 只处理有数值且relation为'='的数据
        mask_valid = (
            df['standard_value'].notna() & 
            (df['standard_relation'] == '=')
        )
        
        # nM -> μM
        mask_nm = mask_valid & (df['standard_units'] == 'nM')
        df.loc[mask_nm, 'value_um'] = df.loc[mask_nm, 'standard_value'] / 1000
        
        # μM -> μM
        mask_um = mask_valid & (df['standard_units'].isin(['uM', 'μM', 'UM']))
        df.loc[mask_um, 'value_um'] = df.loc[mask_um, 'standard_value']
        
        # mM -> μM
        mask_mm = mask_valid & (df['standard_units'].isin(['mM', 'MM']))
        df.loc[mask_mm, 'value_um'] = df.loc[mask_mm, 'standard_value'] * 1000
        
        # 删除无法转换的数据
        df = df.dropna(subset=['value_um'])
        
        converted = len(df)
        self.logger.info(f"  ✓ 单位转换完成")
        self.logger.info(f"  保留 {converted} 条有效记录")
        
        return df
    
    def assign_labels_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        改进的标签分配策略
        
        改进点：
        1. 更合理的阈值
        2. 对中间区域使用加权策略
        3. 保留更多数据
        """
        self.logger.info("\n改进的标签分配...")
        self.logger.info(f"{'='*70}")
        
        df = df.copy()
        df['label'] = -1
        df['label_confidence'] = 0.0  # 标签置信度
        
        # 支持的活性类型
        supported_types = list(self.thresholds.keys())
        self.logger.info(f"支持的活性类型: {supported_types}")
        
        # 显示阈值
        self.logger.info(f"\n活性判断标准（改进版）:")
        for atype, thres in self.thresholds.items():
            self.logger.info(
                f"  {atype}: 活性<{thres['active']}μM, "
                f"非活性>{thres['inactive']}μM"
            )
        
        # 统计每种类型
        stats = {}
        for atype in supported_types:
            mask_type = df['standard_type'] == atype
            data_type = df[mask_type]
            
            if len(data_type) == 0:
                continue
            
            thres = self.thresholds[atype]
            
            # 活性化合物
            mask_active = mask_type & (df['value_um'] < thres['active'])
            df.loc[mask_active, 'label'] = 1
            df.loc[mask_active, 'label_confidence'] = 1.0
            
            # 非活性化合物
            mask_inactive = mask_type & (df['value_um'] > thres['inactive'])
            df.loc[mask_inactive, 'label'] = 0
            df.loc[mask_inactive, 'label_confidence'] = 1.0
            
            # 中间区域（不确定）
            mask_uncertain = mask_type & (
                (df['value_um'] >= thres['active']) & 
                (df['value_um'] <= thres['inactive'])
            )
            
            # 对中间区域使用渐变标签（接近active阈值的标为活性）
            for idx in df[mask_uncertain].index:
                value = df.loc[idx, 'value_um']
                
                # 线性插值计算置信度
                ratio = (value - thres['active']) / (thres['inactive'] - thres['active'])
                
                if ratio < 0.5:  # 更接近活性阈值
                    df.loc[idx, 'label'] = 1
                    df.loc[idx, 'label_confidence'] = 1 - ratio * 2  # 0.5-1.0
                else:  # 更接近非活性阈值
                    df.loc[idx, 'label'] = 0
                    df.loc[idx, 'label_confidence'] = (ratio - 0.5) * 2  # 0.0-0.5
            
            # 统计
            n_active = (df[mask_type]['label'] == 1).sum()
            n_inactive = (df[mask_type]['label'] == 0).sum()
            n_uncertain = mask_uncertain.sum()
            
            stats[atype] = {
                'total': len(data_type),
                'active': n_active,
                'inactive': n_inactive,
                'uncertain_assigned': n_uncertain
            }
        
        # 显示统计
        self.logger.info(f"\n标签分配结果:")
        for atype, stat in stats.items():
            self.logger.info(
                f"  {atype}: 总数={stat['total']}, "
                f"活性={stat['active']}, "
                f"非活性={stat['inactive']}, "
                f"中间区(已分配)={stat['uncertain_assigned']}"
            )
        
        # 总体统计
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        n_unassigned = (df['label'] == -1).sum()
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"总体统计:")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        self.logger.info(f"  未分配: {n_unassigned}")
        self.logger.info(f"  活性/非活性比例: {n_active/max(n_inactive,1):.2f}:1")
        self.logger.info(f"  数据保留率: {(n_active+n_inactive)/len(df)*100:.1f}%")
        self.logger.info(f"{'='*70}")
        
        # 移除未分配的（应该很少或没有）
        if n_unassigned > 0:
            self.logger.info(f"\n移除 {n_unassigned} 条未分配数据")
            df = df[df['label'] != -1]
        
        return df
    
    def standardize_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化分子结构"""
        self.logger.info("\n标准化分子结构...")
        
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
            
            # 获取InChI Key
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
        
        self.logger.info(f"  ✓ 标准化完成")
        self.logger.info(f"  成功: {len(df_clean)}")
        self.logger.info(f"  失败: {failed_count}")
        self.logger.info(f"  成功率: {len(df_clean)/(len(df_clean)+failed_count)*100:.1f}%")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """去重 - 改进策略"""
        self.logger.info("\n智能去重...")
        
        initial_count = len(df)
        
        # 按InChI Key分组
        duplicates = df.groupby('inchi_key').size()
        n_duplicates = (duplicates > 1).sum()
        
        self.logger.info(f"  发现 {n_duplicates} 个分子有重复记录")
        
        # 对于重复的分子，选择最可靠的数据
        def select_best_record(group):
            """选择最佳记录"""
            # 优先级：
            # 1. 置信度最高
            # 2. IC50 > Ki > EC50
            # 3. 值最小（对于活性化合物）
            
            # 按优先级排序
            priority_map = {'IC50': 3, 'Ki': 2, 'EC50': 1, 'Kd': 2, 'AC50': 1}
            group['priority'] = group['standard_type'].map(lambda x: priority_map.get(x, 0))
            
            group = group.sort_values(
                by=['label_confidence', 'priority', 'value_um'],
                ascending=[False, False, True]
            )
            
            return group.iloc[0]
        
        df_dedup = df.groupby('inchi_key', group_keys=False).apply(select_best_record)
        df_dedup = df_dedup.reset_index(drop=True)
        
        removed = initial_count - len(df_dedup)
        self.logger.info(f"  移除了 {removed} 个重复记录")
        self.logger.info(f"  剩余 {len(df_dedup)} 个唯一分子")
        
        # 显示去重后的分布
        n_active = (df_dedup['label'] == 1).sum()
        n_inactive = (df_dedup['label'] == 0).sum()
        self.logger.info(f"  去重后: 活性={n_active}, 非活性={n_inactive}")
        
        return df_dedup
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据平衡 - 改进策略"""
        method = self.config['balancing']['method']
        target_ratio = self.config['balancing']['target_ratio']
        
        self.logger.info(f"\n数据平衡 (方法: {method}, 目标比例: {target_ratio})")
        
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        
        self.logger.info(f"  原始分布: 活性={n_active}, 非活性={n_inactive}")
        self.logger.info(f"  原始比例: {n_active/max(n_inactive,1):.2f}:1")
        
        if method == "undersample":
            target_inactive = int(n_active / target_ratio)
            
            if n_inactive > target_inactive:
                df_active = df[df['label'] == 1]
                df_inactive = df[df['label'] == 0].sample(
                    n=target_inactive, random_state=42
                )
                df = pd.concat([df_active, df_inactive])
                
                self.logger.info(f"  ✓ 欠采样: 活性={len(df_active)}, 非活性={len(df_inactive)}")
            else:
                self.logger.info(f"  非活性样本已少于目标，不进行欠采样")
        
        elif method == "oversample":
            # 过采样少数类
            if n_inactive < n_active / target_ratio:
                target_inactive = int(n_active / target_ratio)
                df_active = df[df['label'] == 1]
                df_inactive = df[df['label'] == 0]
                
                # 重复采样
                df_inactive_over = df_inactive.sample(
                    n=target_inactive, replace=True, random_state=42
                )
                df = pd.concat([df_active, df_inactive_over])
                
                self.logger.info(f"  ✓ 过采样: 活性={len(df_active)}, 非活性={len(df_inactive_over)}")
        
        final_active = (df['label'] == 1).sum()
        final_inactive = (df['label'] == 0).sum()
        self.logger.info(f"  最终分布: 活性={final_active}, 非活性={final_inactive}")
        self.logger.info(f"  最终比例: {final_active/max(final_inactive,1):.2f}:1")
        
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
            'label_confidence',  # 新增
            'MW', 'LogP', 'TPSA', 'HBA', 'HBD',
            'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
        ]
        
        # 检查哪些列存在
        existing_cols = [col for col in columns_to_save if col in df.columns]
        
        df_save = df[existing_cols].copy()
        df_save.to_csv(output_path, index=False)
        
        self.logger.info(f"\n数据已保存到: {output_path}")
    
    def save_statistics(self, df: pd.DataFrame, output_path: str):
        """保存数据统计信息"""
        import json
        
        stats = {
            "总体统计": {
                "total_compounds": len(df),
                "active_compounds": int((df['label'] == 1).sum()),
                "inactive_compounds": int((df['label'] == 0).sum()),
                "active_inactive_ratio": float((df['label'] == 1).sum() / max((df['label'] == 0).sum(), 1))
            },
            "活性类型分布": df['standard_type'].value_counts().to_dict(),
            "分子性质": {
                "MW": {
                    "mean": float(df['MW'].mean()),
                    "std": float(df['MW'].std()),
                    "min": float(df['MW'].min()),
                    "max": float(df['MW'].max())
                },
                "LogP": {
                    "mean": float(df['LogP'].mean()),
                    "std": float(df['LogP'].std())
                }
            },
            "标签置信度": {
                "mean": float(df['label_confidence'].mean()),
                "high_confidence": int((df['label_confidence'] > 0.8).sum()),
                "medium_confidence": int(((df['label_confidence'] > 0.5) & (df['label_confidence'] <= 0.8)).sum()),
                "low_confidence": int((df['label_confidence'] <= 0.5).sum())
            }
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"统计信息已保存到: {output_path}")
    
    def run(self, input_path: str) -> str:
        """运行完整的预处理流程"""
        log_section(self.logger, "改进的数据预处理")
        
        # 1. 加载数据
        df = self.load_raw_data(input_path)
        
        # 2. 单位转换
        df = self.convert_units(df)
        
        # 3. 改进的标签分配
        df = self.assign_labels_improved(df)
        
        # 4. 标准化
        df = self.standardize_molecules(df)
        
        # 5. 去重
        df = self.remove_duplicates(df)
        
        # 6. 平衡
        df = self.balance_dataset(df)
        
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
        self.logger.info(f"最终数据集:")
        self.logger.info(f"  总化合物数: {len(df)}")
        self.logger.info(f"  活性化合物: {(df['label'] == 1).sum()}")
        self.logger.info(f"  非活性化合物: {(df['label'] == 0).sum()}")
        self.logger.info(f"  比例: {(df['label'] == 1).sum() / max((df['label'] == 0).sum(), 1):.2f}:1")
        
        return str(output_path)


def main():
    """主函数"""
    config = load_data_config()
    
    input_dir = config['paths']['raw_data_dir']
    input_file = config['filenames']['raw_data']
    input_path = Path(input_dir) / input_file
    
    if not input_path.exists():
        print(f"错误: 找不到原始数据文件: {input_path}")
        print("请先运行: python experiments/stage0_data/01_download_chembl_improved.py")
        return
    
    preprocessor = ImprovedDataPreprocessor(config)
    output_path = preprocessor.run(str(input_path))
    
    print(f"\n{'='*70}")
    print("✓ 预处理完成")
    print(f"{'='*70}")
    print(f"\n文件位置: {output_path}")
    print("\n💡 改进点:")
    print("  ✓ 更合理的活性阈值（活性<5μM, 非活性>30μM）")
    print("  ✓ 中间区域智能分配（减少数据流失）")
    print("  ✓ 标签置信度评分")
    print("  ✓ 智能去重（保留最可靠记录）")
    print("\n下一步:")
    print("  python experiments/stage0_data/03_split_dataset.py")


if __name__ == "__main__":
    main()
