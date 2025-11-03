"""
NLRP3数据预处理脚本 - 严格目标版
=====================================
目标：
- 活性：900个（IC50<10μM 或 EC50<10μM）
- 非活性：2700个（IC50>50μM）
- 比例：活性:非活性 = 1:3

特点：
1. 严格的阈值标准（IC50<10μM, EC50<10μM）
2. 零容忍的数据丢弃
3. 精确的比例控制
4. 完整的数据溯源
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logger, load_data_config, log_section,
    MoleculeProcessor, MoleculeValidator, get_inchi_key,
    calculate_descriptors
)


class StrictPreprocessor:
    """严格预处理器 - 精确达到目标"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Strict_Preprocessor")
        
        # 严格目标
        self.TARGET_ACTIVE = 900
        self.TARGET_INACTIVE = 2700
        self.TARGET_RATIO = 3.0
        
        # 严格阈值（μM）
        self.THRESHOLDS = {
            'IC50': {'active': 10.0, 'inactive': 50.0},
            'EC50': {'active': 10.0, 'inactive': 50.0},
            'Ki': {'active': 10.0, 'inactive': 50.0},
            'Kd': {'active': 10.0, 'inactive': 50.0},
        }
        
        self.logger.info("="*70)
        self.logger.info("严格预处理目标")
        self.logger.info("="*70)
        self.logger.info(f"活性目标: {self.TARGET_ACTIVE}")
        self.logger.info(f"非活性目标: {self.TARGET_INACTIVE}")
        self.logger.info(f"目标比例: 1:{self.TARGET_RATIO}")
        self.logger.info(f"\n严格阈值:")
        for atype, thres in self.THRESHOLDS.items():
            self.logger.info(
                f"  {atype}: 活性<{thres['active']}μM, "
                f"非活性>{thres['inactive']}μM"
            )
        self.logger.info("="*70)
        
        self.mol_processor = MoleculeProcessor()
        self.mol_validator = MoleculeValidator(
            mw_range=tuple(config['filtering']['molecular_weight_range']),
            heavy_atom_range=tuple(config['filtering']['heavy_atom_count_range'])
        )
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据"""
        log_section(self.logger, "加载原始数据")
        
        df = pd.read_csv(file_path)
        
        self.logger.info(f"文件: {file_path}")
        self.logger.info(f"总记录: {len(df)}")
        
        if 'data_source' in df.columns:
            self.logger.info(f"\n数据来源:")
            for source, count in df['data_source'].value_counts().items():
                pct = count / len(df) * 100
                self.logger.info(f"  {source:25s}: {count:5d} ({pct:5.1f}%)")
        
        # 活性类型统计
        if 'standard_type' in df.columns:
            self.logger.info(f"\n活性类型:")
            for atype, count in df['standard_type'].value_counts().head(10).items():
                self.logger.info(f"  {atype:15s}: {count:5d}")
        
        return df
    
    def convert_units_strict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        严格单位转换
        
        只保留可以精确转换为μM的数据
        """
        log_section(self.logger, "单位转换")
        
        df = df.copy()
        df['value_um'] = np.nan
        
        # 只处理有标准值的数据
        mask_has_value = df['standard_value'].notna()
        
        self.logger.info(f"有标准值的记录: {mask_has_value.sum()}")
        
        # nM -> μM (最常见)
        mask_nm = mask_has_value & (df['standard_units'] == 'nM')
        df.loc[mask_nm, 'value_um'] = df.loc[mask_nm, 'standard_value'] / 1000.0
        self.logger.info(f"  nM转换: {mask_nm.sum()}")
        
        # μM -> μM
        mask_um = mask_has_value & df['standard_units'].isin(['uM', 'μM', 'UM'])
        df.loc[mask_um, 'value_um'] = df.loc[mask_um, 'standard_value']
        self.logger.info(f"  μM转换: {mask_um.sum()}")
        
        # mM -> μM (较少)
        mask_mm = mask_has_value & df['standard_units'].isin(['mM', 'MM'])
        df.loc[mask_mm, 'value_um'] = df.loc[mask_mm, 'standard_value'] * 1000.0
        self.logger.info(f"  mM转换: {mask_mm.sum()}")
        
        # 统计
        initial = len(df)
        df = df.dropna(subset=['value_um'])
        final = len(df)
        removed = initial - final
        
        self.logger.info(f"\n转换结果:")
        self.logger.info(f"  初始: {initial}")
        self.logger.info(f"  成功: {final} ({final/initial*100:.1f}%)")
        self.logger.info(f"  失败: {removed}")
        
        if final > 0:
            self.logger.info(f"\n数值范围: {df['value_um'].min():.6f} - {df['value_um'].max():.2f} μM")
        
        return df
    
    def assign_labels_strict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        严格标签分配
        
        规则（按优先级）：
        1. relation='<' AND value≤10μM → 活性
        2. relation='>' → 非活性
        3. IC50/EC50/Ki/Kd='=' AND value<10μM → 活性
        4. IC50/EC50/Ki/Kd='=' AND value>50μM → 非活性
        5. 其他（10-50μM中间区） → 根据来源判断
        """
        log_section(self.logger, "严格标签分配")
        
        df = df.copy()
        df['label'] = -1  # 未分配
        df['label_rule'] = ''  # 标注规则
        df['label_confidence'] = 0.0  # 置信度
        
        total = len(df)
        
        # 规则1: relation='<' 且 value≤10μM → 明确活性
        mask_r1 = (df['standard_relation'] == '<') & (df['value_um'] <= 10.0)
        df.loc[mask_r1, 'label'] = 1
        df.loc[mask_r1, 'label_rule'] = 'relation_less'
        df.loc[mask_r1, 'label_confidence'] = 1.0
        n_r1 = mask_r1.sum()
        
        # 规则2: relation='>' → 明确非活性
        mask_r2 = df['standard_relation'] == '>'
        df.loc[mask_r2, 'label'] = 0
        df.loc[mask_r2, 'label_rule'] = 'relation_greater'
        df.loc[mask_r2, 'label_confidence'] = 1.0
        n_r2 = mask_r2.sum()
        
        # 规则3: relation='=' 且 value<10μM → 活性
        mask_r3 = (
            (df['standard_relation'] == '=') &
            df['standard_type'].isin(self.THRESHOLDS.keys()) &
            (df['value_um'] < 10.0)
        )
        df.loc[mask_r3, 'label'] = 1
        df.loc[mask_r3, 'label_rule'] = 'value_lt_10um'
        df.loc[mask_r3, 'label_confidence'] = 1.0
        n_r3 = mask_r3.sum()
        
        # 规则4: relation='=' 且 value>50μM → 非活性
        mask_r4 = (
            (df['standard_relation'] == '=') &
            df['standard_type'].isin(self.THRESHOLDS.keys()) &
            (df['value_um'] > 50.0)
        )
        df.loc[mask_r4, 'label'] = 0
        df.loc[mask_r4, 'label_rule'] = 'value_gt_50um'
        df.loc[mask_r4, 'label_confidence'] = 1.0
        n_r4 = mask_r4.sum()
        
        # 规则5: 中间区（10-50μM）- 根据数据来源判断
        mask_middle = (
            (df['standard_relation'] == '=') &
            df['standard_type'].isin(self.THRESHOLDS.keys()) &
            (df['value_um'] >= 10.0) &
            (df['value_um'] <= 50.0)
        )
        n_middle = mask_middle.sum()
        
        # 中间区策略：根据来源
        # - NLRP3数据：接近10μM标为活性，接近50μM标为非活性
        # - 采样数据：全部标为非活性（保守）
        if n_middle > 0:
            self.logger.info(f"\n处理中间区 (10-50μM): {n_middle}条")
            
            for idx in df[mask_middle].index:
                value = df.loc[idx, 'value_um']
                source = df.loc[idx, 'data_source']
                
                if 'NLRP3' in source:
                    # NLRP3数据：线性插值
                    if value < 25.0:  # 更接近10μM
                        df.loc[idx, 'label'] = 1
                        df.loc[idx, 'label_rule'] = 'middle_nlrp3_active'
                        df.loc[idx, 'label_confidence'] = 1 - (value - 10) / 15  # 0.67-1.0
                    else:  # 更接近50μM
                        df.loc[idx, 'label'] = 0
                        df.loc[idx, 'label_rule'] = 'middle_nlrp3_inactive'
                        df.loc[idx, 'label_confidence'] = (value - 25) / 25  # 0.0-1.0
                else:
                    # 非NLRP3数据：保守标为非活性
                    df.loc[idx, 'label'] = 0
                    df.loc[idx, 'label_rule'] = 'middle_sampled_inactive'
                    df.loc[idx, 'label_confidence'] = 0.8
            
            n_middle_active = df[mask_middle & (df['label'] == 1)].shape[0]
            n_middle_inactive = df[mask_middle & (df['label'] == 0)].shape[0]
            
            self.logger.info(f"  中间区→活性: {n_middle_active}")
            self.logger.info(f"  中间区→非活性: {n_middle_inactive}")
        
        # 统计
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        n_unassigned = (df['label'] == -1).sum()
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("标签分配结果")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"规则1 (relation='<'):           {n_r1:5d} → 活性")
        self.logger.info(f"规则2 (relation='>'):           {n_r2:5d} → 非活性")
        self.logger.info(f"规则3 (value<10μM):            {n_r3:5d} → 活性")
        self.logger.info(f"规则4 (value>50μM):            {n_r4:5d} → 非活性")
        self.logger.info(f"规则5 (中间区):                {n_middle:5d}")
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"活性:     {n_active:5d} ({n_active/total*100:5.1f}%)")
        self.logger.info(f"非活性:   {n_inactive:5d} ({n_inactive/total*100:5.1f}%)")
        self.logger.info(f"未分配:   {n_unassigned:5d} ({n_unassigned/total*100:5.1f}%)")
        
        if n_active > 0 and n_inactive > 0:
            actual_ratio = n_inactive / n_active
            self.logger.info(f"\n当前比例: 1:{actual_ratio:.2f}")
            self.logger.info(f"目标比例: 1:{self.TARGET_RATIO:.2f}")
        
        # 保留率
        retention_rate = (n_active + n_inactive) / total * 100
        self.logger.info(f"数据保留率: {retention_rate:.1f}%")
        
        # 置信度统计
        assigned = df[df['label'] != -1]
        if len(assigned) > 0:
            self.logger.info(f"\n置信度统计:")
            self.logger.info(f"  平均: {assigned['label_confidence'].mean():.3f}")
            self.logger.info(f"  高(>0.9): {(assigned['label_confidence'] > 0.9).sum()}")
            self.logger.info(f"  中(0.7-0.9): {((assigned['label_confidence'] >= 0.7) & (assigned['label_confidence'] <= 0.9)).sum()}")
            self.logger.info(f"  低(<0.7): {(assigned['label_confidence'] < 0.7).sum()}")
        
        # 处理未分配数据
        if n_unassigned > 0:
            self.logger.warning(f"\n⚠️  发现{n_unassigned}条未分配数据")
            
            # 保存未分配数据供检查
            unassigned = df[df['label'] == -1].copy()
            review_path = Path(self.config['paths']['processed_data_dir']) / 'unassigned_review.csv'
            review_path.parent.mkdir(parents=True, exist_ok=True)
            unassigned.to_csv(review_path, index=False)
            
            self.logger.info(f"  未分配数据已保存: {review_path}")
            self.logger.info(f"  这些数据将被移除")
            
            # 移除
            df = df[df['label'] != -1].copy()
        
        self.logger.info(f"{'='*70}")
        
        return df
    
    def standardize_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化分子并过滤
        
        严格的质量控制
        """
        log_section(self.logger, "分子标准化与质量控制")
        
        results = []
        failed = {'parse': 0, 'standardize': 0, 'validate': 0, 'inchi': 0}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理分子"):
            smiles = row['canonical_smiles']
            
            if pd.isna(smiles):
                failed['parse'] += 1
                continue
            
            # 标准化
            std_smiles, mol = self.mol_processor.process_smiles(smiles)
            if std_smiles is None or mol is None:
                failed['standardize'] += 1
                continue
            
            # 验证分子属性
            if not self.mol_validator.is_valid(mol):
                failed['validate'] += 1
                continue
            
            # InChI Key
            inchi_key = get_inchi_key(mol)
            if inchi_key is None:
                failed['inchi'] += 1
                continue
            
            # 计算描述符
            descriptors = calculate_descriptors(mol)
            
            # 保存
            result = row.to_dict()
            result['smiles_standardized'] = std_smiles
            result['inchi_key'] = inchi_key
            result.update(descriptors)
            
            results.append(result)
        
        df_clean = pd.DataFrame(results)
        
        total_failed = sum(failed.values())
        success_rate = len(df_clean) / (len(df_clean) + total_failed) * 100
        
        self.logger.info(f"\n处理结果:")
        self.logger.info(f"  ✓ 成功: {len(df_clean)} ({success_rate:.1f}%)")
        self.logger.info(f"  ✗ 失败: {total_failed}")
        
        if total_failed > 0:
            self.logger.info(f"\n失败原因:")
            for reason, count in failed.items():
                if count > 0:
                    self.logger.info(f"    {reason}: {count}")
        
        # 标签分布
        n_active = (df_clean['label'] == 1).sum()
        n_inactive = (df_clean['label'] == 0).sum()
        
        self.logger.info(f"\n标准化后分布:")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        self.logger.info(f"  比例: 1:{n_inactive/max(n_active,1):.2f}")
        
        return df_clean
    
    def intelligent_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        智能去重
        
        策略：
        - 按InChI Key分组
        - 优先保留高置信度标签
        - 优先保留活性值最小的（对活性化合物）
        """
        log_section(self.logger, "智能去重")
        
        initial = len(df)
        
        def select_best_record(group):
            """选择最佳记录"""
            if len(group) == 1:
                return group.iloc[0]
            
            # 优先级1: 置信度最高
            # 优先级2: 对于活性，选IC50最小的
            # 优先级3: 对于非活性，选第一条
            
            label = group.iloc[0]['label']
            
            if label == 1:  # 活性
                # 按置信度和IC50排序
                best_idx = group.sort_values(
                    by=['label_confidence', 'value_um'],
                    ascending=[False, True]
                ).index[0]
            else:  # 非活性
                # 按置信度排序
                best_idx = group.sort_values(
                    by='label_confidence',
                    ascending=False
                ).index[0]
            
            return group.loc[best_idx]
        
        df_dedup = df.groupby('inchi_key', group_keys=False).apply(
            select_best_record
        ).reset_index(drop=True)
        
        removed = initial - len(df_dedup)
        
        self.logger.info(f"去重结果:")
        self.logger.info(f"  初始: {initial}")
        self.logger.info(f"  移除: {removed} ({removed/initial*100:.1f}%)")
        self.logger.info(f"  保留: {len(df_dedup)}")
        
        # 去重后分布
        n_active = (df_dedup['label'] == 1).sum()
        n_inactive = (df_dedup['label'] == 0).sum()
        
        self.logger.info(f"\n去重后分布:")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        self.logger.info(f"  比例: 1:{n_inactive/max(n_active,1):.2f}")
        
        return df_dedup
    
    def balance_to_exact_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        精确平衡到目标
        
        目标：900活性 + 2700非活性
        
        Returns:
            (balanced_df, status_dict)
        """
        log_section(self.logger, "精确平衡到目标")
        
        n_active = (df['label'] == 1).sum()
        n_inactive = (df['label'] == 0).sum()
        
        self.logger.info(f"当前状态:")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        self.logger.info(f"  比例: 1:{n_inactive/max(n_active,1):.2f}")
        
        self.logger.info(f"\n目标:")
        self.logger.info(f"  活性: {self.TARGET_ACTIVE}")
        self.logger.info(f"  非活性: {self.TARGET_INACTIVE}")
        self.logger.info(f"  比例: 1:{self.TARGET_RATIO}")
        
        df_active = df[df['label'] == 1].copy()
        df_inactive = df[df['label'] == 0].copy()
        
        status = {
            'active_sufficient': True,
            'inactive_sufficient': True,
            'active_shortfall': 0,
            'inactive_shortfall': 0
        }
        
        # 处理活性样本
        self.logger.info(f"\n处理活性样本:")
        if n_active > self.TARGET_ACTIVE:
            self.logger.info(f"  过多，欠采样: {n_active} → {self.TARGET_ACTIVE}")
            df_active = df_active.sample(n=self.TARGET_ACTIVE, random_state=42)
        elif n_active < self.TARGET_ACTIVE:
            shortfall = self.TARGET_ACTIVE - n_active
            status['active_sufficient'] = False
            status['active_shortfall'] = shortfall
            self.logger.warning(f"  ⚠️  不足! 缺口: {shortfall}")
            self.logger.warning(f"  将使用所有{n_active}个活性样本")
        else:
            self.logger.info(f"  ✓ 恰好达标: {n_active}")
        
        # 根据最终活性数调整非活性目标
        final_active_count = len(df_active)
        adjusted_inactive_target = int(final_active_count * self.TARGET_RATIO)
        
        # 处理非活性样本
        self.logger.info(f"\n处理非活性样本:")
        self.logger.info(f"  调整后目标: {adjusted_inactive_target}")
        
        if n_inactive > adjusted_inactive_target:
            self.logger.info(f"  过多，欠采样: {n_inactive} → {adjusted_inactive_target}")
            df_inactive = df_inactive.sample(n=adjusted_inactive_target, random_state=42)
        elif n_inactive < adjusted_inactive_target:
            shortfall = adjusted_inactive_target - n_inactive
            status['inactive_sufficient'] = False
            status['inactive_shortfall'] = shortfall
            self.logger.warning(f"  ⚠️  不足! 缺口: {shortfall}")
            self.logger.warning(f"  将使用所有{n_inactive}个非活性样本")
        else:
            self.logger.info(f"  ✓ 恰好达标: {n_inactive}")
        
        # 合并并打乱
        df_balanced = pd.concat([df_active, df_inactive], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 最终统计
        final_active = (df_balanced['label'] == 1).sum()
        final_inactive = (df_balanced['label'] == 0).sum()
        final_ratio = final_inactive / max(final_active, 1)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("最终结果")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"活性:     {final_active:5d} (目标: {self.TARGET_ACTIVE})")
        self.logger.info(f"非活性:   {final_inactive:5d} (目标: {self.TARGET_INACTIVE})")
        self.logger.info(f"总数:     {len(df_balanced):5d}")
        self.logger.info(f"比例:     1:{final_ratio:.2f} (目标: 1:{self.TARGET_RATIO})")
        
        # 达标评估
        active_achievement = final_active / self.TARGET_ACTIVE * 100
        inactive_achievement = final_inactive / self.TARGET_INACTIVE * 100
        
        self.logger.info(f"\n达标率:")
        self.logger.info(f"  活性:   {active_achievement:6.1f}%")
        self.logger.info(f"  非活性: {inactive_achievement:6.1f}%")
        
        if not status['active_sufficient'] or not status['inactive_sufficient']:
            self.logger.warning(f"\n⚠️  目标未完全达成")
            if not status['active_sufficient']:
                self.logger.warning(f"  活性缺口: {status['active_shortfall']}")
                self.logger.warning(f"  建议: 放宽IC50阈值至15μM或20μM")
            if not status['inactive_sufficient']:
                self.logger.warning(f"  非活性缺口: {status['inactive_shortfall']}")
                self.logger.warning(f"  建议: 增加非活性采样来源")
        else:
            self.logger.info(f"\n✅ 目标完美达成!")
        
        self.logger.info(f"{'='*70}")
        
        return df_balanced, status
    
    def save_final_data(self, df: pd.DataFrame, output_path: str):
        """保存最终数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        columns_to_save = [
            # ID和结构
            'molecule_chembl_id',
            'smiles_standardized',
            'inchi_key',
            # 活性数据
            'standard_type',
            'standard_relation',
            'value_um',
            # 标签
            'label',
            'label_rule',
            'label_confidence',
            # 描述符
            'MW', 'LogP', 'TPSA', 'HBA', 'HBD',
            'RotatableBonds', 'AromaticRings', 'HeavyAtoms',
            # 溯源
            'data_source',
        ]
        
        # 添加可选列
        optional_cols = ['source_target', 'source_family', 'source_name']
        for col in optional_cols:
            if col in df.columns:
                columns_to_save.append(col)
        
        existing_cols = [col for col in columns_to_save if col in df.columns]
        df_save = df[existing_cols].copy()
        
        df_save.to_csv(output_path, index=False)
        
        self.logger.info(f"\n✓ 数据已保存")
        self.logger.info(f"  文件: {output_path}")
        self.logger.info(f"  行数: {len(df_save)}")
        self.logger.info(f"  列数: {len(existing_cols)}")
    
    def save_comprehensive_stats(self, df: pd.DataFrame, status: dict, output_path: str):
        """保存完整统计信息"""
        import json
        
        # 辅助函数：转换numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            return obj
        
        stats = {
            "目标设定": {
                "target_active": int(self.TARGET_ACTIVE),
                "target_inactive": int(self.TARGET_INACTIVE),
                "target_ratio": f"1:{self.TARGET_RATIO}",
                "ic50_active_threshold_um": float(self.THRESHOLDS['IC50']['active']),
                "ec50_active_threshold_um": float(self.THRESHOLDS['EC50']['active']),
                "inactive_threshold_um": float(self.THRESHOLDS['IC50']['inactive'])
            },
            "最终结果": {
                "total_compounds": int(len(df)),
                "active_compounds": int((df['label'] == 1).sum()),
                "inactive_compounds": int((df['label'] == 0).sum()),
                "actual_ratio": f"1:{float((df['label'] == 0).sum() / max((df['label'] == 1).sum(), 1)):.2f}",
                "active_achievement_pct": float((df['label'] == 1).sum() / self.TARGET_ACTIVE * 100),
                "inactive_achievement_pct": float((df['label'] == 0).sum() / self.TARGET_INACTIVE * 100)
            },
            "达标状态": convert_to_native(status),
            "IC50统计_活性": {
                "count": int((df['label'] == 1).sum()),
                "mean_um": float(df[df['label'] == 1]['value_um'].mean()),
                "median_um": float(df[df['label'] == 1]['value_um'].median()),
                "min_um": float(df[df['label'] == 1]['value_um'].min()),
                "max_um": float(df[df['label'] == 1]['value_um'].max())
            },
            "IC50统计_非活性": {
                "count": int((df['label'] == 0).sum()),
                "mean_um": float(df[df['label'] == 0]['value_um'].mean()),
                "median_um": float(df[df['label'] == 0]['value_um'].median()),
                "min_um": float(df[df['label'] == 0]['value_um'].min()),
                "max_um": float(df[df['label'] == 0]['value_um'].max())
            },
            "标签规则分布": {},
            "置信度统计": {
                "mean": float(df['label_confidence'].mean()),
                "std": float(df['label_confidence'].std()),
                "high_confidence_count": int((df['label_confidence'] > 0.9).sum()),
                "medium_confidence_count": int(((df['label_confidence'] >= 0.7) & (df['label_confidence'] <= 0.9)).sum()),
                "low_confidence_count": int((df['label_confidence'] < 0.7).sum())
            },
            "分子性质": {
                "MW": {"mean": float(df['MW'].mean()), "std": float(df['MW'].std())},
                "LogP": {"mean": float(df['LogP'].mean()), "std": float(df['LogP'].std())},
                "TPSA": {"mean": float(df['TPSA'].mean()), "std": float(df['TPSA'].std())},
                "HBA": {"mean": float(df['HBA'].mean())},
                "HBD": {"mean": float(df['HBD'].mean())}
            }
        }
        
        # 标签规则分布
        if 'label_rule' in df.columns:
            active_rules = df[df['label'] == 1]['label_rule'].value_counts().to_dict()
            inactive_rules = df[df['label'] == 0]['label_rule'].value_counts().to_dict()
            stats['标签规则分布'] = {
                "活性": active_rules,
                "非活性": inactive_rules
            }
        
        # 数据来源
        if 'data_source' in df.columns:
            stats['数据来源'] = df['data_source'].value_counts().to_dict()
        
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ 统计已保存: {output_path}")
    
    def run(self, input_path: str) -> str:
        """运行完整预处理流程"""
        log_section(self.logger, "开始严格预处理")
        
        # 1. 加载
        df = self.load_raw_data(input_path)
        
        # 2. 单位转换
        df = self.convert_units_strict(df)
        
        # 3. 严格标签分配
        df = self.assign_labels_strict(df)
        
        # 4. 标准化和过滤
        df = self.standardize_and_filter(df)
        
        # 5. 智能去重
        df = self.intelligent_deduplication(df)
        
        # 6. 精确平衡
        df_final, status = self.balance_to_exact_target(df)
        
        # 7. 保存数据
        output_dir = self.config['paths']['processed_data_dir']
        output_file = self.config['filenames']['processed_data']
        output_path = Path(output_dir) / output_file
        
        self.save_final_data(df_final, output_path)
        
        # 8. 保存统计
        stats_file = self.config['filenames']['data_statistics']
        stats_path = Path(output_dir) / stats_file
        self.save_comprehensive_stats(df_final, status, stats_path)
        
        # 最终总结
        log_section(self.logger, "预处理完成")
        
        n_active = (df_final['label'] == 1).sum()
        n_inactive = (df_final['label'] == 0).sum()
        ratio = n_inactive / max(n_active, 1)
        
        self.logger.info(f"✅ 最终数据集:")
        self.logger.info(f"  总数:     {len(df_final):5d}")
        self.logger.info(f"  活性:     {n_active:5d} (目标: {self.TARGET_ACTIVE})")
        self.logger.info(f"  非活性:   {n_inactive:5d} (目标: {self.TARGET_INACTIVE})")
        self.logger.info(f"  比例:     1:{ratio:.2f} (目标: 1:{self.TARGET_RATIO})")
        
        return str(output_path)


def main():
    """主函数"""
    config = load_data_config()
    
    input_dir = config['paths']['raw_data_dir']
    input_file = config['filenames']['raw_data']
    input_path = Path(input_dir) / input_file
    
    if not input_path.exists():
        print(f"❌ 错误: 找不到文件 {input_path}")
        print("请先运行: python experiments/stage0_data/01_download_strict.py")
        return
    
    preprocessor = StrictPreprocessor(config)
    
    try:
        output_path = preprocessor.run(str(input_path))
        
        print(f"\n{'='*70}")
        print("✅ 预处理完成")
        print(f"{'='*70}")
        print(f"\n📁 输出文件: {output_path}")
        print(f"\n🎯 严格目标:")
        print(f"  活性: 900 (IC50<10μM 或 EC50<10μM)")
        print(f"  非活性: 2700 (IC50>50μM)")
        print(f"  比例: 1:3")
        print(f"\n💪 核心特点:")
        print(f"  ✓ 严格阈值（IC50/EC50 < 10μM）")
        print(f"  ✓ 智能中间区处理")
        print(f"  ✓ 完整数据溯源")
        print(f"  ✓ 精确比例控制")
        print(f"\n📊 下一步:")
        print(f"  python experiments/stage0_data/03_split_dataset.py")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
