"""
NLRP3数据采集 - 优化版
关键改进：
1. 大幅增加NLRP3原始数据量（考虑预处理损失）
2. 更激进的过采样策略
3. 更全面的靶点搜索
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, load_data_config, log_section


class OptimizedDataCollector:
    """优化的数据采集器 - 确保足够的原始数据"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Optimized_Data_Collector")
        
        # 目标参数（考虑各阶段损失）
        self.target_final_active = 900
        self.target_ratio = 3.0
        
        # 预估损失率
        self.loss_rates = {
            'unit_conversion': 0.23,    # 单位转换失败率
            'standardization': 0.01,    # 标准化失败率
            'deduplication': 0.23,      # 去重损失率
            'labeling': 0.45,           # 标签分配（55%变活性）
        }
        
        # 计算需要的原始数据量
        self.calculate_raw_data_needed()
        
        # 检查ChEMBL
        try:
            from chembl_webresource_client.new_client import new_client
            self.client_available = True
            self.chembl = new_client
            self.logger.info("✓ ChEMBL客户端可用")
        except ImportError:
            self.client_available = False
            self.logger.warning("✗ ChEMBL不可用，将使用示例数据")
    
    def calculate_raw_data_needed(self):
        """计算需要采集的原始数据量"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("数据需求计算")
        self.logger.info(f"{'='*70}")
        
        # 反向计算
        needed = self.target_final_active
        
        self.logger.info(f"\n预期损失率:")
        for stage, rate in self.loss_rates.items():
            needed = needed / (1 - rate)
            self.logger.info(f"  {stage}: {rate*100:.0f}% → 需要 {int(needed)}")
        
        # 最终需要的NLRP3原始数据（再加20%安全边际）
        self.raw_nlrp3_needed = int(needed * 1.2)
        
        # 非活性数据（目标3倍，也需要过采样）
        self.raw_inactive_needed = int(self.target_final_active * self.target_ratio * 1.5)
        
        self.logger.info(f"\n📊 最终采集目标:")
        self.logger.info(f"  NLRP3数据: {self.raw_nlrp3_needed} 条")
        self.logger.info(f"  非活性数据: {self.raw_inactive_needed} 条")
        self.logger.info(f"  总计: {self.raw_nlrp3_needed + self.raw_inactive_needed} 条")
    
    def download_nlrp3_data_comprehensive(self) -> pd.DataFrame:
        """
        全面下载NLRP3数据
        
        策略：
        1. 搜索所有NLRP3相关靶点
        2. 下载所有测量类型（不只IC50）
        3. 不限制organism（包括小鼠等）
        """
        if not self.client_available:
            return pd.DataFrame()
        
        log_section(self.logger, "步骤1: 全面下载NLRP3数据")
        
        try:
            # 搜索所有NLRP3相关靶点
            search_terms = ['NLRP3', 'NALP3', 'Cryopyrin', 'CIAS1']
            
            all_targets = []
            for term in search_terms:
                try:
                    targets = self.chembl.target.filter(
                        target_synonym__icontains=term
                    ).only(['target_chembl_id', 'pref_name', 'organism', 'target_type'])
                    all_targets.extend(list(targets))
                except:
                    continue
            
            # 去重
            unique_targets = {t['target_chembl_id']: t for t in all_targets}
            
            self.logger.info(f"找到 {len(unique_targets)} 个NLRP3相关靶点:")
            for target_id, target in unique_targets.items():
                self.logger.info(f"  {target_id}: {target.get('pref_name', 'N/A')} "
                               f"({target.get('organism', 'N/A')})")
            
            # 下载所有靶点的活性数据
            all_data = []
            
            for target_id, target in unique_targets.items():
                try:
                    self.logger.info(f"\n下载 {target_id} 数据...")
                    
                    # 不限制测量类型，下载所有
                    activities = self.chembl.activity.filter(
                        target_chembl_id=target_id
                    ).only([
                        'molecule_chembl_id',
                        'canonical_smiles',
                        'standard_type',
                        'standard_relation',
                        'standard_value',
                        'standard_units',
                        'pchembl_value',
                        'data_validity_comment'
                    ])
                    
                    data = list(activities)
                    
                    if data:
                        df_temp = pd.DataFrame(data)
                        df_temp['data_source'] = 'NLRP3_Target'
                        df_temp['source_detail'] = f"{target_id}_{target.get('organism', 'Unknown')}"
                        all_data.append(df_temp)
                        self.logger.info(f"  ✓ 获取 {len(df_temp)} 条")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.warning(f"  下载 {target_id} 失败: {e}")
                    continue
            
            if all_data:
                df_all = pd.concat(all_data, ignore_index=True)
                
                # 去重（同一化合物可能在多个靶点中）
                initial_count = len(df_all)
                df_all = df_all.drop_duplicates(
                    subset=['molecule_chembl_id', 'standard_type', 'standard_value'],
                    keep='first'
                )
                
                self.logger.info(f"\n✓ NLRP3数据采集完成:")
                self.logger.info(f"  原始: {initial_count}")
                self.logger.info(f"  去重后: {len(df_all)}")
                self.logger.info(f"  目标: {self.raw_nlrp3_needed}")
                
                if len(df_all) < self.raw_nlrp3_needed:
                    shortage = self.raw_nlrp3_needed - len(df_all)
                    self.logger.warning(f"  ⚠️  数据不足 {shortage} 条")
                    self.logger.warning(f"  将用示例数据补充")
                
                return df_all
            else:
                self.logger.warning("未找到NLRP3数据")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"下载NLRP3数据失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def sample_negative_compounds_aggressive(self, n_needed: int) -> pd.DataFrame:
        """
        更激进的负样本采样
        
        改进：
        1. 更多靶点家族
        2. 更大的采样量
        3. 包含更多测量类型
        """
        if not self.client_available:
            return pd.DataFrame()
        
        log_section(self.logger, f"步骤2: 采样负样本 (目标{n_needed}个)")
        
        try:
            # 扩展靶点列表（10个不同家族）
            diverse_targets = [
                ('CHEMBL1862', 'JAK2', 'Kinase'),
                ('CHEMBL203', 'EGFR', 'RTK'),
                ('CHEMBL1824', 'Cathepsin', 'Protease'),
                ('CHEMBL1951', 'hERG', 'Ion_Channel'),
                ('CHEMBL2035', 'CCR5', 'GPCR'),
                ('CHEMBL1075104', 'PDE4', 'Enzyme'),
                ('CHEMBL1955', 'AKT1', 'Kinase'),
                ('CHEMBL2095192', 'HDAC1', 'Epigenetic'),
                ('CHEMBL3371', 'NOS', 'Enzyme'),
                ('CHEMBL340', 'Tubulin', 'Structural'),
            ]
            
            all_negatives = []
            samples_per_target = (n_needed // len(diverse_targets)) + 300  # 更多备用
            
            for target_id, target_name, family in diverse_targets:
                try:
                    self.logger.info(f"\n  采样自 {target_name} ({family})...")
                    
                    # 扩展采样策略
                    activities = self.chembl.activity.filter(
                        target_chembl_id=target_id,
                        standard_type__in=['IC50', 'EC50', 'Ki'],  # 多种类型
                        standard_relation='>',
                        standard_value__gte=10000
                    ).only([
                        'molecule_chembl_id',
                        'canonical_smiles',
                        'standard_type',
                        'standard_relation',
                        'standard_value',
                        'standard_units'
                    ])
                    
                    data = list(activities)[:samples_per_target]
                    
                    if data:
                        df_temp = pd.DataFrame(data)
                        df_temp['data_source'] = 'Sampled_Negative'
                        df_temp['source_detail'] = f"{family}_{target_name}"
                        all_negatives.append(df_temp)
                        self.logger.info(f"    ✓ 获取 {len(df_temp)}")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"    采样 {target_name} 失败: {e}")
                    continue
                
                # 检查是否够了
                if all_negatives:
                    total = sum(len(df) for df in all_negatives)
                    if total >= n_needed * 1.2:
                        break
            
            if all_negatives:
                df_neg = pd.concat(all_negatives, ignore_index=True)
                
                # 去重
                initial_count = len(df_neg)
                df_neg = df_neg.drop_duplicates(subset=['canonical_smiles'], keep='first')
                self.logger.info(f"\n  去重: {initial_count} → {len(df_neg)}")
                
                # 采样到目标
                if len(df_neg) > n_needed:
                    df_neg = df_neg.sample(n=n_needed, random_state=42)
                
                self.logger.info(f"\n✓ 负样本采集完成: {len(df_neg)}")
                return df_neg
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"采样失败: {e}")
            return pd.DataFrame()
    
    def generate_realistic_data(self, n_active: int, n_inactive: int) -> pd.DataFrame:
        """生成高质量示例数据（作为备用）"""
        self.logger.info(f"\n生成示例数据:")
        self.logger.info(f"  活性: {n_active}")
        self.logger.info(f"  非活性: {n_inactive}")
        
        np.random.seed(42)
        
        # 真实NLRP3抑制剂核心结构
        active_scaffolds = [
            "c1ccc2c(c1)c(c(s2)S(=O)(=O)N)NC(=O)C",
            "c1ccc(cc1)S(=O)(=O)Nc2nccs2",
            "COc1ccc(cc1)C(=O)Nc2ccc(cc2)S(=O)(=O)N",
            "c1ccc(cc1)c2nnc(s2)SCC(=O)N",
            "Cc1ccc(cc1)S(=O)(=O)Nc2ncccn2",
            "c1ccc2c(c1)nc(s2)NC(=O)c3ccccc3",
        ]
        
        inactive_scaffolds = [
            "c1ccccc1", "CCCCc1ccccc1", "c1ccc(cc1)O",
            "COc1ccccc1", "c1ccc(cc1)C(=O)O", "CCCCn1ccnc1",
            "c1ccc2c(c1)cccn2", "c1ccc(cc1)N", "c1ccc(cc1)Cl",
        ]
        
        data = []
        
        # 生成活性
        for i in range(n_active):
            scaffold = np.random.choice(active_scaffolds)
            
            dist = np.random.choice(['potent', 'moderate', 'weak'], p=[0.2, 0.6, 0.2])
            
            if dist == 'potent':
                ic50_nm = np.random.lognormal(np.log(50), 0.5)
                ic50_nm = np.clip(ic50_nm, 10, 100)
            elif dist == 'moderate':
                ic50_nm = np.random.lognormal(np.log(800), 0.8)
                ic50_nm = np.clip(ic50_nm, 100, 5000)
            else:
                ic50_nm = np.random.lognormal(np.log(12000), 0.4)
                ic50_nm = np.clip(ic50_nm, 5000, 20000)
            
            assay_type = np.random.choice(['IC50', 'EC50', 'Ki'], p=[0.7, 0.2, 0.1])
            relation = '='
            
            data.append({
                'molecule_chembl_id': f'CHEMBL{1000000+i}',
                'canonical_smiles': scaffold,
                'standard_type': assay_type,
                'standard_relation': relation,
                'standard_value': ic50_nm,
                'standard_units': 'nM',
                'pchembl_value': -np.log10(ic50_nm / 1e9),
                'data_source': 'Example_Active',
                'source_detail': 'Generated'
            })
        
        # 生成非活性
        for i in range(n_inactive):
            scaffold = np.random.choice(inactive_scaffolds)
            ic50_nm = np.random.uniform(20000, 100000)
            relation = np.random.choice(['=', '>'], p=[0.3, 0.7])
            
            if relation == '>':
                ic50_nm = np.random.choice([10000, 20000, 50000])
            
            data.append({
                'molecule_chembl_id': f'CHEMBL{3000000+i}',
                'canonical_smiles': scaffold,
                'standard_type': 'IC50',
                'standard_relation': relation,
                'standard_value': ic50_nm,
                'standard_units': 'nM',
                'pchembl_value': -np.log10(ic50_nm / 1e9) if relation == '=' else None,
                'data_source': 'Example_Inactive',
                'source_detail': 'Generated'
            })
        
        df = pd.DataFrame(data)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def run(self) -> str:
        """运行完整采集流程"""
        log_section(self.logger, "优化数据采集")
        
        all_data = []
        
        # 1. 下载NLRP3数据
        nlrp3_data = self.download_nlrp3_data_comprehensive()
        
        if len(nlrp3_data) > 0:
            all_data.append(nlrp3_data)
            
            # 如果不够，补充示例数据
            shortage = max(0, self.raw_nlrp3_needed - len(nlrp3_data))
            if shortage > 0:
                self.logger.warning(f"\nNLRP3数据不足，补充{shortage}条示例数据")
                supplement = self.generate_realistic_data(
                    n_active=shortage,
                    n_inactive=0
                )
                all_data.append(supplement)
            
            # 2. 采样负样本
            negative_data = self.sample_negative_compounds_aggressive(self.raw_inactive_needed)
            
            if len(negative_data) > 0:
                all_data.append(negative_data)
            else:
                self.logger.warning("负样本采集失败，使用示例数据")
                neg_supplement = self.generate_realistic_data(
                    n_active=0,
                    n_inactive=self.raw_inactive_needed
                )
                all_data.append(neg_supplement)
        else:
            # ChEMBL完全不可用
            self.logger.warning("ChEMBL不可用，使用完整示例数据")
            example_data = self.generate_realistic_data(
                n_active=self.raw_nlrp3_needed,
                n_inactive=self.raw_inactive_needed
            )
            all_data.append(example_data)
        
        # 合并并保存
        final_df = pd.concat(all_data, ignore_index=True)
        
        output_dir = self.config['paths']['raw_data_dir']
        output_file = self.config['filenames']['raw_data']
        output_path = Path(output_dir) / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(output_path, index=False)
        
        # 最终报告
        log_section(self.logger, "采集完成")
        
        self.logger.info(f"总记录数: {len(final_df)}")
        self.logger.info(f"目标: NLRP3={self.raw_nlrp3_needed}, 非活性={self.raw_inactive_needed}")
        
        if 'data_source' in final_df.columns:
            self.logger.info(f"\n数据来源:")
            for source, count in final_df['data_source'].value_counts().items():
                pct = count / len(final_df) * 100
                self.logger.info(f"  {source}: {count} ({pct:.1f}%)")
        
        self.logger.info(f"\n保存位置: {output_path}")
        
        return str(output_path)


def main():
    """主函数"""
    config = load_data_config()
    
    collector = OptimizedDataCollector(config)
    output_path = collector.run()
    
    print(f"\n{'='*70}")
    print("✅ 数据采集完成")
    print(f"{'='*70}")
    print(f"\n📁 文件: {output_path}")
    print("\n💡 改进说明:")
    print("  ✓ 全面搜索所有NLRP3相关靶点")
    print("  ✓ 包含所有测量类型")
    print("  ✓ 激进的过采样策略")
    print("  ✓ 充分考虑预处理损失")
    print("\n📊 下一步:")
    print("  python experiments/stage0_data/02_preprocess_data.py")


if __name__ == "__main__":
    main()