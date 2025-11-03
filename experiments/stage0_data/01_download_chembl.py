"""
ChEMBL数据下载脚本
从ChEMBL数据库下载NLRP3抑制剂数据
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import time

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, load_data_config, log_section


class ChEMBLDownloader:
    """ChEMBL数据下载器"""
    
    def __init__(self, config: dict):
        """
        初始化下载器
        
        Args:
            config: 数据配置字典
        """
        self.config = config
        self.logger = setup_logger("ChEMBL_Downloader")
        
        # 尝试导入ChEMBL客户端
        try:
            from chembl_webresource_client.new_client import new_client
            self.client_available = True
            self.activity = new_client.activity
            self.target = new_client.target
            self.molecule = new_client.molecule
        except ImportError:
            self.logger.warning("ChEMBL客户端未安装，将使用示例数据")
            self.client_available = False
    
    def search_target(self, target_name: str) -> Optional[List[Dict]]:
        """
        搜索目标蛋白
        
        Args:
            target_name: 目标蛋白名称
            
        Returns:
            目标列表
        """
        if not self.client_available:
            return None
        
        self.logger.info(f"搜索目标: {target_name}")
        
        try:
            targets = self.target.filter(
                target_synonym__icontains=target_name
            ).only([
                'target_chembl_id',
                'pref_name',
                'target_type',
                'organism'
            ])
            
            target_list = list(targets)
            self.logger.info(f"找到 {len(target_list)} 个匹配的目标")
            
            for i, t in enumerate(target_list[:5]):  # 只显示前5个
                self.logger.info(f"  [{i+1}] {t['target_chembl_id']}: {t['pref_name']}")
            
            return target_list
        except Exception as e:
            self.logger.error(f"搜索目标失败: {e}")
            return None
    
    def download_activities(
        self,
        target_chembl_id: str,
        activity_types: List[str],
        min_confidence: int = 7
    ) -> pd.DataFrame:
        """
        下载活性数据
        
        Args:
            target_chembl_id: 目标ChEMBL ID
            activity_types: 活性类型列表
            min_confidence: 最小置信度分数
            
        Returns:
            活性数据DataFrame
        """
        if not self.client_available:
            return self._generate_example_data()
        
        self.logger.info(f"下载活性数据: {target_chembl_id}")
        self.logger.info(f"活性类型: {', '.join(activity_types)}")
        
        all_activities = []
        
        for activity_type in activity_types:
            try:
                self.logger.info(f"  查询 {activity_type} 数据...")
                
                activities = self.activity.filter(
                    target_chembl_id=target_chembl_id,
                    standard_type=activity_type,
                    pchembl_value__isnull=False
                ).only([
                    'molecule_chembl_id',
                    'canonical_smiles',
                    'standard_type',
                    'standard_value',
                    'standard_units',
                    'pchembl_value',
                    'assay_chembl_id',
                    'assay_description',
                    'confidence_score'
                ])
                
                activity_list = list(activities)
                self.logger.info(f"    找到 {len(activity_list)} 条记录")
                
                # 过滤置信度
                filtered = [
                    a for a in activity_list
                    if a.get('confidence_score', 0) >= min_confidence
                ]
                self.logger.info(f"    过滤后剩余 {len(filtered)} 条 (confidence >= {min_confidence})")
                
                all_activities.extend(filtered)
                
                # 避免请求过快
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"  查询 {activity_type} 失败: {e}")
                continue
        
        # 转换为DataFrame
        if not all_activities:
            self.logger.warning("未找到符合条件的数据，使用示例数据")
            return self._generate_example_data()
        
        df = pd.DataFrame(all_activities)
        self.logger.info(f"\n总计下载 {len(df)} 条活性数据")
        
        return df
    
    def _generate_example_data(self) -> pd.DataFrame:
        """
        生成示例数据（当无法连接ChEMBL时使用）
        
        Returns:
            示例数据DataFrame
        """
        self.logger.info("生成示例数据...")
        
        # 创建示例SMILES（一些已知的NLRP3抑制剂和随机分子）
        np.random.seed(42)
        
        example_data = []
        
        # 活性化合物示例
        active_smiles = [
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # 布洛芬结构（示例）
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因结构（示例）
            "CC(=O)Oc1ccccc1C(=O)O",  # 阿司匹林
            "COc1ccc2nc(sc2c1)S(=O)(=O)N",  # 示例化合物
            "Cc1ccc(cc1)S(=O)(=O)N",  # 示例化合物
        ] * 80  # 重复以获得更多样本
        
        for i, smiles in enumerate(active_smiles[:400]):
            ic50 = np.random.lognormal(0, 1.5) * 1000  # nM
            example_data.append({
                'molecule_chembl_id': f'CHEMBL{1000000+i}',
                'canonical_smiles': smiles,
                'standard_type': 'IC50',
                'standard_value': ic50,
                'standard_units': 'nM',
                'pchembl_value': -np.log10(ic50 / 1e9),
                'assay_chembl_id': f'CHEMBL{2000000+i%10}',
                'assay_description': 'NLRP3 inhibition assay',
                'confidence_score': 9
            })
        
        # 非活性化合物示例
        inactive_smiles = [
            "CCCCCCCCCCCC",  # 简单烷烃
            "c1ccccc1",  # 苯
            "CC(C)C",  # 异丁烷
            "CCO",  # 乙醇
            "CC(C)O",  # 异丙醇
        ] * 120
        
        for i, smiles in enumerate(inactive_smiles[:600]):
            ic50 = np.random.uniform(50000, 200000)  # nM
            example_data.append({
                'molecule_chembl_id': f'CHEMBL{3000000+i}',
                'canonical_smiles': smiles,
                'standard_type': 'IC50',
                'standard_value': ic50,
                'standard_units': 'nM',
                'pchembl_value': -np.log10(ic50 / 1e9),
                'assay_chembl_id': f'CHEMBL{2000000+i%10}',
                'assay_description': 'NLRP3 inhibition assay',
                'confidence_score': 8
            })
        
        df = pd.DataFrame(example_data)
        self.logger.info(f"生成了 {len(df)} 条示例数据")
        self.logger.warning("注意: 这是示例数据，仅用于测试流程")
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, output_path: str):
        """
        保存原始数据
        
        Args:
            df: 数据DataFrame
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"数据已保存到: {output_path}")
    
    def run(self) -> str:
        """
        运行完整的下载流程
        
        Returns:
            输出文件路径
        """
        log_section(self.logger, "ChEMBL数据下载")
        
        # 获取配置
        target_name = self.config['chembl']['target_name']
        target_id = self.config['chembl']['target_chembl_id']
        activity_types = self.config['chembl']['activity_types']
        min_confidence = self.config['chembl']['min_confidence_score']
        
        # 如果没有指定target_id，先搜索
        if target_id is None and self.client_available:
            self.logger.info("未指定target_chembl_id，开始搜索...")
            targets = self.search_target(target_name)
            
            if targets and len(targets) > 0:
                # 使用第一个匹配的目标
                target_id = targets[0]['target_chembl_id']
                self.logger.info(f"使用目标: {target_id}")
            else:
                self.logger.warning("未找到匹配的目标，使用示例数据")
        
        # 下载活性数据
        if target_id and self.client_available:
            df = self.download_activities(target_id, activity_types, min_confidence)
        else:
            df = self._generate_example_data()
        
        # 显示数据统计
        self.logger.info(f"\n数据统计:")
        self.logger.info(f"  总记录数: {len(df)}")
        self.logger.info(f"  唯一分子数: {df['molecule_chembl_id'].nunique()}")
        self.logger.info(f"  活性类型分布:")
        for activity_type, count in df['standard_type'].value_counts().items():
            self.logger.info(f"    {activity_type}: {count}")
        
        # 保存数据
        output_dir = self.config['paths']['raw_data_dir']
        output_file = self.config['filenames']['raw_data']
        output_path = Path(output_dir) / output_file
        
        self.save_raw_data(df, output_path)
        
        log_section(self.logger, "数据下载完成")
        
        return str(output_path)


def main():
    """主函数"""
    # 加载配置
    config = load_data_config()
    
    # 创建下载器
    downloader = ChEMBLDownloader(config)
    
    # 运行下载
    output_path = downloader.run()
    
    print(f"\n✓ 数据已保存到: {output_path}")
    print("\n下一步: 运行数据预处理")
    print("  python experiments/stage0_data/02_preprocess_data.py")


if __name__ == "__main__":
    main()
