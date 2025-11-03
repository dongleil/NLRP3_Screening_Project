"""
虚拟筛选脚本
============
使用训练好的模型筛选化合物库
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, log_section


class VirtualScreener:
    """虚拟筛选器"""
    
    def __init__(self, model_path, logger):
        self.logger = logger
        self.model = self.load_model(model_path)
        
        # 输出目录
        self.output_dir = Path('results/stage1_1d/virtual_screening')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path):
        """加载模型"""
        self.logger.info(f"加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.logger.info("模型加载成功")
        return model
    
    def smiles_to_ecfp4(self, smiles):
        """转换SMILES为ECFP4指纹"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return np.array(fp)
        except:
            return None
    
    def screen_compounds(self, input_file, smiles_column='smiles', id_column=None):
        """筛选化合物"""
        log_section(self.logger, "开始虚拟筛选")
        
        # 读取化合物库
        self.logger.info(f"读取化合物库: {input_file}")
        df = pd.read_csv(input_file)
        
        self.logger.info(f"化合物总数: {len(df)}")
        
        # 准备结果
        results = []
        failed = 0
        
        # 逐个预测
        self.logger.info("\n正在预测...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_column]
            
            # 获取ID（如果有）
            if id_column and id_column in df.columns:
                compound_id = row[id_column]
            else:
                compound_id = f"Compound_{idx}"
            
            # 生成指纹
            fp = self.smiles_to_ecfp4(smiles)
            if fp is None:
                failed += 1
                continue
            
            # 预测
            fp_2d = fp.reshape(1, -1)
            pred = self.model.predict(fp_2d)[0]
            proba = self.model.predict_proba(fp_2d)[0]
            
            results.append({
                'id': compound_id,
                'smiles': smiles,
                'prediction': 'Active' if pred == 1 else 'Inactive',
                'active_probability': proba[1],
                'inactive_probability': proba[0]
            })
        
        self.logger.info(f"\n预测完成:")
        self.logger.info(f"  成功: {len(results)}")
        self.logger.info(f"  失败: {failed}")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 统计
        n_active = (results_df['prediction'] == 'Active').sum()
        n_inactive = (results_df['prediction'] == 'Inactive').sum()
        
        self.logger.info(f"\n预测结果统计:")
        self.logger.info(f"  预测为活性: {n_active} ({n_active/len(results_df)*100:.1f}%)")
        self.logger.info(f"  预测为非活性: {n_inactive} ({n_inactive/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def rank_and_filter(self, results_df, top_n=100, threshold=0.5):
        """排序和筛选"""
        log_section(self.logger, "排序和筛选")
        
        # 按活性概率排序
        ranked_df = results_df.sort_values('active_probability', ascending=False)
        
        # 筛选高于阈值的
        filtered_df = ranked_df[ranked_df['active_probability'] >= threshold]
        
        self.logger.info(f"活性概率 >= {threshold}: {len(filtered_df)} 个化合物")
        
        # Top N
        top_df = ranked_df.head(top_n)
        
        self.logger.info(f"Top {top_n} 化合物:")
        self.logger.info(f"  平均活性概率: {top_df['active_probability'].mean():.4f}")
        self.logger.info(f"  最高活性概率: {top_df['active_probability'].max():.4f}")
        self.logger.info(f"  最低活性概率: {top_df['active_probability'].min():.4f}")
        
        return ranked_df, filtered_df, top_df
    
    def save_results(self, ranked_df, filtered_df, top_df, top_n=100, threshold=0.5):
        """保存结果"""
        log_section(self.logger, "保存结果")
        
        # 完整排序结果
        ranked_path = self.output_dir / 'all_predictions_ranked.csv'
        ranked_df.to_csv(ranked_path, index=False)
        self.logger.info(f"完整结果: {ranked_path}")
        
        # 筛选后的结果
        if len(filtered_df) > 0:
            filtered_path = self.output_dir / f'filtered_probability_{threshold}.csv'
            filtered_df.to_csv(filtered_path, index=False)
            self.logger.info(f"筛选结果 (>= {threshold}): {filtered_path}")
        
        # Top N
        top_path = self.output_dir / f'top_{top_n}_candidates.csv'
        top_df.to_csv(top_path, index=False)
        self.logger.info(f"Top {top_n}: {top_path}")
        
        # 保存统计信息
        stats = {
            'total_compounds': len(ranked_df),
            'predicted_active': int((ranked_df['prediction'] == 'Active').sum()),
            'predicted_inactive': int((ranked_df['prediction'] == 'Inactive').sum()),
            'filtered_count': len(filtered_df),
            'threshold': threshold,
            'top_n': top_n,
            'mean_probability': float(ranked_df['active_probability'].mean()),
            'top_n_mean_probability': float(top_df['active_probability'].mean())
        }
        
        import json
        stats_path = self.output_dir / 'screening_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"统计信息: {stats_path}")


def main():
    """主函数"""
    logger = setup_logger("Virtual_Screener")
    
    print("="*70)
    print("NLRP3虚拟筛选")
    print("="*70)
    print()
    
    # 使用最佳模型（LightGBM）
    model_path = 'results/stage1_1d/traditional_ml/lightgbm_model.pkl'
    
    if not Path(model_path).exists():
        print(f"[ERROR] 找不到模型: {model_path}")
        print("\n可用的模型:")
        ml_dir = Path('results/stage1_1d/traditional_ml')
        if ml_dir.exists():
            for f in ml_dir.glob('*.pkl'):
                print(f"  - {f}")
        return
    
    # 创建筛选器
    screener = VirtualScreener(model_path, logger)
    
    # 示例：筛选测试集（实际使用时替换为你的化合物库）
    print("\n[示例] 筛选测试集")
    print("实际使用时，请替换为你的化合物库文件")
    print()
    
    input_file = 'data/processed/test.csv'
    
    if not Path(input_file).exists():
        print(f"[ERROR] 找不到输入文件: {input_file}")
        print("\n使用方法:")
        print("  准备一个CSV文件，包含'smiles'列")
        print("  然后运行: screener.screen_compounds('your_file.csv')")
        return
    
    # 筛选
    results_df = screener.screen_compounds(
        input_file,
        smiles_column='smiles_standardized',
        id_column='molecule_chembl_id'
    )
    
    # 排序和筛选
    ranked_df, filtered_df, top_df = screener.rank_and_filter(
        results_df,
        top_n=100,
        threshold=0.7  # 活性概率阈值
    )
    
    # 保存
    screener.save_results(ranked_df, filtered_df, top_df, top_n=100, threshold=0.7)
    
    print("\n" + "="*70)
    print("[OK] 虚拟筛选完成")
    print("="*70)
    print(f"\n结果保存在: results/stage1_1d/virtual_screening/")
    print(f"\n推荐查看:")
    print(f"  - top_100_candidates.csv (最有潜力的100个化合物)")
    print(f"  - filtered_probability_0.7.csv (活性概率>=0.7的化合物)")
    print(f"  - screening_statistics.json (筛选统计)")


if __name__ == "__main__":
    main()
