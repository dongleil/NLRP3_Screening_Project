"""
模型深度分析脚本
==============
分析模型的特征重要性、错误案例等
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from rdkit import Chem
from rdkit.Chem import AllChem

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, log_section


class ModelAnalyzer:
    """模型分析器"""
    
    def __init__(self, logger):
        self.logger = logger
        self.output_dir = Path('results/stage1_1d/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_data(self, model_path):
        """加载模型和测试数据"""
        log_section(self.logger, "加载模型和数据")
        
        # 加载模型
        self.logger.info(f"加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 加载测试数据
        test_df = pd.read_csv('data/processed/test.csv')
        
        # 生成指纹
        self.logger.info("生成ECFP4指纹...")
        X_test = []
        for smiles in test_df['smiles_standardized']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                X_test.append(np.array(fp))
            else:
                X_test.append(np.zeros(2048))
        
        X_test = np.array(X_test)
        y_test = test_df['label'].values
        
        return model, X_test, y_test, test_df
    
    def analyze_feature_importance(self, model, model_name='LightGBM'):
        """分析特征重要性"""
        log_section(self.logger, "特征重要性分析")
        
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("模型不支持特征重要性分析")
            return
        
        importances = model.feature_importances_
        
        # 找出最重要的位
        top_n = 20
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        
        self.logger.info(f"\nTop {top_n} 最重要的指纹位:")
        for i, (idx, imp) in enumerate(zip(top_indices, top_importances), 1):
            self.logger.info(f"  {i}. Bit {idx}: {imp:.6f}")
        
        # 可视化
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), top_importances)
        plt.yticks(range(top_n), [f'Bit {i}' for i in top_indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{model_name} - Top {top_n} Important Features', fontsize=14)
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{model_name.lower()}_feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"特征重要性图: {plot_path}")
    
    def analyze_predictions(self, model, X_test, y_test, test_df):
        """分析预测结果"""
        log_section(self.logger, "预测结果分析")
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        self.logger.info("\n混淆矩阵:")
        self.logger.info(f"  真阴性(TN): {cm[0,0]}")
        self.logger.info(f"  假阳性(FP): {cm[0,1]}")
        self.logger.info(f"  假阴性(FN): {cm[1,0]}")
        self.logger.info(f"  真阳性(TP): {cm[1,1]}")
        
        # 分类报告
        self.logger.info("\n详细分类报告:")
        report = classification_report(y_test, y_pred, 
                                       target_names=['Inactive', 'Active'])
        self.logger.info(f"\n{report}")
        
        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Inactive', 'Active'],
                   yticklabels=['Inactive', 'Active'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"\n混淆矩阵图: {cm_path}")
        
        return y_pred, y_pred_proba
    
    def find_misclassified(self, y_test, y_pred, y_pred_proba, test_df):
        """找出错分的样本"""
        log_section(self.logger, "错分样本分析")
        
        # 找出错分样本
        misclassified_mask = y_test != y_pred
        misclassified_df = test_df[misclassified_mask].copy()
        misclassified_df['true_label'] = y_test[misclassified_mask]
        misclassified_df['predicted_label'] = y_pred[misclassified_mask]
        misclassified_df['predicted_proba'] = y_pred_proba[misclassified_mask]
        
        self.logger.info(f"\n错分样本总数: {len(misclassified_df)}")
        
        # 假阳性（预测为活性但实际非活性）
        fp_df = misclassified_df[misclassified_df['true_label'] == 0]
        self.logger.info(f"  假阳性(FP): {len(fp_df)}")
        
        # 假阴性（预测为非活性但实际活性）
        fn_df = misclassified_df[misclassified_df['true_label'] == 1]
        self.logger.info(f"  假阴性(FN): {len(fn_df)}")
        
        # 保存错分样本
        if len(misclassified_df) > 0:
            misclass_path = self.output_dir / 'misclassified_compounds.csv'
            misclassified_df.to_csv(misclass_path, index=False)
            self.logger.info(f"\n错分样本已保存: {misclass_path}")
        
        # 分析高置信度错分
        high_conf_errors = misclassified_df[
            (misclassified_df['predicted_proba'] > 0.7) | 
            (misclassified_df['predicted_proba'] < 0.3)
        ]
        
        if len(high_conf_errors) > 0:
            self.logger.info(f"\n高置信度错分: {len(high_conf_errors)}")
            high_conf_path = self.output_dir / 'high_confidence_errors.csv'
            high_conf_errors.to_csv(high_conf_path, index=False)
            self.logger.info(f"已保存: {high_conf_path}")
    
    def plot_probability_distribution(self, y_test, y_pred_proba):
        """绘制概率分布"""
        log_section(self.logger, "概率分布分析")
        
        plt.figure(figsize=(12, 6))
        
        # 活性和非活性的概率分布
        active_probs = y_pred_proba[y_test == 1]
        inactive_probs = y_pred_proba[y_test == 0]
        
        plt.hist(inactive_probs, bins=50, alpha=0.6, label='True Inactive', color='blue')
        plt.hist(active_probs, bins=50, alpha=0.6, label='True Active', color='red')
        
        plt.xlabel('Predicted Probability (Active)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Probability Distribution by True Label', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        dist_path = self.output_dir / 'probability_distribution.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"概率分布图: {dist_path}")
    
    def generate_analysis_report(self):
        """生成分析报告"""
        log_section(self.logger, "生成分析报告")
        
        report_path = self.output_dir / 'ANALYSIS_REPORT.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LightGBM模型深度分析报告\n\n")
            f.write("## 分析内容\n\n")
            f.write("1. 特征重要性分析\n")
            f.write("2. 预测结果分析\n")
            f.write("3. 错分样本分析\n")
            f.write("4. 概率分布分析\n\n")
            
            f.write("## 生成的文件\n\n")
            f.write("- `lightgbm_feature_importance.png` - 特征重要性\n")
            f.write("- `confusion_matrix.png` - 混淆矩阵\n")
            f.write("- `probability_distribution.png` - 概率分布\n")
            f.write("- `misclassified_compounds.csv` - 错分化合物\n")
            f.write("- `high_confidence_errors.csv` - 高置信度错误\n\n")
            
            f.write("## 使用建议\n\n")
            f.write("1. 查看特征重要性，了解哪些结构特征最重要\n")
            f.write("2. 分析错分样本，改进模型或数据质量\n")
            f.write("3. 检查概率分布，评估模型的置信度校准\n")
        
        self.logger.info(f"分析报告: {report_path}")


def main():
    """主函数"""
    logger = setup_logger("Model_Analyzer")
    
    print("="*70)
    print("模型深度分析")
    print("="*70)
    print()
    
    # 使用LightGBM（最佳模型）
    model_path = 'results/stage1_1d/traditional_ml/lightgbm_model.pkl'
    
    if not Path(model_path).exists():
        print(f"[ERROR] 找不到模型: {model_path}")
        return
    
    analyzer = ModelAnalyzer(logger)
    
    # 加载模型和数据
    model, X_test, y_test, test_df = analyzer.load_model_and_data(model_path)
    
    # 特征重要性
    analyzer.analyze_feature_importance(model, 'LightGBM')
    
    # 预测分析
    y_pred, y_pred_proba = analyzer.analyze_predictions(model, X_test, y_test, test_df)
    
    # 错分样本
    analyzer.find_misclassified(y_test, y_pred, y_pred_proba, test_df)
    
    # 概率分布
    analyzer.plot_probability_distribution(y_test, y_pred_proba)
    
    # 生成报告
    analyzer.generate_analysis_report()
    
    print("\n" + "="*70)
    print("[OK] 分析完成")
    print("="*70)
    print(f"\n结果保存在: results/stage1_1d/analysis/")


if __name__ == "__main__":
    main()
