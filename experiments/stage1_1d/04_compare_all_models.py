"""
模型对比与可视化脚本
==================
对比所有训练的模型并生成可视化报告
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, log_section

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, logger):
        self.logger = logger
        self.ml_dir = Path('results/stage1_1d/traditional_ml')
        self.nn_dir = Path('results/stage1_1d/neural_network')
        self.output_dir = Path('results/stage1_1d/comparison')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self):
        """加载所有结果"""
        log_section(self.logger, "加载模型结果")
        
        results = {}
        
        # 加载传统ML结果
        ml_results_path = self.ml_dir / 'results_summary.json'
        if ml_results_path.exists():
            with open(ml_results_path, 'r') as f:
                ml_results = json.load(f)
                results.update(ml_results)
                self.logger.info(f"加载了 {len(ml_results)} 个传统ML模型")
        
        # 加载DNN结果
        nn_results_path = self.nn_dir / 'results.json'
        if nn_results_path.exists():
            with open(nn_results_path, 'r') as f:
                nn_results = json.load(f)
                results['DNN'] = nn_results
                self.logger.info("加载了 DNN 模型")
        
        self.logger.info(f"\n总共加载了 {len(results)} 个模型")
        
        return results
    
    def create_comparison_table(self, results):
        """创建详细对比表"""
        log_section(self.logger, "生成对比表")
        
        # 准备数据
        rows = []
        for model_name, model_results in results.items():
            test_metrics = model_results['test_metrics']
            rows.append({
                'Model': model_name,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1': test_metrics['f1'],
                'ROC-AUC': test_metrics['roc_auc'],
                'PR-AUC': test_metrics['pr_auc']
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('ROC-AUC', ascending=False)
        
        # 保存CSV
        csv_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        
        # 保存格式化表格
        table_path = self.output_dir / 'comparison_table.txt'
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("="*90 + "\n")
            f.write("模型性能对比 (测试集)\n")
            f.write("="*90 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*90 + "\n")
            
            # 最佳模型
            best_model = df.iloc[0]
            f.write(f"\n🏆 最佳模型: {best_model['Model']}\n")
            f.write(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}\n")
            f.write(f"   F1 Score: {best_model['F1']:.4f}\n")
        
        self.logger.info(f"对比表已保存: {table_path}")
        
        return df
    
    def plot_metrics_comparison(self, df):
        """绘制指标对比图"""
        self.logger.info("\n绘制指标对比图...")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 排序
            df_sorted = df.sort_values(metric, ascending=True)
            
            # 绘制条形图
            bars = ax.barh(df_sorted['Model'], df_sorted[metric])
            
            # 颜色
            colors = plt.cm.RdYlGn(df_sorted[metric])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel(metric, fontsize=12)
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (model, value) in enumerate(zip(df_sorted['Model'], df_sorted[metric])):
                ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  保存: {plot_path}")
    
    def plot_roc_auc_comparison(self, df):
        """绘制ROC-AUC排名图"""
        self.logger.info("绘制ROC-AUC排名图...")
        
        plt.figure(figsize=(10, 6))
        
        df_sorted = df.sort_values('ROC-AUC', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
        bars = plt.barh(df_sorted['Model'], df_sorted['ROC-AUC'], color=colors)
        
        plt.xlabel('ROC-AUC Score', fontsize=14, fontweight='bold')
        plt.title('模型ROC-AUC性能对比', fontsize=16, fontweight='bold')
        plt.xlim(0.5, 1.0)
        plt.grid(axis='x', alpha=0.3)
        
        # 添加数值
        for i, (model, value) in enumerate(zip(df_sorted['Model'], df_sorted['ROC-AUC'])):
            plt.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'roc_auc_ranking.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  保存: {plot_path}")
    
    def plot_radar_chart(self, df):
        """绘制雷达图"""
        self.logger.info("绘制雷达图...")
        
        # 选择前5个模型
        top_models = df.nlargest(5, 'ROC-AUC')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, (_, row) in enumerate(top_models.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=12)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Top 5 模型性能雷达图', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'radar_chart.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  保存: {plot_path}")
    
    def plot_heatmap(self, df):
        """绘制性能热力图"""
        self.logger.info("绘制性能热力图...")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
        data = df.set_index('Model')[metrics].T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0.5,
            vmax=1.0,
            cbar_kws={'label': 'Score'},
            linewidths=0.5
        )
        
        plt.title('模型性能热力图', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Metric', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'performance_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  保存: {plot_path}")
    
    def generate_summary_report(self, df, results):
        """生成总结报告"""
        log_section(self.logger, "生成总结报告")
        
        report_path = self.output_dir / 'SUMMARY_REPORT.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# NLRP3 筛选模型训练报告\n\n")
            f.write("## 📊 实验概览\n\n")
            f.write(f"- 训练模型数量: {len(results)}\n")
            f.write(f"- 特征类型: ECFP4指纹 (2048位)\n")
            f.write(f"- 数据集划分: 70% 训练 / 15% 验证 / 15% 测试\n\n")
            
            f.write("## 🏆 模型排名 (按ROC-AUC)\n\n")
            f.write("| 排名 | 模型 | ROC-AUC | F1 Score | Precision | Recall |\n")
            f.write("|------|------|---------|----------|-----------|--------|\n")
            
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                f.write(
                    f"| {idx} | {row['Model']} | {row['ROC-AUC']:.4f} | "
                    f"{row['F1']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} |\n"
                )
            
            f.write("\n## 📈 最佳模型\n\n")
            best = df.iloc[0]
            f.write(f"**{best['Model']}** 在所有模型中表现最佳：\n\n")
            f.write(f"- ROC-AUC: {best['ROC-AUC']:.4f}\n")
            f.write(f"- F1 Score: {best['F1']:.4f}\n")
            f.write(f"- Precision: {best['Precision']:.4f}\n")
            f.write(f"- Recall: {best['Recall']:.4f}\n\n")
            
            f.write("## 📊 可视化\n\n")
            f.write("生成的可视化文件：\n")
            f.write("- `metrics_comparison.png` - 全指标对比\n")
            f.write("- `roc_auc_ranking.png` - ROC-AUC排名\n")
            f.write("- `radar_chart.png` - Top 5模型雷达图\n")
            f.write("- `performance_heatmap.png` - 性能热力图\n\n")
            
            f.write("## 💡 结论\n\n")
            f.write(f"第一阶段（1D特征）训练完成，共训练{len(results)}个模型。\n")
            f.write(f"推荐使用 **{best['Model']}** 进行虚拟筛选。\n\n")
            
            f.write("## 📁 文件位置\n\n")
            f.write("```\n")
            f.write("results/stage1_1d/\n")
            f.write("├── traditional_ml/       # 传统ML模型\n")
            f.write("├── neural_network/       # DNN模型\n")
            f.write("└── comparison/           # 对比结果和可视化\n")
            f.write("```\n")
        
        self.logger.info(f"总结报告已保存: {report_path}")
    
    def run(self):
        """运行完整对比流程"""
        log_section(self.logger, "模型对比与可视化")
        
        # 1. 加载结果
        results = self.load_results()
        
        if len(results) == 0:
            self.logger.error("没有找到任何模型结果！")
            return
        
        # 2. 创建对比表
        df = self.create_comparison_table(results)
        
        # 3. 生成可视化
        log_section(self.logger, "生成可视化")
        self.plot_metrics_comparison(df)
        self.plot_roc_auc_comparison(df)
        self.plot_radar_chart(df)
        self.plot_heatmap(df)
        
        # 4. 生成报告
        self.generate_summary_report(df, results)
        
        log_section(self.logger, "对比完成")
        
        return df


def main():
    """主函数"""
    logger = setup_logger("Model_Comparator")
    
    comparator = ModelComparator(logger)
    
    try:
        df = comparator.run()
        
        print(f"\n{'='*70}")
        print("[OK] 模型对比完成")
        print(f"{'='*70}")
        
        print(f"\n[BEST] 最佳模型:")
        best = df.iloc[0]
        print(f"  {best['Model']}")
        print(f"  ROC-AUC: {best['ROC-AUC']:.4f}")
        print(f"  F1 Score: {best['F1']:.4f}")
        
        print(f"\n📊 查看结果:")
        print(f"  报告: results/stage1_1d/comparison/SUMMARY_REPORT.md")
        print(f"  可视化: results/stage1_1d/comparison/*.png")
        
        print(f"\n[DONE] 第一阶段（1D特征）训练完成！")
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
