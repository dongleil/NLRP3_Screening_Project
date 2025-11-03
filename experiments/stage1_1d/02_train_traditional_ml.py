"""
传统机器学习模型训练脚本
======================
训练多个传统ML模型：
- Random Forest
- XGBoost
- LightGBM
- SVM
- Logistic Regression

特征：ECFP4指纹 (2048位)
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm

# ML库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost训练")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM未安装，将跳过LightGBM训练")

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, load_data_config, log_section


class ECFP4Featurizer:
    """ECFP4指纹特征化器"""
    
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
    
    def transform(self, smiles_list):
        """转换SMILES为ECFP4指纹"""
        fps = []
        failed = 0
        
        for smiles in tqdm(smiles_list, desc="生成ECFP4"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fps.append(np.zeros(self.n_bits))
                failed += 1
                continue
            
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            fps.append(np.array(fp))
        
        if failed > 0:
            print(f"  警告: {failed}个SMILES解析失败")
        
        return np.array(fps)


class TraditionalMLTrainer:
    """传统机器学习训练器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.featurizer = ECFP4Featurizer()
        
        # 输出目录
        self.output_dir = Path('results/stage1_1d/traditional_ml')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """加载数据"""
        log_section(self.logger, "加载数据")
        
        data_dir = Path(self.config['paths']['processed_data_dir'])
        
        train_df = pd.read_csv(data_dir / 'train.csv')
        val_df = pd.read_csv(data_dir / 'val.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        
        self.logger.info(f"训练集: {len(train_df)}")
        self.logger.info(f"验证集: {len(val_df)}")
        self.logger.info(f"测试集: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """准备特征"""
        log_section(self.logger, "特征工程")
        
        self.logger.info("使用ECFP4指纹 (radius=2, nBits=2048)")
        
        # 生成指纹
        X_train = self.featurizer.transform(train_df['smiles_standardized'].values)
        X_val = self.featurizer.transform(val_df['smiles_standardized'].values)
        X_test = self.featurizer.transform(test_df['smiles_standardized'].values)
        
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        self.logger.info(f"\n特征维度:")
        self.logger.info(f"  X_train: {X_train.shape}")
        self.logger.info(f"  X_val: {X_val.shape}")
        self.logger.info(f"  X_test: {X_test.shape}")
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        self.class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        self.logger.info(f"\n类别权重: {self.class_weight_dict}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train):
        """训练Random Forest"""
        self.logger.info("\n训练Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=self.class_weight_dict,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """训练XGBoost"""
        if not XGBOOST_AVAILABLE:
            self.logger.error("XGBoost不可用")
            return None
        
        self.logger.info("\n训练XGBoost...")
        
        # 计算scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        self.models['XGBoost'] = model
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """训练LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            self.logger.error("LightGBM不可用")
            return None
        
        self.logger.info("\n训练LightGBM...")
        
        # 计算scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.log_evaluation(50)]
        )
        
        self.models['LightGBM'] = model
        
        return model
    
    def train_svm(self, X_train, y_train):
        """训练SVM"""
        self.logger.info("\n训练SVM (这可能需要较长时间)...")
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight=self.class_weight_dict,
            probability=True,
            random_state=42,
            verbose=True
        )
        
        model.fit(X_train, y_train)
        self.models['SVM'] = model
        
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """训练Logistic Regression"""
        self.logger.info("\n训练Logistic Regression...")
        
        model = LogisticRegression(
            C=1.0,
            class_weight=self.class_weight_dict,
            random_state=42,
            max_iter=1000,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        
        return model
    
    def evaluate_model(self, model_name, model, X, y, set_name="Test"):
        """评估模型"""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'pr_auc': average_precision_score(y, y_pred_proba)
        }
        
        cm = confusion_matrix(y, y_pred)
        
        self.logger.info(f"\n{model_name} - {set_name}:")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        self.logger.info(f"\n  Confusion Matrix:")
        self.logger.info(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        self.logger.info(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
    
    def save_model(self, model_name, model):
        """保存模型"""
        model_path = self.output_dir / f'{model_name.lower()}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f"  模型已保存: {model_path}")
    
    def save_results(self):
        """保存所有结果"""
        results_path = self.output_dir / 'results_summary.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"\n结果已保存: {results_path}")
        
        # 创建对比表
        self.create_comparison_table()
    
    def create_comparison_table(self):
        """创建模型对比表"""
        table_path = self.output_dir / 'model_comparison.txt'
        
        with open(table_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("模型性能对比 (测试集)\n")
            f.write("="*80 + "\n\n")
            
            # 表头
            f.write(f"{'Model':<20} {'ACC':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'ROC-AUC':<8} {'PR-AUC':<8}\n")
            f.write("-"*80 + "\n")
            
            # 每个模型
            for model_name, results in self.results.items():
                test_metrics = results['test_metrics']
                f.write(
                    f"{model_name:<20} "
                    f"{test_metrics['accuracy']:<8.4f} "
                    f"{test_metrics['precision']:<8.4f} "
                    f"{test_metrics['recall']:<8.4f} "
                    f"{test_metrics['f1']:<8.4f} "
                    f"{test_metrics['roc_auc']:<8.4f} "
                    f"{test_metrics['pr_auc']:<8.4f}\n"
                )
            
            f.write("="*80 + "\n")
        
        self.logger.info(f"对比表已保存: {table_path}")
    
    def run(self):
        """运行完整训练流程"""
        log_section(self.logger, "传统机器学习模型训练")
        
        # 1. 加载数据
        train_df, val_df, test_df = self.load_data()
        
        # 2. 特征工程
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(
            train_df, val_df, test_df
        )
        
        # 3. 训练所有模型
        log_section(self.logger, "模型训练")
        
        models_to_train = [
            ('RandomForest', lambda: self.train_random_forest(X_train, y_train)),
            ('LogisticRegression', lambda: self.train_logistic_regression(X_train, y_train)),
        ]
        
        # 添加XGBoost（如果可用）
        if XGBOOST_AVAILABLE:
            models_to_train.append(
                ('XGBoost', lambda: self.train_xgboost(X_train, y_train, X_val, y_val))
            )
        else:
            self.logger.warning("跳过XGBoost训练（未安装）")
        
        # 添加LightGBM（如果可用）
        if LIGHTGBM_AVAILABLE:
            models_to_train.append(
                ('LightGBM', lambda: self.train_lightgbm(X_train, y_train, X_val, y_val))
            )
        else:
            self.logger.warning("跳过LightGBM训练（未安装）")
        
        # SVM较慢，默认跳过
        # models_to_train.append(('SVM', lambda: self.train_svm(X_train, y_train)))
        
        for model_name, train_func in models_to_train:
            try:
                model = train_func()
                
                # 如果模型训练失败（返回None），跳过
                if model is None:
                    self.logger.warning(f"{model_name} 训练被跳过")
                    continue
                
                # 评估
                val_metrics = self.evaluate_model(
                    model_name, model, X_val, y_val, "Validation"
                )
                test_metrics = self.evaluate_model(
                    model_name, model, X_test, y_test, "Test"
                )
                
                # 保存
                self.save_model(model_name, model)
                
                # 记录结果
                self.results[model_name] = {
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"{model_name} 训练失败: {e}")
                continue
        
        # 4. 保存结果
        self.save_results()
        
        log_section(self.logger, "训练完成")
        
        return self.results


def main():
    """主函数"""
    logger = setup_logger("Traditional_ML_Trainer")
    config = load_data_config()
    
    trainer = TraditionalMLTrainer(config, logger)
    
    try:
        results = trainer.run()
        
        print(f"\n{'='*70}")
        print("[OK] 传统机器学习模型训练完成")
        print(f"{'='*70}")
        print(f"\n训练了 {len(results)} 个模型:")
        for model_name in results.keys():
            print(f"  - {model_name}")
        
        print(f"\n结果保存在: results/stage1_1d/traditional_ml/")
        print(f"\n查看对比:")
        print(f"  cat results/stage1_1d/traditional_ml/model_comparison.txt")
        
        print(f"\n下一步:")
        print(f"  python experiments/stage1_1d/03_train_neural_network.py")
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
