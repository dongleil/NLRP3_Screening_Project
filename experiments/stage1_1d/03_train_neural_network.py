"""
神经网络模型训练脚本
==================
训练深度神经网络（DNN）用于分子活性预测

特征：ECFP4指纹 (2048位)
架构：多层前馈神经网络
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from tqdm import tqdm

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, load_data_config, log_section


class MoleculeDataset(Dataset):
    """分子数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DNNClassifier(nn.Module):
    """深度神经网络分类器"""
    
    def __init__(self, input_dim=2048, hidden_dims=[512, 256, 128], dropout=0.3):
        super(DNNClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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


class DNNTrainer:
    """DNN训练器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.featurizer = ECFP4Featurizer()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 输出目录
        self.output_dir = Path('results/stage1_1d/neural_network')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
        """创建数据加载器"""
        train_dataset = MoleculeDataset(X_train, y_train)
        val_dataset = MoleculeDataset(X_val, y_val)
        test_dataset = MoleculeDataset(X_test, y_test)
        
        # 计算类别权重
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        
        sample_weights = class_weights[y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, model, data_loader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'roc_auc': roc_auc_score(all_labels, all_probs),
            'pr_auc': average_precision_score(all_labels, all_probs)
        }
        
        return metrics, all_preds, all_probs
    
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001):
        """训练模型"""
        log_section(self.logger, "训练DNN")
        
        # 创建模型
        model = DNNClassifier(
            input_dim=2048,
            hidden_dims=[512, 256, 128],
            dropout=0.3
        ).to(self.device)
        
        self.logger.info(f"模型架构:")
        self.logger.info(f"  输入: 2048")
        self.logger.info(f"  隐藏层: [512, 256, 128]")
        self.logger.info(f"  输出: 2")
        self.logger.info(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 训练循环
        best_val_auc = 0
        best_model_state = None
        
        self.logger.info(f"\n开始训练 ({epochs} epochs)...")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证
            val_metrics, _, _ = self.evaluate(model, val_loader)
            
            # 学习率调整
            scheduler.step(val_metrics['roc_auc'])
            
            # 保存最佳模型
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_model_state = model.state_dict().copy()
            
            # 日志
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss:.4f} - "
                    f"Val AUC: {val_metrics['roc_auc']:.4f} - "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        self.logger.info(f"\n最佳验证AUC: {best_val_auc:.4f}")
        
        return model
    
    def save_model(self, model):
        """保存模型"""
        model_path = self.output_dir / 'dnn_model.pt'
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"模型已保存: {model_path}")
    
    def run(self):
        """运行完整训练流程"""
        log_section(self.logger, "DNN模型训练")
        
        # 1. 加载数据
        train_df, val_df, test_df = self.load_data()
        
        # 2. 特征工程
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(
            train_df, val_df, test_df
        )
        
        # 3. 创建数据加载器
        train_loader, val_loader, test_loader = self.create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # 4. 训练模型
        model = self.train_model(train_loader, val_loader, epochs=50)
        
        # 5. 评估
        log_section(self.logger, "最终评估")
        
        val_metrics, _, _ = self.evaluate(model, val_loader)
        test_metrics, test_preds, test_probs = self.evaluate(model, test_loader)
        
        self.logger.info("\n验证集:")
        for metric, value in val_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        self.logger.info("\n测试集:")
        for metric, value in test_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # 6. 保存
        self.save_model(model)
        
        results = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log_section(self.logger, "训练完成")
        
        return results


def main():
    """主函数"""
    logger = setup_logger("DNN_Trainer")
    config = load_data_config()
    
    trainer = DNNTrainer(config, logger)
    
    try:
        results = trainer.run()
        
        print(f"\n{'='*70}")
        print("[OK] DNN模型训练完成")
        print(f"{'='*70}")
        print(f"\n测试集性能:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\n结果保存在: results/stage1_1d/neural_network/")
        
        print(f"\n下一步:")
        print(f"  python experiments/stage1_1d/04_compare_all_models.py")
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
