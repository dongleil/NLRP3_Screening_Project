# NLRP3抑制剂虚拟筛选项目

## 📖 项目简介

本项目旨在开发基于机器学习和深度学习的NLRP3抑制剂虚拟筛选模型，系统对比1D/2D/3D分子表征方法。

## 🎯 研究目标

- 构建高质量的NLRP3抑制剂数据集
- 对比不同维度（1D/2D/3D）的分子表征方法
- 评估机器学习和深度学习模型的性能
- 开发多模态融合模型

## 📊 实验方案

### 实验矩阵
| 维度 | 表征方法 | 模型 | 类型 |
|-----|---------|------|------|
| 1D | ECFP4 | Random Forest | 机器学习 |
| 1D | ECFP4 | XGBoost | 机器学习 |
| 1D | RDKit描述符 | Random Forest | 机器学习 |
| 1D | SMILES | Transformer | 深度学习 |
| 2D | 分子图 | GCN | 深度学习 |
| 2D | 分子图 | GAT | 深度学习 |
| 2D | 分子图 | AttentiveFP | 深度学习 |
| 2D | 拓扑指纹 | XGBoost | 机器学习 |
| 3D | 3D药效团 | Random Forest | 机器学习 |
| 3D | 点云 | SchNet | 深度学习 |
| 融合 | 多模态 | Stacking | 集成学习 |

## 🔧 环境配置

### 1. 创建虚拟环境
```bash
conda create -n nlrp3 python=3.9
conda activate nlrp3
```

### 2. 安装依赖

#### 方式一：使用conda（推荐）
```bash
# 安装PyTorch (CPU版本)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 如果有GPU (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装RDKit
conda install -c conda-forge rdkit

# 安装其他依赖
pip install -r requirements.txt
```

#### 方式二：使用pip
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python scripts/check_environment.py
```

## 📁 项目结构

```
NLRP3_Screening_Project/
├── data/                  # 数据目录
├── src/                   # 源代码
├── experiments/           # 实验脚本
├── results/              # 结果输出
├── notebooks/            # Jupyter notebooks
└── docs/                 # 文档
```

## 🚀 快速开始

### 步骤1: 数据采集
```bash
python experiments/stage0_data/01_download_chembl.py
```

### 步骤2: 数据预处理
```bash
python experiments/stage0_data/02_preprocess_data.py
```

### 步骤3: 数据集划分
```bash
python experiments/stage0_data/03_split_dataset.py
```

### 步骤4: 特征生成
```bash
python experiments/stage0_data/04_generate_features.py --all
```

### 步骤5: 运行实验
```bash
# 运行单个实验
python experiments/stage1_1d/exp1_ecfp_rf.py

# 运行所有1D实验
bash experiments/run_all_1d.sh

# 运行完整pipeline
bash experiments/run_complete_pipeline.sh
```

### 步骤6: 查看结果
```bash
jupyter notebook notebooks/04_visualization.ipynb
```

## 📊 评估指标

- **分类性能**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- **虚拟筛选**: EF1%, EF5%, EF10%, BEDROC
- **统计检验**: 5-fold CV, DeLong test

## 📚 文档

- [安装指南](docs/installation.md)
- [数据准备](docs/data_preparation.md)
- [模型详解](docs/model_details.md)
- [实验流程](docs/experimental_protocol.md)
- [常见问题](docs/troubleshooting.md)

## 🤝 贡献

欢迎提出问题和建议！

## 📄 许可证

MIT License

## 👤 作者

[你的名字]

## 📧 联系方式

[你的邮箱]
