# 🚀 NLRP3 第一阶段完整训练包

## 📦 包含内容

这个完整包包含：
1. ✅ 修复的数据预处理脚本
2. ✅ 数据集划分脚本
3. ✅ 传统机器学习训练（RF, XGBoost, LightGBM, LogReg）
4. ✅ 深度神经网络训练（DNN）
5. ✅ 模型对比与可视化
6. ✅ 一键运行脚本

---

## 🎯 快速开始（2种方式）

### 方式1：一键运行全部（推荐）⭐

```bash
# 1. 解压
tar -xzf NLRP3_Complete_Training_Package.tar.gz
cd NLRP3_Screening_Project

# 2. 激活环境
conda activate nlrp3

# 3. 一键运行所有训练
python experiments/stage1_1d/run_all_1d_training.py

# 完成！等待15-30分钟
```

**这一条命令会自动完成：**
- ✅ 数据集划分（70/15/15）
- ✅ 训练4个传统ML模型
- ✅ 训练DNN模型
- ✅ 生成对比报告和可视化

---

### 方式2：逐步运行（精细控制）

```bash
# 激活环境
conda activate nlrp3

# 步骤1: 划分数据集
python experiments/stage1_1d/01_split_dataset.py

# 步骤2: 训练传统ML (约10分钟)
python experiments/stage1_1d/02_train_traditional_ml.py

# 步骤3: 训练DNN (约5-10分钟)
python experiments/stage1_1d/03_train_neural_network.py

# 步骤4: 对比和可视化
python experiments/stage1_1d/04_compare_all_models.py
```

---

## 📊 训练内容详解

### 1. 数据集划分 (01_split_dataset.py)

```python
功能：
- 分层划分：训练70% / 验证15% / 测试15%
- 保证各集比例一致
- 保存到data/processed/

输出：
- train.csv (训练集)
- val.csv (验证集)  
- test.csv (测试集)
- split_info.json (划分信息)

耗时：<1分钟
```

---

### 2. 传统机器学习 (02_train_traditional_ml.py)

```python
训练模型：
✓ Random Forest (200棵树，深度20)
✓ XGBoost (200轮，学习率0.1)
✓ LightGBM (200轮，叶子31)
✓ Logistic Regression (L2正则化)

特征：
- ECFP4指纹 (2048位)
- 类别权重平衡

输出：
- results/stage1_1d/traditional_ml/
  ├── randomforest_model.pkl
  ├── xgboost_model.pkl
  ├── lightgbm_model.pkl
  ├── logisticregression_model.pkl
  ├── results_summary.json
  └── model_comparison.txt

耗时：10-15分钟
```

---

### 3. 深度神经网络 (03_train_neural_network.py)

```python
模型架构：
输入: 2048 (ECFP4)
  ↓
隐藏层1: 512 (ReLU + BatchNorm + Dropout 0.3)
  ↓
隐藏层2: 256 (ReLU + BatchNorm + Dropout 0.3)
  ↓
隐藏层3: 128 (ReLU + BatchNorm + Dropout 0.3)
  ↓
输出: 2 (活性/非活性)

训练设置：
- Epochs: 50
- Batch size: 64
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- 加权采样处理不平衡

输出：
- results/stage1_1d/neural_network/
  ├── dnn_model.pt
  └── results.json

耗时：5-10分钟 (CPU) / 2-3分钟 (GPU)
```

---

### 4. 模型对比 (04_compare_all_models.py)

```python
生成内容：
✓ 性能对比表 (CSV + TXT)
✓ 6种指标对比图
✓ ROC-AUC排名图
✓ Top 5模型雷达图
✓ 性能热力图
✓ 完整的Markdown报告

输出：
- results/stage1_1d/comparison/
  ├── SUMMARY_REPORT.md       (总结报告)
  ├── comparison_table.csv    (对比表)
  ├── metrics_comparison.png  (指标对比)
  ├── roc_auc_ranking.png     (ROC排名)
  ├── radar_chart.png         (雷达图)
  └── performance_heatmap.png (热力图)

耗时：<1分钟
```

---

## 📁 完整目录结构

```
NLRP3_Screening_Project/
├── data/
│   ├── raw/
│   │   └── chembl_nlrp3_raw.csv          (原始数据)
│   └── processed/
│       ├── molecules.csv                  (处理后数据)
│       ├── train.csv                      (训练集)
│       ├── val.csv                        (验证集)
│       ├── test.csv                       (测试集)
│       └── split_info.json                (划分信息)
│
├── experiments/
│   └── stage1_1d/
│       ├── 01_split_dataset.py           (划分脚本)
│       ├── 02_train_traditional_ml.py    (传统ML)
│       ├── 03_train_neural_network.py    (DNN)
│       ├── 04_compare_all_models.py      (对比)
│       └── run_all_1d_training.py        (一键运行)
│
└── results/
    └── stage1_1d/
        ├── traditional_ml/               (传统ML结果)
        ├── neural_network/               (DNN结果)
        └── comparison/                   (对比和可视化)
```

---

## ⏱️ 时间估算

| 步骤 | CPU | GPU |
|-----|-----|-----|
| 数据划分 | <1分钟 | <1分钟 |
| 传统ML | 10-15分钟 | 10-15分钟 |
| DNN训练 | 5-10分钟 | 2-3分钟 |
| 对比可视化 | <1分钟 | <1分钟 |
| **总计** | **15-30分钟** | **15-20分钟** |

---

## 📊 预期输出示例

### 模型性能表

```
Model                 ACC      Prec     Recall   F1       ROC-AUC  PR-AUC
--------------------------------------------------------------------------------
XGBoost              0.8923   0.9245   0.8801   0.9018   0.9456   0.9234
LightGBM             0.8897   0.9198   0.8789   0.8989   0.9423   0.9201
RandomForest         0.8756   0.9023   0.8645   0.8830   0.9289   0.9067
DNN                  0.8701   0.8967   0.8598   0.8779   0.9245   0.9012
LogisticRegression   0.8534   0.8712   0.8423   0.8565   0.9012   0.8756
```

### 最佳模型

```
🏆 最佳模型: XGBoost
   ROC-AUC: 0.9456
   F1 Score: 0.9018
   Precision: 0.9245
   Recall: 0.8801
```

---

## 🔧 如果遇到问题

### 问题1: 没有数据文件

```bash
# 确保先运行了数据预处理
python experiments/stage0_data/02_preprocess_strict.py
```

### 问题2: 内存不足

```python
# 修改batch_size (DNN训练)
# 在03_train_neural_network.py中:
batch_size = 32  # 改小一点
```

### 问题3: PyTorch未安装

```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题4: XGBoost/LightGBM错误

```bash
pip install xgboost lightgbm --upgrade
```

---

## 💡 训练技巧

### 1. 调整类别权重

```python
# 如果觉得模型偏向某一类，调整权重
# 在02_train_traditional_ml.py中修改：

# 当前是自动计算
class_weights = compute_class_weight('balanced', ...)

# 也可以手动设置
class_weight_dict = {0: 1.5, 1: 0.5}  # 增加非活性权重
```

### 2. 调整DNN架构

```python
# 在03_train_neural_network.py中修改：

# 更深的网络
hidden_dims = [1024, 512, 256, 128]

# 更多dropout
dropout = 0.5

# 更多epochs
epochs = 100
```

### 3. 早停

```python
# DNN训练时添加早停
# 修改train_model函数，添加：

patience = 10
no_improve = 0
for epoch in range(epochs):
    # ... 训练代码 ...
    if val_auc < best_val_auc:
        no_improve += 1
        if no_improve >= patience:
            break
    else:
        no_improve = 0
```

---

## 📈 查看结果

### 1. 查看总结报告

```bash
cat results/stage1_1d/comparison/SUMMARY_REPORT.md
```

### 2. 查看对比表

```bash
cat results/stage1_1d/comparison/comparison_table.txt
```

### 3. 查看可视化

```bash
# 在文件管理器中打开
results/stage1_1d/comparison/*.png
```

### 4. 加载模型进行预测

```python
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 加载最佳模型
with open('results/stage1_1d/traditional_ml/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 预测新分子
smiles = "c1ccc(cc1)S(=O)(=O)N"  # 你的SMILES
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
fp_array = np.array(fp).reshape(1, -1)

# 预测
pred = model.predict(fp_array)[0]
proba = model.predict_proba(fp_array)[0]

print(f"预测: {'活性' if pred == 1 else '非活性'}")
print(f"概率: {proba[1]:.4f}")
```

---

## ✅ 检查清单

运行前确保：
- [x] 已运行数据预处理
- [x] 环境已激活 (conda activate nlrp3)
- [x] 已安装所有依赖
- [x] 有足够磁盘空间 (>1GB)

运行后检查：
- [x] results/stage1_1d/ 目录存在
- [x] 至少有4-5个模型文件
- [x] comparison/ 目录有报告和图片
- [x] 所有图片都能正常打开

---

## 🎯 下一步

训练完成后，你可以：

1. **分析结果**
   - 阅读 SUMMARY_REPORT.md
   - 查看可视化图表
   - 比较不同模型性能

2. **选择最佳模型**
   - 通常XGBoost或LightGBM表现最好
   - 根据ROC-AUC选择

3. **虚拟筛选**
   - 使用最佳模型筛选化合物库
   - 预测新化合物的活性

4. **继续改进**
   - 训练2D图神经网络
   - 训练3D模型
   - 集成多个模型

---

## 🌟 核心优势

✅ **一键运行** - 单条命令完成所有训练  
✅ **完整流程** - 从数据到结果全覆盖  
✅ **多个模型** - 5个不同算法对比  
✅ **自动可视化** - 生成专业图表  
✅ **详细报告** - Markdown格式易读  
✅ **可复现** - 固定随机种子  
✅ **生产就绪** - 可直接用于筛选  

---

## 📞 需要帮助？

查看日志输出或：
- 检查 results/ 目录
- 阅读错误信息
- 查看每个脚本的注释

---

**准备好了吗？现在就开始训练吧！** 🚀

```bash
python experiments/stage1_1d/run_all_1d_training.py
```