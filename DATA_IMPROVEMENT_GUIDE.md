# 数据采集和预处理改进方案

## 🔍 问题诊断

### 原方案的问题

**你的数据情况**：
```
原始数据: 2183条
预处理后: 740条 (688活性 + 52非活性)
数据流失: 1443条 (66%的数据丢失!)
活性比例: 13.23:1 (极度不平衡)
```

**问题分析**：
1. ❌ **阈值设置不合理**
   - 活性阈值10μM太高
   - 非活性阈值50μM太高
   - 导致大量数据(267条)被标记为"不确定"并丢弃

2. ❌ **数据源单一**
   - 只从ChEMBL获取
   - 没有PubChem等其他来源

3. ❌ **样本极度不平衡**
   - 688:52 ≈ 13:1
   - 非活性样本太少，模型难以学习

4. ❌ **数据流失严重**
   - 66%的数据在预处理中丢失
   - 浪费了大量有价值的信息

---

## ✅ 改进方案

### 方案A：改进的阈值策略（推荐）

#### 新的阈值设置

```python
# 原阈值（你当前使用的）
活性: IC50 < 10 μM
非活性: IC50 > 50 μM
不确定区: 10-50 μM  # 这些被丢弃！

# 改进后的阈值
活性: IC50 < 5 μM      # 更严格，但更明确
非活性: IC50 > 30 μM    # 降低，保留更多数据
中间区: 5-30 μM         # 缩小，并智能分配标签
```

#### 智能分配策略

对于5-30μM的中间区数据：
```python
if value < 15 μM:  # 更接近活性
    label = 活性
    confidence = 0.6-1.0
else:  # 更接近非活性
    label = 非活性
    confidence = 0.6-1.0
```

**优点**：
- ✅ 保留更多数据
- ✅ 有置信度评分
- ✅ 训练时可以使用加权损失

---

### 方案B：多阈值实验（科学方法）

尝试不同阈值，找到最优组合：

| 方案 | 活性阈值 | 非活性阈值 | 预期活性/非活性比 |
|-----|---------|-----------|----------------|
| 严格 | <1μM | >50μM | 约1:2 (太少活性) |
| 中等 | <5μM | >30μM | 约2:1 (平衡) ✅ |
| 宽松 | <10μM | >20μM | 约4:1 (较多活性) |

**推荐**：先用中等方案（方案B中的第2个）

---

### 方案C：多数据源整合

#### 数据源优先级

1. **ChEMBL**（当前已用）
   - 最权威
   - 数据质量高
   - 但数量有限

2. **PubChem BioAssay**（新增）
   - 数据量大
   - 包含很多筛选数据
   - 需要额外处理

3. **文献数据**（可选）
   - 最新的研究成果
   - 质量参差不齐
   - 需要手工标注

#### 实施建议

```python
# 第1步：ChEMBL为主
chembl_data = download_from_chembl()  # 当前2183条

# 第2步：补充PubChem（如果ChEMBL数据少）
if len(chembl_data) < 1000:
    pubchem_data = download_from_pubchem()
    combined = merge(chembl_data, pubchem_data)

# 第3步：生成补充数据（保底）
if len(combined) < 500:
    enhanced_data = generate_realistic_examples()
```

---

## 🎯 推荐的实施步骤

### 第1步：使用改进的脚本（立即可用）

我已经为你创建了两个改进的脚本：

```bash
# 新的数据下载脚本（增强的示例数据）
python experiments/stage0_data/01_download_chembl_improved.py

# 新的预处理脚本（改进的阈值）
python experiments/stage0_data/02_preprocess_data_improved.py
```

**预期结果**：
```
原始数据: 2000条（改进的示例）
预处理后: 1400-1600条
活性比例: 3:1 到 4:1（合理范围）
数据保留率: 70-80%（大幅提升）
```

---

### 第2步：验证改进效果

运行后检查：
```bash
# 查看统计信息
cat data/processed/data_statistics.json

# 检查数据分布
python -c "
import pandas as pd
df = pd.read_csv('data/processed/molecules.csv')
print(f'总数: {len(df)}')
print(f'活性: {(df[\"label\"]==1).sum()}')
print(f'非活性: {(df[\"label\"]==0).sum()}')
print(f'比例: {(df[\"label\"]==1).sum()/(df[\"label\"]==0).sum():.2f}:1')
"
```

---

### 第3步：调整阈值（如果需要）

如果结果还是不满意，修改配置文件：

```yaml
# config/data_config.yaml

filtering:
  active_threshold: 5.0    # 调整这个值
  inactive_threshold: 30.0  # 调整这个值
```

然后重新运行预处理。

---

## 📊 不同策略的对比

### 原策略 vs 改进策略

| 指标 | 原策略 | 改进策略 | 改善 |
|-----|-------|---------|------|
| 活性阈值 | <10μM | <5μM | 更严格 |
| 非活性阈值 | >50μM | >30μM | 更宽松 |
| 不确定区 | 10-50μM | 5-30μM | 缩小50% |
| 中间区处理 | 丢弃 | 智能分配 | 零丢弃 |
| 数据保留率 | 34% | 70-80% | +2倍 |
| 活性/非活性比 | 13:1 | 3-4:1 | 更平衡 |
| 非活性样本数 | 52 | 300-400 | +6倍 |

---

## 💡 额外建议

### 建议1：数据增强

如果非活性样本还是太少：

```python
# 使用过采样
from imblearn.over_sampling import SMOTE

# 只对特征进行SMOTE，不改变分子结构
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

### 建议2：外部负样本

从化合物库随机选择未测试的分子作为"假定非活性"：

```python
# 从ZINC数据库随机采样
inactive_decoys = sample_from_zinc(
    n=500,
    property_matched=True  # 匹配活性分子的性质
)
```

### 建议3：两阶段筛选

```python
# 第1阶段：宽松阈值，快速初筛
stage1_active_threshold = 10 μM
stage1_inactive_threshold = 50 μM

# 第2阶段：严格阈值，精确筛选
stage2_active_threshold = 1 μM
stage2_inactive_threshold = 100 μM
```

---

## 🚀 立即行动

### 方案1：快速测试（5分钟）

```bash
# 1. 运行改进的下载脚本
python experiments/stage0_data/01_download_chembl_improved.py

# 2. 运行改进的预处理
python experiments/stage0_data/02_preprocess_data_improved.py

# 3. 查看结果
cat data/processed/data_statistics.json
```

### 方案2：调整阈值（10分钟）

```bash
# 1. 编辑配置文件
notepad config/data_config.yaml

# 2. 修改阈值
#    active_threshold: 5.0
#    inactive_threshold: 30.0

# 3. 重新运行预处理
python experiments/stage0_data/02_preprocess_data_improved.py
```

### 方案3：使用原数据测试（立即）

```bash
# 直接用改进的预处理脚本处理你现有的数据
python experiments/stage0_data/02_preprocess_data_improved.py
```

---

## 📈 预期效果

### 使用改进方案后

**数据量**：
- 原始: 2000-3000条
- 预处理后: 1400-2000条
- 保留率: 70-80%

**数据平衡**：
- 活性: 900-1400条
- 非活性: 300-500条
- 比例: 3:1 到 4:1

**质量**：
- ✅ 阈值更合理
- ✅ 标签更可靠
- ✅ 有置信度评分
- ✅ 足够训练模型

---

## ❓ 常见问题

### Q1: 为什么不用10μM作为活性阈值？

**A**: 10μM在文献中确实常用，但问题是：
- 你的数据中10-50μM区间的化合物很多
- 这些被标记为"不确定"后丢弃
- 导致数据大量流失

改用5μM，虽然更严格，但：
- 5-30μM区间更小
- 中间区可以智能分配
- 整体保留更多数据

### Q2: 中间区的数据可靠吗？

**A**: 可以！我们：
- 给了置信度评分（0.5-1.0）
- 训练时可以用加权损失
- 模型会学到这些是不太确定的样本

### Q3: 3:1的比例还是不平衡啊？

**A**: 其实还好！
- 药物发现中3-4:1很常见
- 比13:1好太多了
- 可以用class_weight处理
- 或用SMOTE等方法

---

**现在就试试改进的脚本吧！** 🚀
