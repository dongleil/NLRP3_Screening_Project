# 环境安装指南

## 🎯 三种安装方法

### 方法1：自动安装脚本（最简单）

**双击运行**：
- `install_environment.bat` - 完整安装（10-20分钟）
- `install_minimal.bat` - 最小安装（5分钟）

**如果遇到问题**：
- 右键脚本 → "以管理员身份运行"
- 或在PowerShell中运行：`.\install_environment.bat`

---

### 方法2：手动安装（逐步）

#### 第1步：配置镜像源
```powershell
conda config --set ssl_verify false
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

#### 第2步：安装RDKit
```powershell
conda install -c conda-forge rdkit -y
```

#### 第3步：安装其他包
```powershell
pip install numpy pandas scipy scikit-learn matplotlib xgboost pyyaml tqdm joblib
```

#### 第4步：验证
```powershell
python scripts\check_environment.py
```

---

### 方法3：使用requirements文件

如果网络正常，可以使用requirements文件：

```powershell
# 配置后直接安装
pip install -r requirements.txt
```

---

## 🔧 常见问题

### Q1: SSL错误
**症状**：`SSLEOFError` 或 `SSL: UNEXPECTED_EOF_WHILE_READING`

**解决**：
```powershell
conda config --set ssl_verify false
```

### Q2: 代理错误
**症状**：`ProxyError` 或 `Cannot connect to proxy`

**解决**：
```powershell
pip config unset global.index-url
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: 编码错误
**症状**：`UnicodeDecodeError: 'gbk' codec`

**解决**：不使用requirements.txt，改用脚本安装

### Q4: RDKit安装失败
**解决**：
```powershell
# 尝试方法1
conda install -c conda-forge rdkit -y

# 如果失败，方法2
pip install rdkit

# 如果还失败，方法3（下载whl文件）
# 访问：https://github.com/kuelumbus/rdkit-pypi/releases
# 下载对应版本的.whl文件
pip install rdkit-xxxx.whl
```

---

## 🧪 诊断工具

如果安装遇到问题，运行诊断脚本：

```powershell
.\diagnose_environment.bat
```

这会检查：
- Python和conda版本
- 配置信息
- 已安装的包
- 网络连接

**把诊断结果截图给我，我会帮你分析！**

---

## 📦 最小可运行环境

如果完整安装困难，先装这些核心包：

```powershell
pip install numpy pandas scipy scikit-learn rdkit xgboost pyyaml tqdm joblib
```

有了这些就可以：
- ✅ 运行数据下载和预处理
- ✅ 运行1D实验（ECFP + RF/XGBoost）
- ✅ 基本的可视化

其他包（PyTorch、PyTorch Geometric等）用到时再装！

---

## 🎓 安装优先级

### P0 - 立即安装（核心）
- numpy, pandas, scipy
- scikit-learn
- rdkit
- xgboost
- matplotlib
- pyyaml, tqdm, joblib

### P1 - 尽快安装（重要）
- seaborn, plotly（可视化）
- jupyter（交互式分析）
- imbalanced-learn（数据平衡）
- statsmodels（统计检验）

### P2 - 按需安装（实验时再装）
- pytorch（深度学习实验）
- torch-geometric（2D图神经网络）
- transformers（SMILES Transformer）
- schnetpack（3D实验）

---

## 💡 推荐安装顺序

1. **今天**：运行 `install_minimal.bat` 装核心包
2. **明天**：测试数据处理，确保能跑
3. **第3天**：根据需要装其他包
4. **第4天起**：开始实验

**不要一次装完所有包，容易出错！**

---

## 📞 需要帮助

如果遇到问题：
1. 运行 `diagnose_environment.bat`
2. 截图诊断结果
3. 告诉我具体的错误信息
4. 我会帮你解决

---

## ✅ 验证安装成功

运行以下命令，如果都能成功就OK了：

```powershell
# 测试1：核心包
python -c "import numpy, pandas, sklearn, rdkit; print('✓')"

# 测试2：化学功能
python -c "from rdkit import Chem; mol = Chem.MolFromSmiles('CCO'); print('✓')"

# 测试3：项目检查
python scripts\check_environment.py
```

---

**记住：环境配置是最难的部分，配好后就一帆风顺了！** 💪
