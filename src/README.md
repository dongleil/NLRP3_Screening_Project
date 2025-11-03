# 源代码模块

## 模块结构

```
src/
├── utils/              # 工具模块 ✅ 已完成
│   ├── config_loader.py
│   ├── logger.py
│   └── chem_utils.py
├── data/              # 数据处理 ⏳ 第二批
│   ├── collector.py
│   ├── preprocessor.py
│   └── splitter.py
├── features/          # 特征工程 ⏳ 第二批
│   ├── fingerprints.py
│   ├── descriptors.py
│   ├── graph_builder.py
│   └── ...
├── models/            # 模型模块
│   ├── ml/           # 机器学习 ⏳ 第二/三批
│   ├── dl/           # 深度学习 ⏳ 第四批
│   └── fusion/       # 融合模型 ⏳ 第五批
├── training/          # 训练模块 ⏳ 第二/三批
│   ├── ml_trainer.py
│   └── dl_trainer.py
└── evaluation/        # 评估模块 ⏳ 第三批
    ├── metrics.py
    ├── evaluator.py
    └── visualizer.py
```

## 已完成模块（✅）

### utils - 工具模块
- **config_loader.py**: 配置文件加载
- **logger.py**: 日志系统
- **chem_utils.py**: 化学工具
  - MoleculeProcessor: 分子处理和标准化
  - MoleculeValidator: 分子验证
  - 描述符计算等

## 待实现模块（⏳）

其他模块将在后续批次陆续提供。

## 使用方法

```python
# 导入工具模块
from src.utils import setup_logger, MoleculeProcessor, load_data_config

# 创建logger
logger = setup_logger("MyExperiment")

# 加载配置
config = load_data_config()

# 处理分子
processor = MoleculeProcessor()
smiles, mol = processor.process_smiles("CCO")
```
