# 数据目录

## 目录结构

```
data/
├── raw/                    # 原始数据
│   └── chembl_nlrp3_raw.csv
├── processed/              # 处理后的数据
│   ├── molecules.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── data_statistics.json
│   └── split_info.json
├── features/              # 预计算的特征
│   ├── ecfp4/            # ECFP4指纹
│   ├── descriptors/      # RDKit描述符
│   ├── graphs/           # 分子图
│   ├── conformers/       # 3D构象
│   ├── topological/      # 拓扑指纹
│   └── pharmacophore_3d/ # 3D药效团
└── external/             # 外部验证集（可选）
```

## 数据生成流程

1. **原始数据**: 运行 `01_download_chembl.py`
2. **处理数据**: 运行 `02_preprocess_data.py`
3. **数据划分**: 运行 `03_split_dataset.py`
4. **特征生成**: 运行 `04_generate_features.py`

## 注意事项

- 原始数据约1000条记录（示例数据）
- 处理后约800-900条
- 划分比例：70% / 15% / 15%
- 特征文件可能较大（几百MB）
