# 结果目录

## 目录结构

```
results/
├── stage1_1d/              # 1D实验结果
│   ├── exp1_ecfp_rf/
│   ├── exp2_ecfp_xgb/
│   ├── exp3_desc_rf/
│   └── exp4_smiles_transformer/
├── stage2_2d/              # 2D实验结果
│   ├── exp5_graph_gcn/
│   ├── exp6_graph_gat/
│   ├── exp7_graph_attentivefp/
│   └── exp8_topo_xgb/
├── stage3_3d/              # 3D实验结果
│   ├── exp9_pharm3d_rf/
│   └── exp10_pointcloud_schnet/
├── stage4_fusion/          # 融合实验结果
│   └── exp11_stacking/
└── final_comparison/       # 最终对比分析
    ├── all_metrics_comparison.csv
    ├── statistical_tests.csv
    ├── performance_heatmap.png
    └── final_report.pdf
```

## 每个实验结果包含

- `metrics.json` - 评估指标
- `confusion_matrix.png` - 混淆矩阵
- `roc_curve.png` - ROC曲线
- `pr_curve.png` - PR曲线
- `feature_importance.png` - 特征重要性（ML模型）
- `attention_weights.png` - 注意力权重（GNN模型）
- `model.pkl` 或 `model.pt` - 训练好的模型
- `predictions.csv` - 预测结果
- `training_log.txt` - 训练日志

## 结果文件说明

实验运行后会自动生成所有结果文件到对应目录。
