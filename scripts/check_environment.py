"""
环境检查脚本 - 验证所有依赖是否正确安装
"""
import sys
import importlib
from typing import List, Tuple


def check_python_version() -> bool:
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"  当前版本: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("  ✓ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("  ✗ Python版本过低，需要 >= 3.8")
        return False


def check_package(package_name: str, import_name: str = None) -> bool:
    """
    检查包是否安装
    
    Args:
        package_name: 包名（用于显示）
        import_name: 导入名（如果与包名不同）
    
    Returns:
        是否安装成功
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name:25s} (version: {version})")
        return True
    except ImportError:
        print(f"  ✗ {package_name:25s} (未安装)")
        return False


def check_all_dependencies() -> Tuple[List[str], List[str]]:
    """
    检查所有依赖
    
    Returns:
        (成功列表, 失败列表)
    """
    print("\n" + "="*60)
    print("检查核心依赖...")
    print("="*60)
    
    # 核心依赖列表: (显示名, 导入名)
    core_deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("RDKit", "rdkit"),
    ]
    
    success = []
    failed = []
    
    for display_name, import_name in core_deps:
        if check_package(display_name, import_name):
            success.append(display_name)
        else:
            failed.append(display_name)
    
    # 机器学习依赖
    print("\n检查机器学习库...")
    ml_deps = [
        ("XGBoost", "xgboost"),
        ("imbalanced-learn", "imblearn"),
    ]
    
    for display_name, import_name in ml_deps:
        if check_package(display_name, import_name):
            success.append(display_name)
        else:
            failed.append(display_name)
    
    # 深度学习依赖
    print("\n检查深度学习库...")
    dl_deps = [
        ("PyTorch", "torch"),
        ("PyTorch Geometric", "torch_geometric"),
        ("transformers", "transformers"),
    ]
    
    for display_name, import_name in dl_deps:
        if check_package(display_name, import_name):
            success.append(display_name)
        else:
            failed.append(display_name)
    
    # 可视化依赖
    print("\n检查可视化库...")
    viz_deps = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
    ]
    
    for display_name, import_name in viz_deps:
        if check_package(display_name, import_name):
            success.append(display_name)
        else:
            failed.append(display_name)
    
    # 其他依赖
    print("\n检查其他工具...")
    other_deps = [
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
        ("joblib", "joblib"),
    ]
    
    for display_name, import_name in other_deps:
        if check_package(display_name, import_name):
            success.append(display_name)
        else:
            failed.append(display_name)
    
    return success, failed


def check_cuda() -> bool:
    """检查CUDA是否可用"""
    print("\n" + "="*60)
    print("检查GPU支持...")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA可用")
            print(f"    CUDA版本: {torch.version.cuda}")
            print(f"    GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ⚠ CUDA不可用 (将使用CPU)")
            print("    提示: 深度学习实验可能会比较慢")
            return False
    except ImportError:
        print("  ✗ PyTorch未安装，无法检查CUDA")
        return False


def test_rdkit() -> bool:
    """测试RDKit基本功能"""
    print("\n" + "="*60)
    print("测试RDKit功能...")
    print("="*60)
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        
        # 测试SMILES解析
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  ✗ SMILES解析失败")
            return False
        print(f"  ✓ SMILES解析成功: {smiles}")
        
        # 测试描述符计算
        mw = Descriptors.MolWt(mol)
        print(f"  ✓ 描述符计算成功: MW = {mw:.2f}")
        
        # 测试指纹生成
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        print(f"  ✓ 指纹生成成功: {len(fp)} bits")
        
        return True
    except Exception as e:
        print(f"  ✗ RDKit测试失败: {e}")
        return False


def test_torch_geometric() -> bool:
    """测试PyTorch Geometric"""
    print("\n" + "="*60)
    print("测试PyTorch Geometric...")
    print("="*60)
    
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        
        # 创建简单的图
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        x = torch.tensor([[1], [2]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        print(f"  ✓ 图数据创建成功")
        print(f"    节点数: {data.num_nodes}")
        print(f"    边数: {data.num_edges}")
        
        return True
    except Exception as e:
        print(f"  ✗ PyTorch Geometric测试失败: {e}")
        return False


def print_summary(success: List[str], failed: List[str]):
    """打印检查总结"""
    print("\n" + "="*60)
    print("环境检查总结")
    print("="*60)
    
    print(f"\n成功安装的包 ({len(success)}):")
    for pkg in success:
        print(f"  ✓ {pkg}")
    
    if failed:
        print(f"\n缺少的包 ({len(failed)}):")
        for pkg in failed:
            print(f"  ✗ {pkg}")
        print("\n请运行以下命令安装缺少的包:")
        print("  pip install -r requirements.txt")
    else:
        print("\n✓ 所有依赖包已正确安装！")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    print("="*60)
    print("NLRP3筛选项目 - 环境检查")
    print("="*60)
    
    # 检查Python版本
    if not check_python_version():
        print("\n请升级Python到3.8或更高版本")
        return
    
    # 检查所有依赖
    success, failed = check_all_dependencies()
    
    # 检查CUDA
    check_cuda()
    
    # 测试RDKit
    test_rdkit()
    
    # 测试PyTorch Geometric（如果已安装）
    if "PyTorch Geometric" in success:
        test_torch_geometric()
    
    # 打印总结
    print_summary(success, failed)
    
    # 返回状态
    if not failed:
        print("\n🎉 环境配置完成，可以开始实验！")
        return 0
    else:
        print("\n⚠️  请先安装缺少的依赖包")
        return 1


if __name__ == "__main__":
    sys.exit(main())
