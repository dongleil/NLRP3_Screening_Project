"""
安装缺失的依赖包
================
自动检测并安装训练所需的依赖
"""
import subprocess
import sys

def check_and_install(package_name, import_name=None):
    """检查并安装包"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[OK] {package_name} 已安装")
        return True
    except ImportError:
        print(f"[INSTALL] 正在安装 {package_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ])
            print(f"[OK] {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError:
            print(f"[ERROR] {package_name} 安装失败")
            return False

def main():
    """主函数"""
    print("="*70)
    print("检查并安装训练依赖")
    print("="*70)
    print()
    
    # 必需的包
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    # 可选的包（ML模型）
    optional_packages = [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
    ]
    
    # 可选的包（深度学习）
    dl_packages = [
        ("torch", "torch"),
    ]
    
    print("[必需包]")
    all_required_ok = True
    for pkg, imp in required_packages:
        if not check_and_install(pkg, imp):
            all_required_ok = False
    
    print("\n[可选包 - ML模型]")
    for pkg, imp in optional_packages:
        check_and_install(pkg, imp)
    
    print("\n[可选包 - 深度学习]")
    print("PyTorch需要根据你的系统选择安装方式：")
    print("  CPU版本: pip install torch torchvision torchaudio")
    print("  GPU版本: 访问 https://pytorch.org/get-started/locally/")
    
    for pkg, imp in dl_packages:
        try:
            __import__(imp)
            print(f"[OK] {pkg} 已安装")
        except ImportError:
            print(f"[INFO] {pkg} 未安装（DNN训练需要）")
    
    print()
    print("="*70)
    if all_required_ok:
        print("[SUCCESS] 所有必需包已安装")
        print()
        print("你现在可以运行：")
        print("  python experiments/stage1_1d/run_all_1d_training.py")
    else:
        print("[WARNING] 部分必需包安装失败")
        print("请手动安装失败的包")
    print("="*70)

if __name__ == "__main__":
    main()
