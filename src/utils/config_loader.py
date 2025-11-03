"""
配置文件加载工具
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {config_dir}")
    
    def load_yaml(self, config_name: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_name: 配置文件名（不含.yaml后缀）
            
        Returns:
            配置字典
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有配置文件
        
        Returns:
            包含所有配置的字典
        """
        configs = {}
        config_files = ['data_config', 'model_config', 'experiment_config']
        
        for config_name in config_files:
            try:
                configs[config_name] = self.load_yaml(config_name)
            except FileNotFoundError:
                print(f"警告: 配置文件 {config_name}.yaml 不存在")
        
        return configs
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.load_yaml('data_config')
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.load_yaml('model_config')
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """获取实验配置"""
        return self.load_yaml('experiment_config')


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    current_file = Path(__file__).resolve()
    # 从 src/utils/config_loader.py 向上找到项目根目录
    project_root = current_file.parent.parent.parent
    return project_root


def get_config_path(config_name: str) -> Path:
    """
    获取配置文件的完整路径
    
    Args:
        config_name: 配置文件名
        
    Returns:
        配置文件完整路径
    """
    project_root = get_project_root()
    config_path = project_root / "config" / f"{config_name}.yaml"
    return config_path


# 方便使用的全局函数
def load_data_config() -> Dict[str, Any]:
    """加载数据配置"""
    loader = ConfigLoader(get_project_root() / "config")
    return loader.get_data_config()


def load_model_config() -> Dict[str, Any]:
    """加载模型配置"""
    loader = ConfigLoader(get_project_root() / "config")
    return loader.get_model_config()


def load_experiment_config() -> Dict[str, Any]:
    """加载实验配置"""
    loader = ConfigLoader(get_project_root() / "config")
    return loader.get_experiment_config()


if __name__ == "__main__":
    # 测试配置加载
    print("测试配置加载...")
    
    try:
        loader = ConfigLoader()
        
        # 加载各个配置
        data_config = loader.get_data_config()
        print("✓ 数据配置加载成功")
        print(f"  - 活性阈值: {data_config['filtering']['active_threshold']} μM")
        
        model_config = loader.get_model_config()
        print("✓ 模型配置加载成功")
        print(f"  - Random Forest树数量: {model_config['random_forest']['n_estimators']}")
        
        exp_config = loader.get_experiment_config()
        print("✓ 实验配置加载成功")
        print(f"  - 项目名称: {exp_config['experiment']['project_name']}")
        
        print("\n所有配置加载成功！")
        
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
