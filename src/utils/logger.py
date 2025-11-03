"""
日志工具模块
"""
import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "NLRP3_Screening",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名（可选）
        log_dir: 日志文件目录
        
    Returns:
        配置好的日志记录器
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（如果指定了log_file）
    if log_file:
        # 创建日志目录
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件路径
        log_file_path = log_dir_path / log_file
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    为实验创建专用的日志记录器
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        
    Returns:
        日志记录器
    """
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=experiment_name,
        level="INFO",
        log_file=log_file,
        log_dir=log_dir
    )


class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, logger: logging.Logger, total: int, desc: str = "Progress"):
        """
        初始化进度日志记录器
        
        Args:
            logger: 日志记录器
            total: 总步数
            desc: 描述
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_log_percent = 0
    
    def update(self, n: int = 1):
        """
        更新进度
        
        Args:
            n: 增加的步数
        """
        self.current += n
        percent = int(100 * self.current / self.total)
        
        # 每10%记录一次
        if percent >= self.last_log_percent + 10:
            self.logger.info(f"{self.desc}: {percent}% ({self.current}/{self.total})")
            self.last_log_percent = percent
    
    def finish(self):
        """完成进度"""
        self.logger.info(f"{self.desc}: 100% 完成")


def log_section(logger: logging.Logger, title: str):
    """
    记录一个新的部分标题
    
    Args:
        logger: 日志记录器
        title: 标题
    """
    separator = "=" * 50
    logger.info(separator)
    logger.info(f"  {title}")
    logger.info(separator)


def log_dict(logger: logging.Logger, data: dict, title: str = "数据"):
    """
    以格式化的方式记录字典
    
    Args:
        logger: 日志记录器
        data: 要记录的字典
        title: 标题
    """
    logger.info(f"{title}:")
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


# 创建默认logger
default_logger = setup_logger()


if __name__ == "__main__":
    # 测试日志功能
    print("测试日志功能...")
    
    # 测试基本日志
    logger = setup_logger("TestLogger", level="DEBUG")
    logger.debug("这是DEBUG信息")
    logger.info("这是INFO信息")
    logger.warning("这是WARNING信息")
    logger.error("这是ERROR信息")
    
    # 测试部分标题
    log_section(logger, "测试部分")
    
    # 测试字典记录
    test_data = {
        "模型": "Random Forest",
        "参数": {
            "n_estimators": 500,
            "max_depth": 20
        },
        "性能": 0.85
    }
    log_dict(logger, test_data, "模型信息")
    
    # 测试进度记录
    progress = ProgressLogger(logger, total=100, desc="处理数据")
    for i in range(0, 101, 10):
        progress.update(10)
    progress.finish()
    
    print("\n日志功能测试完成！")
