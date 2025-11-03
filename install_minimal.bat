@echo off
chcp 65001 >nul
echo ====================================
echo NLRP3项目 - 最小环境安装脚本
echo ====================================
echo.
echo 此脚本只安装核心必需包（约5分钟）
echo 足够运行数据处理和1D实验
echo.
pause

echo.
echo [1/3] 配置镜像源...
conda config --set ssl_verify false
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
echo ✓ 配置完成

echo.
echo [2/3] 安装RDKit...
conda install -c conda-forge rdkit -y
if %ERRORLEVEL% NEQ 0 (
    pip install rdkit
)
echo ✓ RDKit完成

echo.
echo [3/3] 安装其他核心包...
pip install numpy pandas scipy scikit-learn matplotlib xgboost pyyaml tqdm joblib
echo ✓ 核心包完成

echo.
echo ====================================
echo 最小环境安装完成！
echo ====================================
echo.
echo 验证安装...
python -c "import numpy, pandas, sklearn, rdkit, xgboost; print('✓ 所有核心包安装成功！')"

echo.
echo 运行项目检查...
python scripts\check_environment.py

echo.
pause
