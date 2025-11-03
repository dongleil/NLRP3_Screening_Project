@echo off
chcp 65001 >nul
echo ====================================
echo NLRP3项目环境自动安装脚本
echo ====================================
echo.
echo 此脚本将自动安装所有必需的Python包
echo 预计需要10-20分钟，请耐心等待...
echo.
pause

echo.
echo [步骤1/5] 配置conda和pip镜像源...
conda config --set ssl_verify false
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
echo ✓ 配置完成

echo.
echo [步骤2/5] 安装RDKit（化学信息学库）...
echo 这可能需要几分钟...
conda install -c conda-forge rdkit -y
if %ERRORLEVEL% NEQ 0 (
    echo ✗ RDKit安装失败，尝试pip安装...
    pip install rdkit
)
echo ✓ RDKit安装完成

echo.
echo [步骤3/5] 安装基础科学计算包...
pip install numpy pandas scipy scikit-learn matplotlib seaborn
if %ERRORLEVEL% NEQ 0 (
    echo ✗ 基础包安装失败
    pause
    exit /b 1
)
echo ✓ 基础包安装完成

echo.
echo [步骤4/5] 安装机器学习包...
pip install xgboost imbalanced-learn statsmodels
if %ERRORLEVEL% NEQ 0 (
    echo ✗ 机器学习包安装失败
    pause
    exit /b 1
)
echo ✓ 机器学习包安装完成

echo.
echo [步骤5/5] 安装工具包...
pip install pyyaml tqdm joblib plotly jupyter ipykernel chembl-webresource-client
if %ERRORLEVEL% NEQ 0 (
    echo ✗ 工具包安装失败
    pause
    exit /b 1
)
echo ✓ 工具包安装完成

echo.
echo ====================================
echo 所有包安装完成！
echo ====================================
echo.
echo 正在运行环境检查...
echo.

python scripts\check_environment.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================
    echo 🎉 环境配置成功！
    echo ====================================
    echo.
    echo 你现在可以开始使用项目了：
    echo   1. 运行数据下载: python experiments\stage0_data\01_download_chembl.py
    echo   2. 运行数据预处理: python experiments\stage0_data\02_preprocess_data.py
    echo.
) else (
    echo.
    echo ====================================
    echo ⚠ 环境检查发现问题
    echo ====================================
    echo.
    echo 部分包可能未正确安装
    echo 请查看上面的错误信息
    echo.
)

pause
