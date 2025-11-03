@echo off
chcp 65001 >nul
echo ====================================
echo NLRP3项目 - 环境诊断工具
echo ====================================
echo.

echo [检查1] Python版本...
python --version
echo.

echo [检查2] Conda版本...
conda --version
echo.

echo [检查3] pip配置...
pip config list
echo.

echo [检查4] conda配置...
conda config --show
echo.

echo [检查5] 已安装的包...
echo.
echo 核心包：
python -c "import sys; packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'rdkit', 'xgboost', 'matplotlib']; [print(f'  {p}: ', end='') or __import__(p) and print('✓ 已安装') or True for p in packages]" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   检测包时出错，部分包未安装
)
echo.

echo [检查6] 网络连接...
echo 测试pip镜像源连接...
ping -n 1 pypi.tuna.tsinghua.edu.cn >nul
if %ERRORLEVEL% EQU 0 (
    echo   ✓ 清华源可访问
) else (
    echo   ✗ 清华源不可访问
)
echo.

echo ====================================
echo 诊断完成
echo ====================================
echo.
echo 如果有问题，请截图发给我
echo.
pause
