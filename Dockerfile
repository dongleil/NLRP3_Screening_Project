# NLRP3筛选项目Docker镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装依赖
RUN conda install -c conda-forge rdkit -y && \
    pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter && \
    pip install xgboost imbalanced-learn && \
    pip install pyyaml tqdm joblib plotly statsmodels chembl-webresource-client

# 设置环境变量
ENV PYTHONPATH=/app

# 默认命令
CMD ["/bin/bash"]
