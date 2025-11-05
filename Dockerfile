#FROM apache/airflow:2.10.2-python3.10
#
#USER root
#WORKDIR /opt/airflow
#
## 可选：装系统工具
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential git && rm -rf /var/lib/apt/lists/*
#
#USER airflow
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
## 把项目整体拷进去（compose 里也会用挂载覆盖，便于本地开发）
#COPY . /opt/airflow
FROM apache/airflow:2.10.2-python3.10

# （可选）如果你需要安装系统包再开启 root；如果不需要，可以删掉这段
USER root
WORKDIR /opt/airflow
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# 重要：切回 airflow 用户，再执行 pip
USER airflow

# 先只拷 requirements.txt，利用缓存
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# 拷贝项目代码（建议带 chown）
COPY --chown=airflow:0 dags/     /opt/airflow/dags/
COPY --chown=airflow:0 scripts/  /opt/airflow/scripts/
COPY --chown=airflow:0 utils/    /opt/airflow/utils/
COPY --chown=airflow:0 plugins/  /opt/airflow/plugins/
# 如果你还需要其它文件，再单独 COPY（避免把 logs/data 打包进去）
# COPY --chown=airflow:0 docker-compose.yaml /opt/airflow/
