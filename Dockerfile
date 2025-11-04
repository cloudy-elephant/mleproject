# ===============================
# Dockerfile for MLE Assignment
# ===============================

FROM apache/airflow:2.10.2-python3.10

# 设置时区
ENV TZ=Asia/Singapore
USER root

# 安装系统依赖（包含 Java 以兼容 Spark）
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"


# ...前面 apt 装 Java 的部分保持 root 身份
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:/home/airflow/.local/bin:$PATH"

# 先复制 requirements
COPY requirements.txt /requirements.txt

USER airflow
RUN pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir pyspark==3.5.2 "pyarrow>=14,<19"
# 如需 Delta Lake：
# RUN pip install --no-cache-dir delta-spark==3.2.0


# 用 root 拷贝文件并直接设定属主
USER root
RUN mkdir -p /opt/airflow/scripts
COPY --chown=airflow:root scripts/ /opt/airflow/scripts/
COPY --chown=airflow:root dags/ /opt/airflow/dags/

# 仅为新建目录设置权限（避免 chown 整个 /opt/airflow 触发 OPNOTPERM）
RUN mkdir -p /opt/airflow/{logs,datamart,model_bank} \
    && chown -R airflow:root /opt/airflow/{logs,datamart,model_bank}

# 最终运行用 airflow 用户
USER airflow
WORKDIR /opt/airflow


# 运行期环境（常见小坑规避）
ENV PYSPARK_PYTHON=python
ENV SPARK_LOCAL_IP=127.0.0.1
ENV PYARROW_IGNORE_TIMEZONE=1

# 将脚本和 DAG 拷贝进容器
RUN mkdir -p /opt/airflow/scripts
COPY scripts/ /opt/airflow/scripts/
COPY dags/ /opt/dags/


USER airflow
WORKDIR /opt/airflow
