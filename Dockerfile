FROM apache/airflow:2.10.2-python3.10

USER root
WORKDIR /opt/airflow

USER airflow

COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

COPY --chown=airflow:0 dags/     /opt/airflow/dags/
COPY --chown=airflow:0 scripts/  /opt/airflow/scripts/
COPY --chown=airflow:0 utils/    /opt/airflow/utils/
COPY --chown=airflow:0 plugins/  /opt/airflow/plugins/

