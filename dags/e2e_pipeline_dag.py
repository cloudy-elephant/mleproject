from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"depends_on_past": False, "retries": 0}

with DAG(
    dag_id="e2e_pipeline",
    start_date=datetime(2023, 2, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["mle", "churn"],
) as dag:

    bronze = BashOperator(
        task_id="bronze_process",
        bash_command="python /opt/airflow/scripts/bronze_process.py"
    )

    silver = BashOperator(
        task_id="silver_process",
        bash_command="python /opt/airflow/scripts/silver_process.py"
    )

    gold = BashOperator(
        task_id="gold_process",
        bash_command="python /opt/airflow/scripts/gold_process.py"
    )

    logreg_train = BashOperator(
        task_id="logreg_train",
        bash_command="python /opt/airflow/scripts/logreg_train.py"
    )

    cat_train = BashOperator(
        task_id="catboost_gold_train",
        bash_command="python /opt/airflow/scripts/catboost_gold_train.py"
    )

    cat_predict = BashOperator(
        task_id="catboost_gold_predict",
        bash_command="python /opt/airflow/scripts/catboost_gold_predict.py"
    )

    monitor = BashOperator(
        task_id="monitor",
        bash_command="python /opt/airflow/scripts/monitor.py"
    )

    bronze >> silver >> gold >> [logreg_train, cat_train] >> cat_predict >> monitor
