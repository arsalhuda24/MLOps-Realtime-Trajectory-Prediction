from datetime import datetime, timedelta
from textwrap import dedent
import os 
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator



with DAG(
    'ETL_pipeline',
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        'depends_on_past': False,
        #'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),

    },
    description='detection_tracking',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2022, 7, 28),
    catchup=False,
    tags=['example'],
) as dag:


    t2 = BashOperator( task_id="data_ingestion",
   
                      bash_command= "$HOME/anaconda3/envs/yolo_deepsort/bin/python $HOME/airflow/dags/data_ingest.py",
                      dag=dag,
)