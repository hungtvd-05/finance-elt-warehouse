import sys
import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import etl_tasks as etl
import pipeline as pipe

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
}

def task_init_schema():
    pool = etl.connect_to_db()
    try:
        pipe.initialize_database_schema(pool)
    finally:
        pool.closeall()

def task_populate_dim_date():
    pool = etl.connect_to_db()
    try:
        pipe.populate_dim_date_table(pool)
    finally:
        pool.closeall()

def task_retrain_all_models():
    pool = etl.connect_to_db()
    try:
        pipe.train_all_stocks(pool)
    finally:
        pool.closeall()

with DAG(
    dag_id='stock_quarterly_retraining',
    default_args=default_args,
    description='Train lại model định kỳ',
    schedule='0 0 1 */3 *',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=['stock', 'training', 'heavy'],
) as dag:

    t0 = PythonOperator(
        task_id='init_schema',
        python_callable=task_init_schema,
    )

    t1 = PythonOperator(
        task_id='populate_dim_date',
        python_callable=task_populate_dim_date,
    )

    train_task = PythonOperator(
        task_id='retrain_all_models',
        python_callable=task_retrain_all_models,
        execution_timeout=timedelta(hours=5),
    )

    t0 >> t1 >> train_task