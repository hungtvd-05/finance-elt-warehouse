import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pipeline as pipe
import etl_tasks as etl

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def task_init_schema():
    pool = etl.connect_to_db()
    try:
        pipe.initialize_database_schema(pool)
    finally:
        pool.closeall()

def task_load_new_data():
    pool = etl.connect_to_db()
    try:
        today = datetime.now().today()
        start_date = (today - timedelta(days=7)).isoformat()

        pipe.load_raw_historical_market_indicators(pool, start_date)
        pipe.load_raw_historical_stock_prices(pool, start_date)
    finally:
        pool.closeall()

def task_transform_data():
    pool = etl.connect_to_db()
    try:
        pipe.transform_historical_market_indicators(pool, update=True)
        pipe.transform_historical_stock_prices(pool, update=True)
    finally:
        pool.closeall()


def task_predict_daily():
    pool = etl.connect_to_db()
    try:
        pipe.predict_all_stocks(pool)
    finally:
        pool.closeall()

dag = DAG(
    dag_id='stock_daily_pipeline',
    default_args=default_args,
    description='Daily Stock ETL & Prediction',
    schedule='0 6 * * *',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=['stock', 'daily']
)

with dag:
    t0 = PythonOperator(
        task_id='init_schema',
        python_callable=task_init_schema,
    )

    t1 = PythonOperator(
        task_id='load_data',
        python_callable=task_load_new_data,
    )

    t2 = PythonOperator(
        task_id='transform_data',
        python_callable=task_transform_data,
    )

    t3 = PythonOperator(
        task_id='predict_data',
        python_callable=task_predict_daily,
    )

    t0 >> t1 >> t2 >> t3