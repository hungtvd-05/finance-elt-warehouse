# import insert_records as ir
#
#
# if __name__ == "__main__":
#     connection_pool = ir.connect_to_db()
#     ir.initialize_database_schema(connection_pool)
#     # ir.populate_dim_date(connection_pool)
#     ir.get_or_create_date_key(connection_pool, '2025-11-13')

import sys
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'description': 'A simple DAG to demonstrate Airflow setup',
    'start_date': datetime(2025, 11, 14),
    'catchup': False,
}

dag = DAG(
    dag_id='api_request',
    default_args=default_args,
    schedule=timedelta(minutes=1),
)

with dag:
    task1 = PythonOperator(
        task_id='print_hello',
        python_callable=lambda: print("Hello, Airflow!"),
    )