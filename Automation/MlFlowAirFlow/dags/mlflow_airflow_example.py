# Filename: mlflow_airflow_example.py

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Define MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Define MLflow experiment name
mlflow.set_experiment("linear_regression")

# Function to train the model
def train_model():
    # Load data
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

# Function to log model and metrics using MLflow
def log_model_and_metrics(model, X_train, X_test, y_train, y_test):
    # Log metrics
    mlflow.log_metric("train_rmse", mean_squared_error(y_train, model.predict(X_train), squared=False))
    mlflow.log_metric("test_rmse", mean_squared_error(y_test, model.predict(X_test), squared=False))
    
    # Log model
    mlflow.sklearn.log_model(model, "model")

# Function to deploy the model
def deploy_model():
    # Load the model
    model = mlflow.sklearn.load_model("model")
    
    # Perform deployment steps here (e.g., serialize model, deploy to production environment)
    # For simplicity, let's just print out the coefficients of the model
    print("Model coefficients:", model.coef_)

# Define Airflow DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 18),
    'retries': 1,
}

dag = DAG(
    'mlflow_airflow_example',
    default_args=default_args,
    description='MLflow and Airflow example',
    schedule_interval='@daily',
)

# Define task for training the model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Define task for logging model and metrics
log_model_and_metrics_task = PythonOperator(
    task_id='log_model_and_metrics',
    provide_context=True,
    python_callable=log_model_and_metrics,
    dag=dag,
)

# Define task for deploying the model
deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Set up task dependencies
train_model_task >> log_model_and_metrics_task >> deploy_model_task
