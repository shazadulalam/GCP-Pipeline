# IMPORT REQUIRED LIBRARIES
import datetime as dt
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        component)
from kfp.v2 import compiler
from google.cloud.aiplatform import pipeline_jobs
#from kfp.components import create_component_from_func

# Parameters
PROJECT_ID = 'gcp-learn-430710'
REGION = 'europe-west1'  # or the region you're using
BUCKET_NAME = 'gcp-pipeline'  # Replace with your actual bucket name

BASE_IMAGE = 'eu.gcr.io/gcp-learn-430710/thelook_training_demo:latest'

# components

@component(
    base_image=BASE_IMAGE,
    output_component_file="get_data.yaml"
)
def create_dataset_from_bq(
    output_dir: Output[Dataset],
):
    import pandas as pd
    from src.preprocess import create_dataset_from_bq as preprocess_create_dataset_from_bq

    try:
        df = preprocess_create_dataset_from_bq()
        df.to_csv(output_dir.path, index=False)
    except Exception as e:
        print(f"Error in create_dataset_from_bq: {e}")
        raise

@component(
    base_image=BASE_IMAGE,
    output_component_file="train_test_split.yaml",
)
def make_data_splits(
    dataset_full: Input[Dataset],
    dataset_train: Output[Dataset],
    dataset_val: Output[Dataset],
    dataset_test: Output[Dataset]
):
    import pandas as pd
    from src.preprocess import make_data_splits

    df_agg = pd.read_csv(dataset_full.path)
    df_agg.fillna('NA', inplace=True)

    df_train, df_val, df_test = make_data_splits(df_agg)
    print(f"{len(df_train)} samples in train")
    print(f"{len(df_val)} samples in val")
    print(f"{len(df_test)} samples in test")

    df_train.to_csv(dataset_train.path, index=False)
    df_val.to_csv(dataset_val.path, index=False)
    df_test.to_csv(dataset_test.path, index=False)

@component(
    base_image=BASE_IMAGE,
    output_component_file="train_model.yaml",
)
def train_model(
    dataset_train: Input[Dataset],
    dataset_val: Input[Dataset],
    dataset_test: Input[Dataset],
    model: Output[Model]
):
    import json
    from src.train import train_and_evaluate

    outputs = train_and_evaluate(
        dataset_train.path,
        dataset_val.path,
        dataset_test.path
    )
    cb_model = outputs['model']
    scores = outputs['scores']

    model.metadata["framework"] = "catboost"
    # Save the model as an artifact
    with open(model.path, 'w') as f:
        json.dump(scores, f)

@component(
    base_image="python:3.9",
    output_component_file="compute_metrics.yaml",
)
def compute_metrics(
    model: Input[Model],
    train_metric: Output[Metrics],
    val_metric: Output[Metrics],
    test_metric: Output[Metrics]
):
    import json

    with open(model.path, 'r') as file:
        model_metrics = json.load(file)

    train_metric.log_metric('train_auc', model_metrics['train'])
    val_metric.log_metric('val_auc', model_metrics['eval'])
    test_metric.log_metric('test_auc', model_metrics['test'])


# Pipeline definition
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = f'thelook-pipeline-{TIMESTAMP}'
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"

@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="pipeline-demo"
)
def pipeline():
    load_data_op = create_dataset_from_bq()
    train_test_split_op = make_data_splits(
        dataset_full=load_data_op.outputs["output_dir"]
    )
    train_model_op = train_model(
        dataset_train=train_test_split_op.outputs["dataset_train"],
        dataset_val=train_test_split_op.outputs["dataset_val"],
        dataset_test=train_test_split_op.outputs["dataset_test"]
    )
    compute_metrics(
        model=train_model_op.outputs["model"]
    )

# Compile the pipeline as JSON
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path='thelook_pipeline.json'
)

# Initialize and run the pipeline
pipeline_job = pipeline_jobs.PipelineJob(
    display_name="thelook-pipeline",
    template_path="thelook_pipeline.json",
    enable_caching=False,
    location=REGION,
    project=PROJECT_ID
)

pipeline_job.run(service_account='bigquery@gcp-learn-430710.iam.gserviceaccount.com')
