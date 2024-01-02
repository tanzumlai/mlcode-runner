import sys
import importlib.metadata
sys.modules['importlib_metadata'] = importlib.metadata
import mlflow
import logging
from mlflow import MlflowClient
import traceback
from . import utils
import os


def get_root_run(active_run_id=None, experiment_names=None):
    runs = mlflow.search_runs(experiment_names=experiment_names, filter_string="tags.runlevel='root'", max_results=1,
                              output_format='list')
    if len(runs):
        parent_run_id = runs[0].info.run_id
        mlflow.set_tags({'mlflow.parentRunId': parent_run_id})
        return parent_run_id
    else:
        mlflow.set_tags({'runlevel': 'root'})
        return active_run_id


try:
    logging.getLogger().setLevel(logging.INFO)
    client = MlflowClient()
    git_repo = utils.get_cmd_arg_or_env_var("git_repo")
    entry_point = utils.get_cmd_arg_or_env_var("mlflow_entry")
    stage = utils.get_cmd_arg_or_env_var("mlflow_stage")
    environment_name = utils.get_cmd_arg_or_env_var("environment_name")
    experiment_name = utils.get_cmd_arg_or_env_var('experiment_name')
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = utils.get_cmd_arg('mlflow_s3_uri') or utils.get_env_var('MLFLOW_S3_ENDPOINT_URL')
    os.environ['MLFLOW_TRACKING_URI'] = utils.get_cmd_arg('mlflow_tracking_uri') or utils.get_env_var('MLFLOW_TRACKING_URI')

    logging.info(
        f"Printing the arguments...git_repo={git_repo},experiment_name={experiment_name},entry_point={entry_point},stage={stage}")

    with mlflow.start_run(nested=True) as active_run:
        os.environ['MLFLOW_RUN_ID'] = get_root_run(active_run_id=active_run.info.run_id, experiment_names=[experiment_name])
        submitted_run = mlflow.run(git_repo,
                                   entry_point,
                                   version=environment_name,
                                   env_manager='local',
                                   parameters={'model-stage': stage})

        submitted_run_metadata = MlflowClient().get_run(submitted_run.run_id)

        logging.info(f"Submitted Run: {submitted_run}\nSubmitted Run Metadata: {submitted_run_metadata}")

except mlflow.exceptions.RestException as e:
    logging.info('REST exception occurred (platform will retry based on pre-configured retry policy): ', exc_info=True)
    logging.info(''.join(traceback.TracebackException.from_exception(e).format()))

except mlflow.exceptions.ExecutionException as ee:
    logging.info("An ExecutionException occurred...", exc_info=True)
    logging.info(str(ee))
    logging.info(''.join(traceback.TracebackException.from_exception(ee).format()))

except BaseException as be:
    logging.info("An unexpected error occurred...", exc_info=True)
    logging.info(str(be))
    logging.info(traceback.format_exc())
    logging.info(''.join(traceback.TracebackException.from_exception(be).format()))

logging.info("End script.")
