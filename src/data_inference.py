# data_evaluation
import sys
import json
import argparse
import warnings
import logging
import mlflow
from loguru import logger

from visualization import log_results
from utils import load_data, get_last_run_id, load_config

logger.remove()
logger.add(sys.stdout,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)

DEFAULT_IDXS = "[0, 1, 2, 3, 4]"
DEFAULT_CONFIG_PATH = None


def read_args(parser):

    config_path = parser.parse_args().config_path

    if config_path:
        config = load_config(config_path)
        model_name = config['model_name']
        idxs = config['prediction_idxs']

    else:
        idxs = parser.parse_args().idxs
        idxs = json.loads(idxs)
        model_name = parser.parse_args().model_name

    return model_name, idxs


if __name__ == '__main__':

    logger.info('Inference started')

    experiment_id = mlflow.set_experiment('Semantic_Drone_Segmentation')\
        .experiment_id

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_IDXS, type=str)
    parser.add_argument("--idxs", default=None, type=str)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH,
                        type=str)

    model_name, idxs = read_args(parser)

    logger.info(f"Inference started with: model {model_name} and {idxs} idxs")

    with mlflow.start_run(run_name='Data_Evaluation') as run:

        # get last finished run id for finetuning
        last_run_id = get_last_run_id('Model_Finetuning', experiment_id)

        last_version = mlflow.MlflowClient()\
            .get_registered_model(model_name)\
            .latest_versions[0].version

        model_uri = f"runs:/{last_run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        eval_dataset, _ = load_data('test', last_run_id)
        log_results(model, eval_dataset, idxs)
        logger.info("Prediction samples logged successfully!")
        logger.success('Evaluation finished')
