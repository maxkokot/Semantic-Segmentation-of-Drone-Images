# data_preprocessing
import sys
import argparse
import mlflow
import warnings
from sklearn.model_selection import train_test_split
from loguru import logger
from visualization import log_samples

from utils import get_drone_paths, get_metadata, \
    make_mlflow_dataset, make_metadata_dataset, \
    load_config


# set up logging
logger.remove()
logger.add(sys.stdout,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')


DEFAULT_ROOT_PATH = '../data/semantic_drone_dataset/semantic_drone_dataset'
DEFAULT_METADATA_PATH = '../data/colormaps.xlsx'
DEFAULT_DATASET_VERSION = '1.0'
DEFAULT_IMG_FOLDER = 'original_images'
DEFAULT_MASK_FOLDER = 'label_images_semantic'
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 2024
DEFAULT_CONFIG_PATH = None

NUM_SAMPLES = 5


def read_args(parser):

    config_path = parser.parse_args().config_path

    if config_path:
        config = load_config(config_path)
        root_path = config['root_path']
        metadata_path = config['metadata_path']
        dataset_version = config['dataset_version']
        img_folder = config['img_folder']
        mask_folder = config['mask_folder']
        test_size = config['test_size']
        random_state = config['random_state']

    else:
        root_path = parser.parse_args().root_path
        metadata_path = parser.parse_args().metadata_path
        dataset_version = parser.parse_args().dataset_version
        img_folder = parser.parse_args().img_folder
        mask_folder = parser.parse_args().mask_folder
        test_size = parser.parse_args().test_size
        random_state = parser.parse_args().random_state

    return root_path, metadata_path, dataset_version, \
        img_folder, mask_folder, test_size, random_state


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default=DEFAULT_ROOT_PATH,
                        type=str)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH,
                        type=str)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION,
                        type=str)
    parser.add_argument("--img-folder", default=DEFAULT_IMG_FOLDER, type=str)
    parser.add_argument("--mask-folder", default=DEFAULT_MASK_FOLDER, type=str)
    parser.add_argument("--test-size", default=DEFAULT_TEST_SIZE, type=float)
    parser.add_argument("--random-state", default=DEFAULT_RANDOM_STATE,
                        type=int)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH,
                        type=str)

    root_path, metadata_path, dataset_version, \
        img_folder, mask_folder, test_size, random_state = read_args(parser)

    logger.info(f"Data preprocessing started with test size: {test_size}"
                f"and random_state: {random_state}")

    experiment_id = mlflow.set_experiment('Semantic_Drone_Segmentation')\
        .experiment_id

    with mlflow.start_run(run_name='Data_Preprocessing'):

        # get paths of images and masks
        img_paths, mask_paths = get_drone_paths(root_path, img_folder,
                                                mask_folder)
        logger.info('Paths downloaded')

        # get metadata
        metadata = get_metadata(metadata_path)

        # log dataset lenth
        mlflow.log_metric('full_data_size', len(img_paths))

        # split dataset to train and test part and log sizes to mlflow
        train_img_paths, test_img_paths, \
            train_mask_paths, \
            test_mask_paths = train_test_split(img_paths,
                                               mask_paths,
                                               test_size=test_size,
                                               random_state=random_state)
        mlflow.log_metric('train_size', len(train_img_paths))
        mlflow.log_metric('test_size', len(test_img_paths))

        # log datasets
        make_mlflow_dataset(train_img_paths, train_mask_paths, 'train')
        make_mlflow_dataset(test_img_paths, test_mask_paths, 'test')
        make_metadata_dataset(metadata)

        # log samples with masks
        log_samples(train_img_paths, train_mask_paths, NUM_SAMPLES)

        logger.info("Dataset samples logged successfully!")
        logger.info('Data preprocessing finished')
