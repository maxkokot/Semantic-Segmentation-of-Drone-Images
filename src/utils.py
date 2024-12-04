import pandas as pd
import os
import yaml
import mlflow.pyfunc

import tempfile

from data import CustomDataset


def load_config(config_path):
    """Loads yaml config
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def get_drone_paths(root_path, ipt, tgt):
    """Returns images' and masks' paths
    """

    img_paths = os.listdir(os.path.join(root_path, ipt))
    mask_paths = list(map(lambda x: x.replace('jpg', 'png'), img_paths))

    img_paths = [os.path.join(root_path, ipt, p) for p in img_paths]
    mask_paths = [os.path.join(root_path, tgt, p) for p in mask_paths]

    return img_paths, mask_paths


def get_metadata(metadata_path):
    """Returns metadata
    """

    mask_metadata = pd.read_excel(metadata_path, usecols=['Classes', 'R', 'G',
                                                          'B', 'Id'])
    return mask_metadata


def make_mlflow_dataset(img_paths, mask_paths, kind='train'):
    """Makes and logs MLflow dataset
    """

    df = pd.DataFrame({'img_path': img_paths, 'mask_path': mask_paths})
    mlflow.log_text(df.to_csv(index=False), f"datasets/{kind}.csv")
    dataset_source_link = mlflow.get_artifact_uri(f"datasets/{kind}.csv")
    dataset = mlflow.data.from_pandas(df, name=kind,
                                      source=dataset_source_link)
    mlflow.log_input(dataset, tags={"framework": "pytorch",
                                    "task": "segmentation"})


def make_metadata_dataset(df):
    """Makes and logs MLflow dataset
    """

    mlflow.log_text(df.to_csv(index=False), "datasets/metadata.csv")
    dataset_source_link = mlflow.get_artifact_uri("datasets/metadata.csv")
    dataset = mlflow.data.from_pandas(df, name='metadata',
                                      source=dataset_source_link)
    mlflow.log_input(dataset, tags={"framework": "pytorch",
                                    "task": "segmentation"})


def load_data(kind, last_run_id):
    """Load artifacts of last preprocessing step
    and creates custom dataset
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.artifacts\
            .download_artifacts(run_id=last_run_id,
                                artifact_path=f"datasets/{kind}.csv",
                                dst_path=tmpdir)
        df = pd.read_csv(os.path.join(tmpdir, f"datasets/{kind}.csv"))

    dataset = CustomDataset(list(df.img_path.values),
                            list(df.mask_path.values))

    return df, dataset


def load_metadata(last_run_id):
    """Load artifacts of last preprocessing step
    and extracts metadata
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.artifacts\
            .download_artifacts(run_id=last_run_id,
                                artifact_path="datasets/metadata.csv",
                                dst_path=tmpdir)
        df = pd.read_csv(os.path.join(tmpdir, "datasets/metadata.csv"))

    id2label = df['Classes'].to_dict()
    label2id = df.set_index('Classes')['Id'].to_dict()

    return id2label, label2id


def get_last_run_id(run_name, experiment_id):
    """Gets last run id amoung runs with specified run_name
    """

    last_run_id = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' "
                      f"and status = 'FINISHED'",
        order_by=["start_time DESC"]
    ).loc[0, 'run_id']
    return last_run_id
