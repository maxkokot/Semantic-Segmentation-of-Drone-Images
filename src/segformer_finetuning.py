# finetuning
import sys
import os
import shutil
import argparse
import logging
import warnings
import mlflow
from mlflow.models import infer_signature
from loguru import logger

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, \
    SegformerImageProcessor, TrainingArguments, Trainer
import evaluate

from model import BaseModelWrapper

from utils import load_data, load_metadata, get_last_run_id, load_config

# set up logging
logger.remove()
logger.add(sys.stdout,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)

DEFAULT_NUM_EPOCHS = 50
DEFAULT_LR = 0.00006
DEFAULT_BATCH_SIZE = 8
DEFAULT_BACKBONE = "nvidia/mit-b0"
DEFAULT_MODEL_NAME = 'semantic-drone-segmentation'
DEFAULT_CONFIG_PATH = None

MODEL_TMP_DIR = "models"


class SegFormerModelWrapper(BaseModelWrapper):
    '''Custom wrapper for the model'''
    def __init__(self, model_path):
        super(SegFormerModelWrapper, self).__init__()
        self.model = SegformerForSemanticSegmentation\
            .from_pretrained(model_path)
        self.processor = SegformerImageProcessor.from_pretrained(model_path)


class SegmentationPyFuncModel(mlflow.pyfunc.PythonModel):
    '''MLflow pyfunc wrapper'''
    def load_context(self, context):
        self.wrapper = SegFormerModelWrapper(context
                                             .artifacts["model_path"])

    def predict(self, context, model_input):
        return self.wrapper.predict(model_input)


def read_args(parser):

    config_path = parser.parse_args().config_path

    if config_path:
        config = load_config(config_path)
        num_epochs = config['num_epochs']
        lr = config['lr']
        batch_size = config['batch_size']
        backbone = config['backbone']
        model_name = config['model_name']

    else:
        num_epochs = parser.parse_args().num_epochs
        lr = parser.parse_args().lr
        batch_size = parser.parse_args().batch_size
        backbone = parser.parse_args().backbone
        model_name = parser.parse_args().model_name

    return num_epochs, lr, batch_size, \
        backbone, model_name


def compute_metrics(eval_pred):

    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=processor.do_reduce_labels,
        )

        metrics.pop("per_category_accuracy")
        metrics.pop("per_category_iou")

        return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", default=DEFAULT_NUM_EPOCHS, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, type=str)
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE, type=str)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH,
                        type=str)

    num_epochs, lr, batch_size, \
        backbone, model_name = read_args(parser)

    logger.info(f"Finetiningtuning started with: {num_epochs} epochs, "
                f"lr = {lr}, batch_size = {batch_size}, backbone = {backbone}")
    mlflow.transformers.autolog()

    experiment_id = mlflow.set_experiment('Semantic_Drone_Segmentation')\
        .experiment_id

    with mlflow.start_run(run_name='Finetuning',
                          log_system_metrics=True) as run:

        # get run id of last preprocessing step
        last_run_id = get_last_run_id('Data_Preprocessing', experiment_id)

        # load artifacts from last preprocessing step
        train_df, train_ds = load_data('train', last_run_id)
        test_df, test_ds = load_data('test', last_run_id)
        id2label, label2id = load_metadata(last_run_id)

        train_ds.processor = SegformerImageProcessor.from_pretrained(backbone)
        test_ds.processor = SegformerImageProcessor.from_pretrained(backbone)

        model = SegformerForSemanticSegmentation.from_pretrained(
                    backbone,
                    ignore_mismatched_sizes=True,
                    id2label=id2label,
                    label2id=label2id)

        logger.info('Starting finetuning')

        training_args = TrainingArguments(
            f"{backbone}-{model_name}",
            learning_rate=lr,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_total_limit=3,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=20,
            eval_steps=20,
            logging_steps=1,
            eval_accumulation_steps=5,
            load_best_model_at_end=True,
            push_to_hub=False,
            hub_strategy="end",
        )

        metric = evaluate.load("mean_iou")
        processor = SegformerImageProcessor.from_pretrained(backbone)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # log model
        trainer.save_model(MODEL_TMP_DIR)
        processor.save_pretrained(MODEL_TMP_DIR)
        wrapped_model = SegFormerModelWrapper(MODEL_TMP_DIR)
        input_example = {'img_path': train_df["img_path"].values[0]}
        output_example = wrapped_model.predict(input_example)
        signature = infer_signature(model_input=input_example,
                                    model_output=output_example)
        mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SegmentationPyFuncModel(),
                artifacts={"model_path": MODEL_TMP_DIR},
                signature=signature
            )

        # log datasets
        mlflow.log_text(train_df.to_csv(index=False),
                        os.path.join('datasets', 'train.csv'))
        mlflow.log_text(test_df.to_csv(index=False),
                        os.path.join('datasets', 'test.csv'))

        # register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        shutil.rmtree(MODEL_TMP_DIR)

        logger.info('Model registered')
