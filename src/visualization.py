import os
import shutil
import numpy as np
import mlflow

from skimage import io
from skimage import color
from PIL import Image
from matplotlib import pyplot as plt

from data import CustomDataset


def log_samples(img_paths, mask_paths, num_samples=5):
    """logs images smaples with masks as artifacts
    """

    sample_dir = "dataset_samples"
    os.makedirs(sample_dir, exist_ok=True)

    dataset = CustomDataset(img_paths,
                            mask_paths)

    for i in range(num_samples):
        image = dataset[i]['pixel_values'].permute(1, 2, 0).numpy()
        mask = dataset[i]['labels'].numpy()
        masked_img = color.label2rgb(mask, image, alpha=0.2)
        masked_img = (masked_img * 255).astype(np.uint8)
        io.imsave(os.path.join(sample_dir, f"sample_{i}.png"), masked_img)

    # Log the directory as an artifact
    mlflow.log_artifacts(sample_dir, artifact_path="samples")
    shutil.rmtree(sample_dir)


def log_results(model, df, idxs):
    """logs prediction results
    """

    sample_dir = "prediction_samples"

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for idx in idxs:

        img_path = df['img_path'].values[idx]
        mask_path = df['mask_path'].values[idx]
        pred = model.predict({'img_path': df['img_path'].values[idx]})

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        image = np.asarray(Image.open(img_path).convert(mode='RGB'))
        mask = np.asarray(Image.open(mask_path))
        masked_img = mask_img(mask, image)
        axes[0].imshow(masked_img)

        masked_img = mask_img(pred, image)
        axes[1].imshow(masked_img)

        axes[0].set_title('ground truth')
        axes[1].set_title('prediction')

        plt.savefig(os.path.join(sample_dir, f"sample_{idx}.png"))

    mlflow.log_artifacts(sample_dir, artifact_path="prediction_samples")
    shutil.rmtree(sample_dir)


def mask_img(mask, image):
    masked_img = color.label2rgb(mask, image, alpha=0.2)
    masked_img = (masked_img * 255).astype(np.uint8)
    return masked_img
