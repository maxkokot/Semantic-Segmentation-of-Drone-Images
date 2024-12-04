import pandas as pd
from torch import nn
from PIL import Image


class BaseModelWrapper:
    '''Base MLflow pyfunc wrapper class'''

    def predict(self, inputs):

        image_path = inputs['img_path']

        if isinstance(image_path, pd.Series):
            image_path = image_path.values[0]

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        logits = self.model(**inputs).logits
        upsampled_logits = nn.functional.interpolate(
                                logits,
                                size=image.size[::-1], 
                                mode='bilinear',
                                align_corners=False
                            )

        outputs = upsampled_logits.argmax(dim=1)[0].numpy()
        return outputs
