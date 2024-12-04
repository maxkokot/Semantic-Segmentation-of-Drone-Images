from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image


class CustomDataset(Dataset):
    '''Custom pytorch dataset'''

    def __init__(self, img_paths, mask_paths, processor=None):
        super(CustomDataset, self).__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.processor = processor

    def __len__(self):

        return len(self.mask_paths)

    def __getitem__(self, idx):

        image = Image.open(self.img_paths[idx]).convert(mode='RGB')
        mask = Image.open(self.mask_paths[idx])

        if self.processor:
            sample = self.processor(image, mask, return_tensors="pt")
            sample['pixel_values'] = sample['pixel_values'][0]
            sample['labels'] = sample['labels'][0]

        else:
            sample = {'pixel_values': F.pil_to_tensor(image),
                      'labels': F.pil_to_tensor(mask)[0]}

        return sample
