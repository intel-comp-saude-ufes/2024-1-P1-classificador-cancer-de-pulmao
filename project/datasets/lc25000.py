from typing import Optional, List, Tuple

import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LC25000(Dataset):
    def __init__(
        self,
        images_paths: List[str],
        labels: List[int],
        transform: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.images_paths = images_paths
        self.labels = labels

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(
        self
    ) -> int:
        return len(self.images_paths)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[Image.Image, int, str]:
        image_path = self.images_paths[idx]
        image = Image.open(image_path).convert("RGB")       
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, image_path