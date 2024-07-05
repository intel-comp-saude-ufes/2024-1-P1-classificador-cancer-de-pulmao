import torch.nn as nn
from torchvision.models import vgg19


class CustomVGG19(nn.Module):

    def __init__(self, num_class: int) -> None:
        super().__init__()
        self.model_conv = vgg19(weights="IMAGENET1K_V1")
        for param in self.model_conv.features.parameters():
            param.requires_grad = False

        self.model_conv.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_class),
        )
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, softmax: bool = False):
        out = self.model_conv(x)
        if softmax:
            out = self.soft(out)
        return out
