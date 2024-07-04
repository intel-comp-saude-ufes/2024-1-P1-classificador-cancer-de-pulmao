import torch.nn as nn
from torchvision.models import resnet50


class CustomResNet(nn.Module):

    def __init__(
        self,
        num_class: int,
        freeze_resnet: bool = False
    ) -> None:
        super().__init__()
        self.model_conv = resnet50(weights="IMAGENET1K_V2")
        for param in self.model_conv.parameters():
            param.requires_grad = not freeze_resnet
        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(num_ftrs, num_class)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, softmax: bool = False):
        out = self.model_conv(x)
        if softmax:
            out = self.soft(out)
        return out
