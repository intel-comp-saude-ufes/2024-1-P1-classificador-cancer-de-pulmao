import torch.nn as nn
from torchvision.models import vgg19


class CustomVGG19(nn.Module):

    def __init__(self, num_class: int) -> None:
        super().__init__()
        self.model_conv = vgg19(weights="IMAGENET1K_V1")
        for param in self.model_conv.parameters():
            param.requires_grad = False
        number_features = self.model_conv.classifier[6].in_features
        features = list(self.model_conv.classifier.children())[:-1]
        features.extend([nn.Linear(number_features, num_class)])
        self.model_conv.classifier = nn.Sequential(*features)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, softmax: bool = False):
        out = self.model_conv(x)
        if softmax:
            out = self.soft(out)
        return out
