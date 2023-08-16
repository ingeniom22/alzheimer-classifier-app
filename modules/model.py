from collections import OrderedDict

from torch import nn
from torchvision.models import efficientnet_v2_s


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.7) -> None:
        super().__init__()
        self.model = efficientnet_v2_s(weights=None)

        for idx, layer in enumerate(self.model.features):
            if idx > 7:
                layer.requires_grad = True
            else:
                layer.requires_grad = False

        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(dropout)),
                    ("inputs", nn.Linear(1280, 512)),
                    ("relu1", nn.ReLU()),
                    ("dropout2", nn.Dropout(dropout)),
                    ("outputs", nn.Linear(512, num_classes)),
                ]
            )
        )

        self.model.classifier = classifier

    def forward(self, x):
        return self.model(x)
