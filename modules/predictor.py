import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn


class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names

        self.transforms = nn.Sequential(
            T.Resize(
                [
                    256,
                ],
                antialias=True,
            ),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            x = self.model(x)
            x = F.softmax(x, dim=1)

            return x
