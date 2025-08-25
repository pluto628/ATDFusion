import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, device="cpu"):
        super(FeatureExtractor, self).__init__()

        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)

        self.features = vgg16.features.to(device)
        for param in self.features.parameters():
            param.requires_grad = False

        self.selected_layers = {
            "conv1_2": 3,
            "conv2_2": 8,
            "conv3_3": 15,
        }

    def forward(self, x):
        features_list = []
        for i, layer in enumerate(self.features):  
            x = layer(x)
            if i in self.selected_layers.values():  
                features_list.append(x)

        return features_list 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor(device=device)

    x = torch.randn(1, 3, 256, 256, device=device)
    # print(f"Input shape: {x.shape}\n")

    features_list = model(x)

    for i, feat in enumerate(features_list):
        print(f"Feature {i}: {feat.shape}")
