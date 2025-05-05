import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_resnet18(num_classes=6, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle ResNet18 préentraîné sur ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Remplacer la couche de classification finale pour correspondre à 6 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    return model.to(device)
