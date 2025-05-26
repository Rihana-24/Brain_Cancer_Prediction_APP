import torch.nn as nn
from torchvision import models

def get_pretrained_model():
    # load a pretrained ResNet (e.g., ResNet18)
    resnet = models.resnet18(weights='DEFAULT')

    # freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # replace the final classification layer to match  b_cancer classes (4 digits)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 4)  # b_cancer has 10 classes
    return resnet