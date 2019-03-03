import torch 
import torch.nn as nn 
from torchvision import models 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class Features(nn.Module):
    """Feature Extractor outputs convolutional features 
       using the imagenet pre-trained models. 
    """
    def __init__(self, arch = 'resnet18', pretrained=True):
        super(Features, self).__init__()

        model = models.__dict__[arch](pretrained=pretrained)
        
        # remove the classification layer 
        # Note: this doesn't work with inception-v3 model 
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x)

    @property
    def models(self):
        return model_names
