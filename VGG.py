# Implementation of VGG16
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

VGG16 = [64,64,"M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG16)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        
        return x
    
    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(x),
                                     nn.ReLU(x)]
                
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        
        # Unpack all layers and create the network
        return nn.Sequential(*layers)

model = VGG_net(in_channels=3, num_classes=1000)
x = torch.randn(64, 3, 224, 224)
print(model(x).shape)