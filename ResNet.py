# ResNet: General Idea - accuracy should go down as we increase accuracy
# But - sometimes gets worse as we deepend architecture. 
# Introduction now "Skip-Connections", takes input of previous layer 
# but as well from n-layers before
# Identity Mapping: we choose what we get into next layer, combination of both. 
# It should never become worse.

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride = 1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride = 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.relu = nn.ReLU()
        
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Use it if we need to change the shape in some way
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    # Block, layers (list, how many times block is reused), image_channels (input channels, e.g. RGB)
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size= 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding=1)
        
        # ResNet layers (out always times 4)
        self.layer1 = self.make_layers(block, layers[0], out_channels=64, stride = 1)
        self.layer2 = self.make_layers(block, layers[1], out_channels=128, stride = 2)
        self.layer3 = self.make_layers(block, layers[2], out_channels=256, stride = 2)
        self.layer4 = self.make_layers(block, layers[3], out_channels=512, stride = 2) 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # breakpoint()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc(x)
        
        return x 
    
    
    def make_layers(self, block, num_of_residual_block, out_channels, stride):
        # block (as above), num_of_residual_block (number of times block is used),
        # out_channels (number of channels when done with that layer)
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                          stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        # -1 because already one computed
        for i in range(num_of_residual_block -1):
            layers.append(Block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    
def ResNet50(image_channels=1, num_classes=10):
    return ResNet(Block, [3,4,6,3], image_channels, num_classes)

def ResNet101(image_channels=1, num_classes=10):
    return ResNet(Block, [3,4,23,3], image_channels, num_classes)

def ResNet152(image_channels=1, num_classes=10):
    return ResNet(Block, [3,8,36,3], image_channels, num_classes)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)

x = torch.rand(1,1,224,224)

model = ResNet50()
# model.to(device)

print(model(x).shape)

