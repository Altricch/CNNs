# EfficientNet: Rethinking Model Scaling for CNNs

import torch 
import torch.nn as nn
from math import ceil

import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


base_model = [
    # MBConv from paper
    # Expand_ratio, channels, repreats, stride, kernel_size
    # Stride of 2 if we downsample height and width (e.g. divide by 2)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# alpha, beta and gamma values of paper
phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0,224,0.2), # alpga, beta, gamma, depth = alpha ** phi
    "b1": (0.5,240,0.2),
    "b2": (1,260,0.3),
    "b3": (2,300,0.3),
    "b4": (3,380,0.4),
    "b5": (4,456,0.4),
    "b6": (5,528,0.5),
    "b7": (6,600,0.5),
}

class CNNBlock(nn.Module):
    # groups are depth wise convolution for each channel independently
    # If we set group = 1 then it is a regular convolution, if we set it to
    # in channels, then it becomes a depthwise conv
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        
        self.cnn = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             groups=groups,
                             # Since were using Batchnorm
                             bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

# Compute attention scores for each of the channels
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            # Takes h, w and give 1 value as output
            # C x H x W -> C x 1 x 1
            nn.AdaptiveAvgPool2d(1),
            # channel reduction -> C/r, 1, 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            # Ensure non-linearity
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            # Get a value between 0 and 1 for each channel
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        # self attention on each channel, how much to we prioritize 
        # each channel
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    # reduction for se block, survival_prob for stochastic depth
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction = 4, survival_prob = 0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        # Use skip connection if in is equal to out and stride == 1
        self.use_residual = in_channels == out_channels and stride == 1
        # Middle of inverted residual block
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)
        
        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size= 3, stride = 1, padding = 1,
            )
        self.conv = nn.Sequential(
            CNNBlock(
                # Depth wise convolution
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups= hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduced_dim), 
            nn.Conv2d(hidden_dim, out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    # Randomly removes a certain layer 
    # Sometimes we are going to skip certain layers with stochastic depth
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1,1,1, device=x.device) < self.survival_prob
        
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # This is going to be our network
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )
        
    # resolution we have (gamma), need to calculate alpha and beta
    # Alpha for depth scaling, beta for width scaling
    def calculate_factors(self, version, alpha=1.2, beta= 1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
            
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding = 1)]
        in_channels = channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            # make sure its always divisible by 4. Here we increase width
            out_channels = 4*ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            
            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        expand_ratio = expand_ratio,
                        # Only want to downsample at the first one
                        stride = stride if layer == 0 else 1,
                        kernel_size = kernel_size,
                        padding = kernel_size // 2, # if k=1: pad=0, if k=3:pad=1, if k=5:pad=2
                    )
                )
                
                in_channels = out_channels
        
        
        features.append(
            CNNBlock(in_channels, 
                     last_channels, 
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     )
        )
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
    
        
# ------------ Set Device ------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)


# ------------ Hyperparameters ------------- #
IN_CHANNELS = 1
CLASSES= 10
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 20
VERSION = "b0"
PHI, RES, DROP_RATE = phi_values[VERSION] 

# ------------ Load Data ------------- #


my_transforms = transforms.Compose([
    transforms.Pad(2),
    transforms.Resize((RES, RES)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=my_transforms, download=True)
print(train_dataset[0][0].shape)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train= False, transform=my_transforms, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------ INITIALIZE NETWORK ------------- #

model = EfficientNet(version=VERSION, num_classes=CLASSES).to(device)

# ------------ LOSS FUNCTION ------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------ Train Loop ------------- #
for epoch in range(EPOCHS):
    print("EPOCH", epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        print("batch_idx", batch_idx)
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        # Set all gradients to 0 for each batch, s.t. it doesnt store the back prop calculations
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent 
        optimizer.step()
        
        
# ------------ Check Accuracy ------------- #

def check_accuracy(loader, model):
    num_correct = 0
    num_sampels = 0
    
    model.eval()
    
    # No gradent computation needed
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sampels += predictions.size(0)
        
        print(f"Got {num_correct} / {num_sampels} with accuracy {float(num_correct)/float(num_sampels)*100}") 
    
    model.train()
        
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
    
        