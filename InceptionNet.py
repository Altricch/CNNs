# Idea: How to find a good kernel size -> let the network decide (paper from 2014 - thus computation a bigger issue back then)
# 5x5 conv can be expensive on a lot of filters, thus, we try different conv layers and take all results
# concatenate them into the next layer (naive approach). 
# Now: we do dimension reduction meaning we use a 1x1 conv prior to a 3x3 and 5x5 conv to reduce number of filters (inception blocks)
# They use auxillary classifiers on the side inbetween training (to ensure layers do something useful inbetween), not done here for now

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class InceptionNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes=10):
        super(InceptionNet, self).__init__()
        
        self.conv1 = conv_block(in_channels=in_channels, out_channels = 64, kernel_size = (7,7), stride= (2,2), padding= (3,3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride = 1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # In order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride =1)
        
        self.dropout = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # breakpoint()
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        return x
        

class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inception_block, self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )
        
    def forward(self, x):
        # N (number of images) x filters x width x height -> we want to concatenate the number of filters
        # thats why dim = 1
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


# Create conv block that can be reused
class conv_block(nn.Module):
    # **Kwargs for e.g. kernel size, padding, stride -> 
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    

# ------------ Hyperparameters ------------- #
IN_CHANNELS = 1
CLASSES= 10
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 1



# ------------ Load Data ------------- #
my_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=my_transforms, download=True)
# print("SHAPE", train_dataset[0])
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train= False, transform=my_transforms, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)




# ------------ INITIALIZE NETWORK ------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)
model = InceptionNet().to(device)

# ------------ LOSS FUNCTION ------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------ Train Loop ------------- #
for epoch in range(EPOCHS):
    print("EPOCH", epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
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



# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("DEVICE", device)
#     x = torch.randn(3,3,224,224)
#     print("X def")
#     model = InceptionNet()
#     print("Model created")
#     # model.to(device)
#     print("Model on device")
#     print(model(x).shape)