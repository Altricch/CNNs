import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# LetNet Architecture per paper
# Input:    1x32x32
# C1:       6 feature maps
# S2:       6 feature maps
# C3:       16 feature maps
# S4:       16 feature maps
# F5:       120 
# F6:       84
# Out:      10


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Paper uses tanH and sigmoid (we use ReLU)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 6, kernel_size=(5,5), stride = (1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels = 16, kernel_size=(5,5), stride = (1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels = 120, kernel_size=(5,5), stride = (1,1), padding=(0,0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


x = torch.randn(64,1,32,32)
model = LeNet()
print(model(x).shape)        

        
# ------------ Set Device ------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)


# ------------ Hyperparameters ------------- #
IN_CHANNELS = 1
CLASSES= 10
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 20


# ------------ Load Data ------------- #
my_transforms = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=my_transforms, download=True)
print("SHAPE", train_dataset[0])
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train= False, transform=my_transforms, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------ INITIALIZE NETWORK ------------- #
model = LeNet().to(device)

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