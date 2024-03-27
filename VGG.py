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
    def __init__(self, in_channels = 1, num_classes = 10):
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
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train= False, transform=my_transforms, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)




# ------------ INITIALIZE NETWORK ------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)
model = VGG_net().to(device)

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
