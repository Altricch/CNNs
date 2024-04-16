# CNN replaced by patch embedding which is then transformed into 1D vector


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import torchvision.datasets as datasets


np.random.seed(0)
torch.manual_seed(0)

LR = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 32
IN_CHANNELS = 1
NUM_HEAD = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.999)
ACTIVATION_FUNCTION = "gelu"
NUM_ENCODERS = 4
EMDED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
EPOCHS = 10
BATCH_SIZE = 64

device = 'cuda' if torch.cuda.is_available() else "cpu"


# Transforms images into smaller patches, e.g. splits 
# the image up into e.g. a 3x3 grid -> 9 patches
# This part conatenates Patch + Position embedding + extra learnable class embedding
# When doing forward pass of this class, we find ourselves AFTER the linear projection of flattened patches of 
# Figure 1 of the paper 
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels) -> None:
        super().__init__()
        
        # we take our image (n) and break it into the patches
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride = patch_size,
            ),
            nn.Flatten(2)
        )
        
        # Number classes -> for MNIST 1 ->  
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad = True)
        # Number of positions is number of patches + 1 (look at diagram from paper)
        self.pos_embed = nn.Parameter(torch.randn(size=(1,num_patches + 1, embed_dim)), requires_grad = True)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        # Match cls_token to match size of batch numbe
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        # Switch around 2 and 1st dimension
        x = self.patch_embedding(x).permute(0,2,1)
        # Merge CLS token with patches
        x = torch.cat([cls_token, x], dim = 1)
        # Add positional embeddings to x
        x = self.pos_embed + x
        x = self.dropout(x)
        return x


class VIT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        # Define our Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoders)
        
        
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )
        
    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        
        # print(x.shape)
        # print(x[:, 0, :].shape)
        
        # Only take cls token
        x = self.mlp_head(x[:, 0, :])
        return x
        
        
        
# Test if it working
# model = PatchEmbedding(EMDED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS).to(device)
# x = torch.randn(512, 1, 28,28).to(device)
# print(model(x).shape)

# model = VIT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMDED_DIM, NUM_ENCODERS, NUM_HEAD, HIDDEN_DIM, DROPOUT, ACTIVATION_FUNCTION, IN_CHANNELS).to(device)
# x = torch.randn(512, 1, 28,28).to(device)
# print(model(x).shape)



# ------------ Load Data ------------- #
my_transforms = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=my_transforms, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train= False, transform=my_transforms, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------ INITIALIZE NETWORK ------------- #
model = VIT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMDED_DIM, NUM_ENCODERS, NUM_HEAD, HIDDEN_DIM, DROPOUT, ACTIVATION_FUNCTION, IN_CHANNELS).to(device)

# ------------ LOSS FUNCTION ------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------ Train Loop ------------- #
for epoch in range(EPOCHS):
    print("EPOCH", epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        # print("DATA SHAOE", data.shape)
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
