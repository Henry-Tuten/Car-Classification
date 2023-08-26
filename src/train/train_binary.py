import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from Dataset import CustomDataset
from tqdm import tqdm
import sys
sys.path.append('src/')

from models.CNN import CarNet


# Check if CUDA is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("GPU Found")
#-------------------Initialize Dataset---------------------------------

# Data Augmentation
transform_train = transforms.Compose([

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Initialize dataset and dataloader
train_dataset = datasets.ImageFolder(root='data/processed/train_images/', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)

test_dataset = datasets.ImageFolder(root='data/processed/val_images/', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)



#---------------------------------------------------------------------
dataiter = iter(train_loader)
images, labels = next(iter(train_loader))
print(images.shape)
#-----------------Initialize Model, Loss, and Optimizer---------------

# Load the pre-trained InceptionV3 model
model = CarNet()

model = model.to(device)

# Define Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#---------------------------------------------------------------------

#-----------------------Training Loop---------------------------------

num_epochs = 3


for epoch in range(num_epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(data)  # Get model outputs
        loss = criterion(outputs.squeeze(), targets.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        train_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()  # Convert probabilities to predictions
        total_train += targets.size(0)
        correct_train += (predicted == targets.float()).sum().item()
        
    average_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets.float())
            
            val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total_val += targets.size(0)
            correct_val += (predicted == targets.float()).sum().item()
    
    average_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct_val / total_val
    
    print(f"\nEpoch {epoch+1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

# Save the model's state_dict at the end of training
torch.save(model.state_dict(), "model_checkpoint.pth")
#--------------------------------------------------------------------------