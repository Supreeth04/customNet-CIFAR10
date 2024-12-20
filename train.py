import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from models.network import CIFAR10Net
from utils.transforms import get_transforms
from datasets.cifar10_dataset import CIFAR10Dataset
from tqdm import tqdm
import numpy as np
from torchsummary import summary
from torch.amp import autocast, GradScaler

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return accuracy, avg_loss

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_epochs = 35
    batch_size = 128
    accumulation_steps = 4
    learning_rate = 0.001
    learning_rate *= (batch_size * accumulation_steps) / 128
    
    # Mean and std for CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Get transforms
    train_transform, test_transform = get_transforms(mean, std)
    
    # Load CIFAR10 dataset
    train_data = datasets.CIFAR10(root='./data', train=True, download=True)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Create custom datasets
    train_dataset = CIFAR10Dataset(train_data, transform=train_transform)
    test_dataset = CIFAR10Dataset(test_data, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    
    # Initialize the model
    model = CIFAR10Net().to(device)
    
    # Print model summary
    print("\nModel Summary:")
    summary(model, input_size=(3, 32, 32))
    print("\n")
    
    # Print model parameters
    total_params, trainable_params = get_model_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Add OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Lists to store metrics
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        optimizer.zero_grad()  # Zero gradients at start of epoch
        for idx, (images, labels) in enumerate(train_loop):
            images = images.to(device)
            labels = labels.to(device)
            
            if device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
            
            if (idx + 1) % accumulation_steps == 0:
                if device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item())
            
            # Step the scheduler
            scheduler.step()
        
        # Calculate training metrics
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        
        # Evaluation phase
        test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    # Print best results
    best_test_accuracy = max(test_accuracies)
    best_epoch = np.argmax(test_accuracies) + 1
    print(f'\nBest Test Accuracy: {best_test_accuracy:.2f}% at epoch {best_epoch}')

if __name__ == '__main__':
    train_model() 
