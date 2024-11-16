import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
import random
import matplotlib.pyplot as plt
from model import MNISTConvNet
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def send_metrics(epoch, loss, train_acc, val_acc):
    try:
        requests.post(
            'http://localhost:5000/update',
            json={
                'epoch': epoch,
                'loss': loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }
        )
    except Exception as e:
        logger.warning(f"Failed to send metrics: {str(e)}")


# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.2f} GB")

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

logger.info("Loading MNIST dataset...")
train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    './data', train=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
logger.info(f"Dataset loaded. Training samples: {len(train_dataset)}")

# Initialize model, loss, and optimizer
model = MNISTConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
logger.info("Model initialized and moved to device")

# Training loop
num_epochs = 10
logger.info("Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar for each epoch
    pbar = tqdm(
        train_loader,
        desc=f'Epoch {epoch+1}/{num_epochs}',
        ncols=100
    )
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar every batch
        avg_loss = running_loss / (batch_idx + 1)
        train_accuracy = 100. * correct / total
        
        # Calculate validation accuracy periodically
        if batch_idx % 100 == 99:
            val_accuracy = evaluate_model(model, test_loader, device)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'train_acc': f'{train_accuracy:.2f}%',
                'val_acc': f'{val_accuracy:.2f}%'
            })
            send_metrics(epoch, avg_loss, train_accuracy, val_accuracy)
    
    # Evaluate at epoch end
    val_accuracy = evaluate_model(model, test_loader, device)
    logger.info(
        f'Epoch {epoch+1} Summary - '
        f'Loss: {avg_loss:.4f}, '
        f'Train Accuracy: {train_accuracy:.2f}%, '
        f'Val Accuracy: {val_accuracy:.2f}%'
    )

logger.info("Training completed. Evaluating on random samples...")

# Evaluate on random samples
model.eval()
with torch.no_grad():
    indices = random.sample(range(len(test_dataset)), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        image, label = test_dataset[sample_idx]
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1, keepdim=True).item()
        
        axes[idx].imshow(image.squeeze(), cmap='gray')
        axes[idx].set_title(f'True: {label}\nPred: {pred}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('static/results.png')
    logger.info("Results saved as 'static/results.png'") 