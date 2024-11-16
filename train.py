"""Training script for MNIST CNN model comparison."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
import logging
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy on the provided data loader."""
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


def train_model(model_id, kernel_config):
    """Train a model with the specified configuration."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Notify server about training start
    requests.post(
        'http://localhost:5000/start_training',
        json={'model_id': model_id, 'kernel_config': kernel_config}
    )

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model
    model = nn.DataParallel(MNISTConvNet(kernel_config)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        running_loss = 0.0
        correct = 0
        total = 0

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

            if batch_idx % 100 == 99:
                avg_loss = running_loss / 100
                train_acc = 100. * correct / total
                val_acc = evaluate_model(model, test_loader, device)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'train_acc': f'{train_acc:.2f}%',
                    'val_acc': f'{val_acc:.2f}%'
                })
                
                try:
                    requests.post(
                        'http://localhost:5000/update',
                        json={
                            'model_id': model_id,
                            'epoch': epoch,
                            'loss': avg_loss,
                            'train_accuracy': train_acc,
                            'val_accuracy': val_acc
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to send metrics: {str(e)}")

                running_loss = 0.0
                correct = 0
                total = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        choices=['model1', 'model2']
    )
    parser.add_argument(
        '--kernels',
        type=int,
        nargs=4,
        required=True,
        help='Four integers for kernel configurations'
    )
    
    args = parser.parse_args()
    train_model(args.model_id, args.kernels) 