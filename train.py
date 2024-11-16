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
from model import MNISTConvNet
import json


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


def get_optimizer(optimizer_name, model_params, learning_rate):
    """Get the specified optimizer."""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model_params, lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_params, lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_params, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_model(model_id, config):
    """Train a model with the specified configuration."""
    try:
        # Extract configuration
        kernel_config = config['kernel_config']
        optimizer_name = config['optimizer']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        num_epochs = config['epochs']
        dropout_rate = config['dropout_rate']
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Data preparation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            './data', train=False, transform=transform
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # Initialize model and optimizer
        model = MNISTConvNet(
            kernel_config, 
            dropout_rate=dropout_rate
        ).to(device)
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
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
                        send_metrics(epoch, avg_loss, train_acc, val_acc, model_id)
                    except Exception as e:
                        logger.warning(f"Failed to send metrics: {str(e)}")

                    running_loss = 0.0
                    correct = 0
                    total = 0

        # Notify server that training is completed
        send_metrics(epoch, avg_loss, train_acc, val_acc, model_id, status='completed')
        logger.info(f"Training completed for {model_id}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # Notify server about the error
        send_metrics(epoch, avg_loss, train_acc, val_acc, model_id, status='error')
        raise


def send_metrics(epoch, loss, train_acc, val_acc, model_id, status='running'):
    """Send metrics to visualization server."""
    try:
        requests.post(
            'http://localhost:5001/update',
            json={
                'model_id': model_id,
                'epoch': epoch,
                'loss': loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'status': status
            }
        )
    except Exception as e:
        logger.warning(f"Failed to send metrics: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument('--model-id', type=str, required=True, 
                      choices=['model1', 'model2'])
    parser.add_argument('--config', type=str, required=True,
                      help='JSON string containing model configuration')
    
    args = parser.parse_args()
    config = json.loads(args.config)
    train_model(args.model_id, config) 