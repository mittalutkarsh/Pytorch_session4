import sys
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
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(model_id, kernel_config):
    # Notify server about training start
    requests.post(
        'http://localhost:5000/start_training',
        json={'model_id': model_id, 'kernel_config': kernel_config}
    )
    
    # Rest of your training code...
    # Make sure to include model_id in send_metrics calls:
    def send_metrics(epoch, loss, train_acc, val_acc):
        try:
            requests.post(
                'http://localhost:5000/update',
                json={
                    'model_id': model_id,
                    'epoch': epoch,
                    'loss': loss,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send metrics: {str(e)}")

    # Your existing training code here...
    # Just make sure to use the kernel_config when creating the model:
    model = MNISTConvNet(kernel_config).to(device)
    
    # Rest of the training loop remains the same...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str, required=True, 
                      choices=['model1', 'model2'])
    parser.add_argument('--kernels', type=int, nargs=4, required=True,
                      help='Four integers for kernel configurations')
    
    args = parser.parse_args()
    train_model(args.model_id, args.kernels) 