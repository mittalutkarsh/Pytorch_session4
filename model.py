"""
MNIST CNN Model Definition
This module defines the CNN architecture for MNIST digit classification.
"""

import torch.nn as nn


class MNISTConvNet(nn.Module):
    """
    A 4-layer Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 4 convolutional layers with ReLU activation
    - 2 max pooling layers
    - 2 fully connected layers with dropout
    """
    
    def __init__(self, kernel_config, dropout_rate=0.5):
        """
        Initialize the network with configurable kernel sizes and dropout.
        
        Args:
            kernel_config (list): List of integers specifying number of kernels
                                for each conv layer [k1, k2, k3, k4]
            dropout_rate (float): Dropout probability (default: 0.5)
        """
        super(MNISTConvNet, self).__init__()
        
        k1, k2, k3, k4 = kernel_config
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, k1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(k1, k2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(k2, k3, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Fourth conv layer
            nn.Conv2d(k3, k4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(k4 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x 