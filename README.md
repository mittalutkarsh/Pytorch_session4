# MNIST CNN Model Comparison Tool

A real-time visualization tool for comparing different CNN architectures on the MNIST dataset. This tool allows users to train and compare two different CNN models simultaneously with configurable kernel sizes.

## Features

- Real-time training visualization
- Comparative analysis of two CNN models
- Configurable kernel sizes for each model
- Live loss and accuracy curves
- Training and validation accuracy tracking
- CUDA support for GPU acceleration
- Web-based monitoring interface

## Architecture

The project uses a 4-layer CNN with configurable kernel sizes:
- 4 convolutional layers with ReLU activation
- 2 max pooling layers
- Dropout for regularization
- Final fully connected layers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mittalutkarsh/Pytorch_session4.git
cd Pytorch_session4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. Start the visualization server:
```bash
python visualizer.py
```

2. Open your browser:
```
http://localhost:5000
```

3. Train two models with different configurations:
```bash
# Terminal 1
python train.py --model-id model1 --kernels 16 32 64 32

# Terminal 2
python train.py --model-id model2 --kernels 8 8 8 8
```

## Model Configuration

Each model can be configured with four numbers representing kernel sizes:
```python
[k1, k2, k3, k4]  # Number of kernels in each conv layer
```

Example configurations:
- Standard: `16,32,64,32`
- Uniform: `8,8,8,8`
- Increasing: `8,16,32,64`
- Decreasing: `64,32,16,8`

## Web Interface

The web interface provides:
- Configuration forms for both models
- Real-time loss comparison graph
- Real-time accuracy comparison graph (90-100% range)
- Training and validation accuracy curves
- Model configuration display in legends

## Project Structure

```
mnist_cnn/
├── README.md
├── HowTo.md
├── train.py          # Training script with CLI
├── model.py          # CNN model definition
├── visualizer.py     # Flask server for visualization
├── templates/        # HTML templates
│   └── index.html    # Web interface
└── static/          # Static files
    └── style.css    # CSS styling
```

## Technical Details

- Framework: PyTorch
- Visualization: Plotly.js
- Backend: Flask
- Training: CUDA-enabled (if available)
- Batch Size: 512
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Flask ≥ 2.0.0
- Other dependencies in requirements.txt

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Author

Utkarsh Mittal

## Acknowledgments

- PyTorch team for the framework
- MNIST dataset creators
- Flask team for the web framework
