# MNIST CNN Model Comparison Tool

This project implements a 4-layer CNN trained on MNIST with real-time visualization and model comparison capabilities.

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision flask numpy matplotlib tqdm requests plotly
```

## Project Structure

```bash
mnist_cnn/
├── HowTo.md
├── train.py
├── model.py
├── visualizer.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## How to Run

1. First, start the visualization server:
```bash
python visualizer.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. In the web interface, you'll see two model configuration forms:
   - Model 1 (default: 16,32,64,32 kernels)
   - Model 2 (default: 8,8,8,8 kernels)
   
4. To train the models, open two new terminal windows and run:

For Model 1:
```bash
python train.py --model-id model1 --kernels 16 32 64 32
```

For Model 2:
```bash
python train.py --model-id model2 --kernels 8 8 8 8
```

You will see:
- Real-time loss curves for both models
- Real-time accuracy curves (training and validation)
- Training metrics in the terminals
- Direct comparison of model performances

## Components

- `model.py`: Defines the configurable 4-layer CNN architecture
- `train.py`: Handles model training with command-line configuration
- `visualizer.py`: Flask server for real-time visualization
- `templates/index.html`: Web interface for model comparison
- `static/style.css`: Styling for the web interface

## Features

- CUDA support for GPU acceleration
- Real-time training visualization
- Comparative analysis of two models
- Configurable kernel sizes for each model
- Live updating loss and accuracy curves
- Training and validation accuracy tracking
- Web-based monitoring interface

## Model Configuration

Each model can be configured with four numbers representing the number of kernels in each convolutional layer:
- First number: kernels in first conv layer
- Second number: kernels in second conv layer
- Third number: kernels in third conv layer
- Fourth number: kernels in fourth conv layer

Example configurations:
- Standard: 16,32,64,32
- Uniform: 8,8,8,8
- Increasing: 8,16,32,64
- Decreasing: 64,32,16,8

## Troubleshooting

1. If the visualization server fails to start:
   - Check if port 5000 is already in use
   - Try changing the port in visualizer.py

2. If CUDA is not working:
   - Verify PyTorch is installed with CUDA support
   - Check your GPU drivers are up to date
   - Ensure enough GPU memory for both models

3. If plots are not updating:
   - Refresh the browser page
   - Check if both training scripts are running
   - Verify server connection in browser console

4. If training seems slow:
   - Reduce batch size in train.py
   - Check GPU utilization
   - Consider running models sequentially

## Notes

- Training progress is updated every 100 batches
- Web interface refreshes automatically every 2 seconds
- Training runs for 10 epochs by default
- Accuracy plot focuses on 90-100% range for better comparison
- Models can be started at different times
- Previous training data is cleared when starting a new training session
