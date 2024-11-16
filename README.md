# MNIST CNN Training Visualizer

This project implements a 4-layer CNN trained on MNIST with real-time training visualization.

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision flask numpy matplotlib
```

## Project Structure

```
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

2. In a new terminal window, start the training:
   ```bash
   python train.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

You will see the live training progress with:
- Real-time loss curve
- Real-time accuracy curve
- Training metrics in the terminal

After training completes:
- The model will automatically evaluate 10 random MNIST images
- Results will be saved as 'static/results.png'
- You can view the results by opening the PNG file

## Components

- `model.py`: Defines the 4-layer CNN architecture
- `train.py`: Handles model training and evaluation
- `visualizer.py`: Flask server for real-time visualization
- `templates/index.html`: Web interface for monitoring training
- `static/style.css`: Styling for the web interface

## Features

- CUDA support for GPU acceleration
- Real-time training visualization
- Live updating loss and accuracy curves
- Automatic evaluation of random test samples
- Web-based monitoring interface

## Troubleshooting

### If the visualization server fails to start:
- Check if port 5000 is already in use
- Try changing the port in visualizer.py

### If CUDA is not working:
- Verify PyTorch is installed with CUDA support
- Check your GPU drivers are up to date

### If plots are not updating:
- Refresh the browser page
- Check if both server and training scripts are running

## Notes

- Training progress is updated every 100 batches
- The web interface refreshes automatically every 2 seconds
- Training runs for 10 epochs by default (can be modified in train.py)
- The model architecture uses 4 convolutional layers with ReLU activation
