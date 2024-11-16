from flask import Flask, render_template, jsonify, request
import json
import subprocess
import sys
import psutil
import time
import logging

app = Flask(__name__)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Store configurations and histories for both models
model_configs = {
    'model1': None,
    'model2': None
}

model_processes = {
    'model1': None,
    'model2': None
}

training_histories = {
    'model1': {
        'epochs': [], 
        'losses': [], 
        'train_accuracies': [],
        'val_accuracies': [],
        'status': 'idle',
        'start_time': None,
        'end_time': None,
        'training_time': None
    },
    'model2': {
        'epochs': [], 
        'losses': [], 
        'train_accuracies': [],
        'val_accuracies': [],
        'status': 'idle',
        'start_time': None,
        'end_time': None,
        'training_time': None
    }
}


def check_process_status(process):
    """Check if a process is still running."""
    if process is None:
        return False
    return psutil.pid_exists(process.pid)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        data = request.json
        model_id = data['model_id']
        config = data['config']
        
        # Check if model is already running
        if (model_processes[model_id] is not None and 
            check_process_status(model_processes[model_id])):
            return jsonify({
                'status': 'error',
                'message': f'{model_id} is already running'
            }), 400
        
        model_configs[model_id] = config
        
        # Clear previous training history for this model
        training_histories[model_id] = {
            'epochs': [], 
            'losses': [], 
            'train_accuracies': [],
            'val_accuracies': [],
            'status': 'running',
            'start_time': time.time(),
            'end_time': None,
            'training_time': None
        }
        
        # Start training process
        config_str = json.dumps(config)
        python_executable = sys.executable
        process = subprocess.Popen([
            python_executable,
            'train.py',
            '--model-id', model_id,
            '--config', config_str
        ])
        
        model_processes[model_id] = process
        
        return jsonify({
            'status': 'success',
            'message': f'Started training {model_id}'
        })
    except Exception as e:
        training_histories[model_id]['status'] = 'error'
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.json
        model_id = data['model_id']
        history = training_histories[model_id]
        
        # Update status if provided
        if 'status' in data:
            history['status'] = data['status']
            if data['status'] == 'error' and 'error_message' in data:
                logger.error(f"Model {model_id} error: {data['error_message']}")
            elif data['status'] == 'completed':
                history['end_time'] = time.time()
                history['training_time'] = history['end_time'] - history['start_time']
        
        # Update metrics if provided
        if all(key in data for key in ['epoch', 'loss', 'train_accuracy', 'val_accuracy']):
            history['epochs'].append(data['epoch'])
            history['losses'].append(data['loss'])
            history['train_accuracies'].append(data['train_accuracy'])
            history['val_accuracies'].append(data['val_accuracy'])
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error in update: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/status')
def get_status():
    """Get the current status of all models."""
    status = {}
    for model_id in ['model1', 'model2']:
        process = model_processes[model_id]
        if process is not None:
            is_running = check_process_status(process)
            if not is_running:
                training_histories[model_id]['status'] = 'completed'
                model_processes[model_id] = None
        status[model_id] = training_histories[model_id]['status']
    return jsonify(status)


@app.route('/data')
def get_data():
    return jsonify({
        'configs': model_configs,
        'histories': training_histories
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001) 