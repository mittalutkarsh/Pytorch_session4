from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__)

# Store configurations and histories for both models
model_configs = {
    'model1': None,
    'model2': None
}

training_histories = {
    'model1': {
        'epochs': [], 
        'losses': [], 
        'train_accuracies': [],
        'val_accuracies': []
    },
    'model2': {
        'epochs': [], 
        'losses': [], 
        'train_accuracies': [],
        'val_accuracies': []
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json
    model_id = data['model_id']
    kernel_config = data['kernel_config']
    model_configs[model_id] = kernel_config
    
    # Clear previous training history for this model
    training_histories[model_id] = {
        'epochs': [], 
        'losses': [], 
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    return jsonify({'status': 'success'})

@app.route('/update', methods=['POST'])
def update():
    data = request.json
    model_id = data['model_id']
    history = training_histories[model_id]
    
    history['epochs'].append(data['epoch'])
    history['losses'].append(data['loss'])
    history['train_accuracies'].append(data['train_accuracy'])
    history['val_accuracies'].append(data['val_accuracy'])
    
    return jsonify({'status': 'success'})

@app.route('/data')
def get_data():
    return jsonify({
        'configs': model_configs,
        'histories': training_histories
    })

if __name__ == '__main__':
    app.run(debug=True) 