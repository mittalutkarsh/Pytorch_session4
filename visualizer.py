from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
training_history = {
    'epochs': [], 
    'losses': [], 
    'train_accuracies': [],
    'val_accuracies': []
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/update', methods=['POST'])
def update():
    data = request.json
    training_history['epochs'].append(data['epoch'])
    training_history['losses'].append(data['loss'])
    training_history['train_accuracies'].append(data['train_accuracy'])
    training_history['val_accuracies'].append(data['val_accuracy'])
    return jsonify({'status': 'success'})


@app.route('/data')
def get_data():
    return jsonify(training_history)


if __name__ == '__main__':
    app.run(debug=True) 