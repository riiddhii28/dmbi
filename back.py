import csv
import random
import math

# Step 1: Load dataset and encode
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]
    return data

# Step 2: Encode features into integers
def encode(row):
    age = int(row[1])
    income = {'Low': 0, 'Medium': 1, 'High': 2}[row[2]]
    student = 1 if row[3] == 'Yes' else 0
    credit = 1 if row[4] == 'Excellent' else 0
    label = 1 if row[5] == 'Yes' else 0
    return [age, income, student, credit], label

# Step 3: Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 4: Initialize weights
def initialize_weights(n_inputs, n_hidden, n_outputs):
    weights_input_hidden = [[random.random() for _ in range(n_hidden)] for _ in range(n_inputs)]
    weights_hidden_output = [random.random() for _ in range(n_hidden)]
    return weights_input_hidden, weights_hidden_output

# Step 5: Forward propagation
def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_layer = [0] * len(weights_input_hidden[0])
    for i in range(len(X)):
        for j in range(len(weights_input_hidden[i])):
            hidden_layer[j] += X[i] * weights_input_hidden[i][j]
    hidden_layer = [sigmoid(x) for x in hidden_layer]
    
    output = sum([hidden_layer[i] * weights_hidden_output[i] for i in range(len(hidden_layer))])
    output = sigmoid(output)
    return hidden_layer, output

# Step 6: Backpropagation and weight update
def backpropagate(X, y, weights_input_hidden, weights_hidden_output, learning_rate=0.01):
    hidden_layer, output = forward_propagation(X, weights_input_hidden, weights_hidden_output)
    
    # Calculate output error
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)
    
    # Calculate hidden layer error
    hidden_errors = [output_delta * weights_hidden_output[i] for i in range(len(weights_hidden_output))]
    hidden_deltas = [hidden_errors[i] * sigmoid_derivative(hidden_layer[i]) for i in range(len(hidden_layer))]
    
    # Update weights
    for i in range(len(weights_hidden_output)):
        weights_hidden_output[i] += learning_rate * output_delta * hidden_layer[i]
    
    for i in range(len(weights_input_hidden)):
        for j in range(len(weights_input_hidden[i])):
            weights_input_hidden[i][j] += learning_rate * hidden_deltas[j] * X[i]

    return weights_input_hidden, weights_hidden_output

# Step 7: Train neural network
def train_nn(X, y, n_hidden, n_epochs=1000, learning_rate=0.01):
    n_inputs = len(X[0])
    n_outputs = 1
    weights_input_hidden, weights_hidden_output = initialize_weights(n_inputs, n_hidden, n_outputs)
    
    for epoch in range(n_epochs):
        for i in range(len(X)):
            weights_input_hidden, weights_hidden_output = backpropagate(X[i], y[i], weights_input_hidden, weights_hidden_output, learning_rate)
    
    return weights_input_hidden, weights_hidden_output

# Step 8: Predict
def predict(X, weights_input_hidden, weights_hidden_output):
    hidden_layer, output = forward_propagation(X, weights_input_hidden, weights_hidden_output)
    return 1 if output >= 0.5 else 0

# Run
raw_data = load_csv("dataset.csv")
X, y = [], []
for row in raw_data:
    features, label = encode(row)
    X.append(features)
    y.append(label)

# Train neural network
n_hidden = 3
weights_input_hidden, weights_hidden_output = train_nn(X, y, n_hidden)

# Test prediction
test_row = ['13', '30', 'High', 'Yes', 'Fair', '?']
x_test, _ = encode(test_row)
prediction = predict(x_test, weights_input_hidden, weights_hidden_output)
label = "Yes" if prediction == 1 else "No"
print(f"Prediction for input {x_test}: {label}")
