import csv
import random

# Load and encode dataset
def load_dataset(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip headers
        data = [row for row in reader]
    
    features = []
    labels = []

    for row in data:
        x, y = encode(row)
        features.append(x)
        labels.append(y)
    
    return features, labels

# Encode categorical values to numbers
def encode(row):
    age = int(row[1])
    income = {'Low': 0, 'Medium': 1, 'High': 2}[row[2]]
    student = 1 if row[3] == 'Yes' else 0
    credit = 1 if row[4] == 'Excellent' else 0
    buys = 1 if row[5] == 'Yes' else -1
    return [age, income, student, credit], buys

# Train SVM using basic gradient descent
def train_svm(X, y, lr=0.001, lambda_param=0.01, epochs=1000):
    w = [0] * len(X[0])
    b = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]
            condition = y_i * (dot(w, x_i) + b) >= 1

            if condition:
                # Only regularization
                for j in range(len(w)):
                    w[j] -= lr * (2 * lambda_param * w[j])
            else:
                for j in range(len(w)):
                    w[j] -= lr * (2 * lambda_param * w[j] - y_i * x_i[j])
                b -= lr * y_i
    return w, b

# Dot product helper
def dot(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

# Predict new input
def predict(x, w, b):
    result = dot(w, x) + b
    return 1 if result >= 0 else -1

# Run
X, y = load_dataset("dataset.csv")
weights, bias = train_svm(X, y)

# Predict example: Age=30, Income=High, Student=Yes, Credit=Fair
test_row = ['13', '30', 'High', 'Yes', 'Fair', '?']
x_test, _ = encode(test_row)
prediction = predict(x_test, weights, bias)
label = "Yes" if prediction == 1 else "No"

print(f"Prediction for input {x_test}: {label}")
