import csv
import math

# Step 1: Load and encode dataset
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]
    return data

# Step 2: Encode features to integers
def encode(row):
    age = int(row[1])
    income = {'Low': 0, 'Medium': 1, 'High': 2}[row[2]]
    student = 1 if row[3] == 'Yes' else 0
    credit = 1 if row[4] == 'Excellent' else 0
    label = 1 if row[5] == 'Yes' else -1
    return [age, income, student, credit], label

# Step 3: Decision stump on one feature
def train_stump(X, y, weights):
    n_features = len(X[0])
    best_feature = 0
    best_threshold = None
    best_polarity = 1
    min_error = float('inf')

    for feature in range(n_features):
        thresholds = list(set([x[feature] for x in X]))
        for thresh in thresholds:
            for polarity in [1, -1]:
                error = 0
                for i in range(len(X)):
                    prediction = polarity * (1 if X[i][feature] <= thresh else -1)
                    if prediction != y[i]:
                        error += weights[i]
                if error < min_error:
                    min_error = error
                    best_feature = feature
                    best_threshold = thresh
                    best_polarity = polarity

    return best_feature, best_threshold, best_polarity, min_error

# Step 4: AdaBoost main loop
def adaboost(X, y, T=5):
    n = len(X)
    weights = [1/n] * n
    classifiers = []

    for t in range(T):
        feat, thresh, polarity, error = train_stump(X, y, weights)
        eps = 1e-10
        alpha = 0.5 * math.log((1 - error + eps) / (error + eps))

        # Save this weak learner
        classifiers.append((feat, thresh, polarity, alpha))

        # Update weights
        for i in range(n):
            prediction = polarity * (1 if X[i][feat] <= thresh else -1)
            weights[i] *= math.exp(-alpha * y[i] * prediction)

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

    return classifiers

# Step 5: Prediction
def predict(x, classifiers):
    final = 0
    for feat, thresh, polarity, alpha in classifiers:
        prediction = polarity * (1 if x[feat] <= thresh else -1)
        final += alpha * prediction
    return 1 if final > 0 else -1

# Run
raw_data = load_csv("dataset.csv")
X, y = [], []
for row in raw_data:
    features, label = encode(row)
    X.append(features)
    y.append(label)

classifiers = adaboost(X, y, T=5)

# Test prediction
test_row = ['13', '30', 'High', 'Yes', 'Fair', '?']
x_test, _ = encode(test_row)
pred = predict(x_test, classifiers)
print(f"Prediction for input {x_test}: {'Yes' if pred == 1 else 'No'}")
