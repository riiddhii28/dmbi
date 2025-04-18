import csv
import random
import math

# Load dataset and encode
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]
    return data

# Encode features into integers
def encode(row):
    age = int(row[1])
    income = {'Low': 0, 'Medium': 1, 'High': 2}[row[2]]
    student = 1 if row[3] == 'Yes' else 0
    credit = 1 if row[4] == 'Excellent' else 0
    label = 1 if row[5] == 'Yes' else -1
    return [age, income, student, credit], label

# Calculate Gini Impurity
def gini_impurity(y):
    total = len(y)
    if total == 0:
        return 0
    pos_count = sum([1 for label in y if label == 1])
    neg_count = total - pos_count
    p_pos = pos_count / total
    p_neg = neg_count / total
    return 1 - p_pos**2 - p_neg**2

# Split dataset based on feature and threshold
def split_dataset(X, y, feature, threshold):
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature] <= threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return left_X, left_y, right_X, right_y

# Decision Tree Recursion (Splitting on Best Feature)
def decision_tree(X, y, max_depth=5, min_size=10, depth=0):
    if len(set(y)) == 1:  # Pure set
        return y[0]
    
    if depth >= max_depth or len(X) <= min_size:
        return max(set(y), key=y.count)  # Return majority class
    
    best_gini = float('inf')
    best_split = None
    best_left = None
    best_right = None
    
    n_features = len(X[0])
    
    for feature in range(n_features):
        thresholds = list(set([x[feature] for x in X]))
        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)
            gini = gini_impurity(left_y) * len(left_y) + gini_impurity(right_y) * len(right_y)
            gini /= len(X)
            
            if gini < best_gini:
                best_gini = gini
                best_split = (feature, threshold)
                best_left = (left_X, left_y)
                best_right = (right_X, right_y)
    
    if best_split is None:
        return max(set(y), key=y.count)
    
    left_tree = decision_tree(best_left[0], best_left[1], max_depth, min_size, depth+1)
    right_tree = decision_tree(best_right[0], best_right[1], max_depth, min_size, depth+1)
    
    return (best_split, left_tree, right_tree)

# Predict using the decision tree
def predict_tree(tree, x):
    if isinstance(tree, int):
        return tree
    feature, threshold = tree[0]
    if x[feature] <= threshold:
        return predict_tree(tree[1], x)
    else:
        return predict_tree(tree[2], x)

# Random Forest Implementation
def random_forest(X, y, n_trees=5, max_depth=5, min_size=10):
    forest = []
    for _ in range(n_trees):
        # Bootstrap sampling
        sample_X, sample_y = [], []
        for _ in range(len(X)):
            idx = random.randint(0, len(X) - 1)
            sample_X.append(X[idx])
            sample_y.append(y[idx])
        
        # Train decision tree on bootstrap sample
        tree = decision_tree(sample_X, sample_y, max_depth, min_size)
        forest.append(tree)
    return forest

# Majority Voting from all trees
def vote(forest, x):
    predictions = [predict_tree(tree, x) for tree in forest]
    return max(set(predictions), key=predictions.count)

# Run
raw_data = load_csv("dataset.csv")
X, y = [], []
for row in raw_data:
    features, label = encode(row)
    X.append(features)
    y.append(label)

# Train Random Forest
forest = random_forest(X, y, n_trees=5)

# Test prediction
test_row = ['13', '30', 'High', 'Yes', 'Fair', '?']
x_test, _ = encode(test_row)
prediction = vote(forest, x_test)
label = "Yes" if prediction == 1 else "No"
print(f"Prediction for input {x_test}: {label}")
