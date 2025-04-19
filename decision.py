import math
import csv

# Load dataset
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        dataset = [row for row in reader]
    return headers, dataset

# Calculate entropy
def entropy(rows):
    total = len(rows)
    if total == 0:
        return 0
    label_counts = {}
    for row in rows:
        label = row[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    ent = 0
    for label in label_counts:
        p = label_counts[label] / total
        ent -= p * math.log2(p)
    return ent

# Split dataset on attribute index
def split_dataset(dataset, index, value):
    return [row for row in dataset if row[index] == value]

# Find unique values for an attribute
def unique_vals(rows, index):
    return list(set([row[index] for row in rows]))

# Choose best feature to split
def best_feature(rows):
    base_entropy = entropy(rows)
    best_info_gain = -1
    best_index = -1
    for i in range(1, len(rows[0]) - 1):  # Skip ID and target
        vals = unique_vals(rows, i)
        new_entropy = 0
        for val in vals:
            subset = split_dataset(rows, i, val)
            new_entropy += (len(subset) / len(rows)) * entropy(subset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_index = i
    return best_index

# Build decision tree recursively
def build_tree(dataset, headers):
    labels = [row[-1] for row in dataset]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(headers) == 1:  # Only target left
        return max(set(labels), key=labels.count)

    best_idx = best_feature(dataset)
    best_attr = headers[best_idx]
    tree = {best_attr: {}}

    values = unique_vals(dataset, best_idx)
    for val in values:
        subset = split_dataset(dataset, best_idx, val)
        reduced_headers = headers[:best_idx] + headers[best_idx+1:]
        reduced_subset = [row[:best_idx] + row[best_idx+1:] for row in subset]
        tree[best_attr][val] = build_tree(reduced_subset, reduced_headers)

    return tree

# Predict for a new instance
def classify(tree, headers, instance):
    if not isinstance(tree, dict):
        return tree  # Leaf node

    attr = next(iter(tree))  # Root attribute
    attr_index = headers.index(attr)
    value = instance[attr_index]

    if value in tree[attr]:
        subtree = tree[attr][value]
        reduced_headers = headers[:attr_index] + headers[attr_index+1:]
        reduced_instance = instance[:attr_index] + instance[attr_index+1:]
        return classify(subtree, reduced_headers, reduced_instance)
    else:
        return "Unknown"

# Run
headers, data = load_csv("dataset.csv")
tree = build_tree(data, headers)

print("Decision Tree:")
print(tree)

# New instance for prediction
new_instance = ["13", "35", "Medium", "No", "Fair"]  # Without target
result = classify(tree, headers[:-1], new_instance)  # Exclude BuysComputer from headers
print(f"\nPrediction for new instance {new_instance}: {result}")
