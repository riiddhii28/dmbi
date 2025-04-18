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
    return [age, income, student, credit]

# Step 3: Calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

# Step 4: Create Clustering Feature (CF) structure
class CFNode:
    def __init__(self, point=None):
        self.points = [point] if point else []
        self.center = point if point else [0] * len(point)  # initialize with 0s for simplicity
        self.n_points = 1 if point else 0

    def update(self, point):
        self.points.append(point)
        self.center = [sum([p[i] for p in self.points]) / len(self.points) for i in range(len(self.center))]
        self.n_points += 1

# Step 5: Incrementally build the CF tree
def birch(X, threshold=1.0):
    # Start with an empty CF tree
    tree = [CFNode(X[0])]  # Start with the first point
    
    for x in X[1:]:
        added = False
        for node in tree:
            if euclidean_distance(x, node.center) < threshold:
                node.update(x)
                added = True
                break
        if not added:
            tree.append(CFNode(x))  # Create a new cluster node if not added
    
    return tree

# Step 6: Output the clusters formed by BIRCH
def print_clusters(tree):
    for idx, node in enumerate(tree):
        print(f"Cluster {idx + 1}:")
        print(f"  Centroid: {node.center}")
        print(f"  Points: {node.points}")

# Run
raw_data = load_csv("dataset.csv")
X = [encode(row) for row in raw_data]

# Perform BIRCH clustering
tree = birch(X, threshold=3.0)

# Output results
print_clusters(tree)
