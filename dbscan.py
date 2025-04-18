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

# Step 4: Find neighbors of a point
def region_query(X, point_idx, epsilon):
    neighbors = []
    for i in range(len(X)):
        if euclidean_distance(X[point_idx], X[i]) < epsilon:
            neighbors.append(i)
    return neighbors

# Step 5: Expand the cluster
def expand_cluster(X, labels, point_idx, neighbors, cluster_id, epsilon, min_pts):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:  # If it's noise, make it part of the current cluster
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:  # If it's unvisited, visit it
            labels[neighbor_idx] = cluster_id
            neighbor_neighbors = region_query(X, neighbor_idx, epsilon)
            if len(neighbor_neighbors) >= min_pts:
                neighbors += neighbor_neighbors
        i += 1

# Step 6: DBSCAN algorithm
def dbscan(X, epsilon, min_pts):
    labels = [0] * len(X)  # 0 means unvisited, -1 means noise, other values are cluster IDs
    cluster_id = 0
    for point_idx in range(len(X)):
        if labels[point_idx] != 0:
            continue
        neighbors = region_query(X, point_idx, epsilon)
        if len(neighbors) < min_pts:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(X, labels, point_idx, neighbors, cluster_id, epsilon, min_pts)
    return labels

# Run
raw_data = load_csv("dataset.csv")
X = [encode(row) for row in raw_data]

# Perform DBSCAN clustering
epsilon = 3.0  # Set a suitable epsilon
min_pts = 2    # Minimum points to form a cluster
labels = dbscan(X, epsilon, min_pts)

# Output results
for i, label in enumerate(labels):
    print(f"Point {i + 1}: {X[i]} => Cluster {label}")
