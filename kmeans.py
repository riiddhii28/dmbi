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

# Step 4: Initialize centroids
def initialize_centroids(X, k):
    return random.sample(X, k)

# Step 5: Assign points to the nearest centroid
def assign_clusters(X, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append(x)
    return clusters

# Step 6: Recompute centroids
def recompute_centroids(clusters):
    return [list(map(lambda x: sum(x) / len(x), zip(*cluster))) for cluster in clusters]

# Step 7: K-Means algorithm
def kmeans(X, k, max_iterations=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(X, centroids)
        new_centroids = recompute_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

# Run
raw_data = load_csv("dataset.csv")
X = [encode(row) for row in raw_data]

# Perform K-Means clustering
k = 3
centroids, clusters = kmeans(X, k)

# Output results
print(f"Centroids: {centroids}")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")
