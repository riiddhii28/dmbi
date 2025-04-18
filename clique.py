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

# Step 4: Create grid for CLIQUE
def create_grid(X, grid_size=2):
    grid = {}
    for point in X:
        # Create a grid key by dividing coordinates by grid_size
        grid_key = tuple(int(p / grid_size) for p in point)
        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append(point)
    return grid

# Step 5: Identify dense regions in the grid
def find_dense_regions(grid, min_points):
    dense_regions = []
    for key, points in grid.items():
        if len(points) >= min_points:
            dense_regions.append(key)
    return dense_regions

# Step 6: Merge neighboring regions
def merge_regions(dense_regions, grid_size):
    clusters = []
    visited = set()
    for region in dense_regions:
        if region not in visited:
            cluster = [region]
            visited.add(region)
            # Merge neighboring regions (8-connected neighbors)
            neighbors = [
                (region[0] + dx, region[1] + dy) 
                for dx in [-1, 0, 1] 
                for dy in [-1, 0, 1] 
                if (dx != 0 or dy != 0)
            ]
            for neighbor in neighbors:
                if neighbor in dense_regions:
                    cluster.append(neighbor)
                    visited.add(neighbor)
            clusters.append(cluster)
    return clusters

# Step 7: CLIQUE algorithm
def clique(X, grid_size=2, min_points=2):
    grid = create_grid(X, grid_size)
    dense_regions = find_dense_regions(grid, min_points)
    clusters = merge_regions(dense_regions, grid_size)
    return clusters

# Run
raw_data = load_csv("dataset.csv")
X = [encode(row) for row in raw_data]

# Perform CLIQUE clustering
clusters = clique(X, grid_size=2, min_points=2)

# Output results
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")
