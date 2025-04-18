import csv

# Load dataset
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data

# Count class frequencies
def class_counts(data):
    counts = {}
    for row in data:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return counts

# Count conditional frequencies for attribute=value given class
def attr_given_class(data, attr_index, attr_value, class_value):
    count = 0
    total = 0
    for row in data:
        if row[-1] == class_value:
            total += 1
            if row[attr_index] == attr_value:
                count += 1
    return count / total if total > 0 else 0

# Predict class for a new input row
def predict_naive_bayes(data, headers, input_row):
    labels = class_counts(data)
    total_rows = len(data)
    probs = {}

    for label in labels:
        # Start with prior probability
        prob = labels[label] / total_rows
        for i in range(1, len(input_row) - 1):  # Skip ID and target
            attr_value = input_row[i]
            prob *= attr_given_class(data, i, attr_value, label)
        probs[label] = prob

    # Return the class with the highest probability
    return max(probs, key=probs.get)

# Example run
headers, data = load_csv("dataset.csv")

# Predict on a test row: ID=13, Age=30, Income=High, Student=No, CreditRating=Fair
test_row = ['13', '30', 'High', 'No', 'Fair', '?']
prediction = predict_naive_bayes(data, headers, test_row)
print(f"Prediction for input {test_row[1:-1]}: {prediction}")
