import csv

def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data

# Get conditional probability P(A = a | B = b, C = c)
def prob_buys_given_student_credit(data, student_val, credit_val, target_val):
    match_count = 0
    target_match = 0
    for row in data:
        if row[3] == student_val and row[4] == credit_val:
            match_count += 1
            if row[-1] == target_val:
                target_match += 1
    if match_count == 0:
        return 0
    return target_match / match_count

# Run
headers, data = load_csv("dataset.csv")
p_yes = prob_buys_given_student_credit(data, "Yes", "Fair", "Yes")
p_no = prob_buys_given_student_credit(data, "Yes", "Fair", "No")

print(f"P(BuysComputer = Yes | Student = Yes, CreditRating = Fair): {p_yes:.2f}")
print(f"P(BuysComputer = No | Student = Yes, CreditRating = Fair): {p_no:.2f}")

# Choose highest
prediction = "Yes" if p_yes > p_no else "No"
print(f"Predicted class: {prediction}")
