import csv
from itertools import combinations

# Step 1: Load dataset and encode
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]
    return data

# Step 2: Create a list of transactions (items purchased in each transaction)
def encode(data):
    transactions = []
    for row in data:
        transaction = set()
        for i, value in enumerate(row[1:]):
            if int(value) == 1:  # if the item is purchased
                transaction.add(i)
        transactions.append(transaction)
    return transactions

# Step 3: Generate candidate itemsets of length 1, 2, 3, etc.
def create_candidates(transactions, length):
    candidates = set()
    for transaction in transactions:
        for itemset in combinations(transaction, length):
            candidates.add(frozenset(itemset))
    return candidates

# Step 4: Calculate support for itemsets
def calculate_support(candidates, transactions):
    support_count = {}
    for candidate in candidates:
        support_count[candidate] = sum(1 for transaction in transactions if candidate.issubset(transaction))
    return support_count

# Step 5: Prune candidates that do not meet minimum support threshold
def prune_candidates(support_count, min_support, num_transactions):
    return {itemset: count for itemset, count in support_count.items() if count / num_transactions >= min_support}

# Step 6: Generate association rules from frequent itemsets
def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules

# Step 7: Apriori algorithm
def apriori(data, min_support, min_confidence):
    transactions = encode(data)
    num_transactions = len(transactions)
    
    # Step 1: Find frequent itemsets of length 1
    candidate_1 = create_candidates(transactions, 1)
    support_1 = calculate_support(candidate_1, transactions)
    frequent_1 = prune_candidates(support_1, min_support, num_transactions)
    
    # Step 2: Find frequent itemsets of length 2, 3, ...
    frequent_itemsets = frequent_1
    k = 2
    while True:
        candidate_k = create_candidates(transactions, k)
        support_k = calculate_support(candidate_k, transactions)
        frequent_k = prune_candidates(support_k, min_support, num_transactions)
        
        if not frequent_k:
            break
        
        frequent_itemsets.update(frequent_k)
        k += 1
    
    # Step 3: Generate association rules
    rules = generate_rules(frequent_itemsets, min_confidence)
    
    return frequent_itemsets, rules

# Run Apriori
data = load_csv("new_dataset.csv")  # Load the dataset
min_support = 0.2  # Minimum support threshold (20%)
min_confidence = 0.5  # Minimum confidence threshold (50%)

frequent_itemsets, rules = apriori(data, min_support, min_confidence)

# Output results
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{set(itemset)}: Support = {count / len(data)}")

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {set(antecedent)} => {set(consequent)} with confidence {confidence}")
