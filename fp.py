import csv
from collections import defaultdict

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

# Step 3: Build a header table to store item counts
def build_header_table(transactions, min_support):
    item_count = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_count[item] += 1
    
    # Remove items that don't meet the min_support threshold
    header_table = {item: count for item, count in item_count.items() if count >= min_support}
    return header_table

# Step 4: Build the FP-Tree
def build_fp_tree(transactions, header_table):
    root = Node('null', 0)  # root node
    for transaction in transactions:
        ordered_items = [item for item in transaction if item in header_table]
        ordered_items.sort(key=lambda item: header_table[item], reverse=True)
        
        current_node = root
        for item in ordered_items:
            current_node = current_node.add_child(item)
    
    return root

# Node class for the FP-tree
class Node:
    def __init__(self, item, count):
        self.item = item
        self.count = count
        self.children = {}
        self.link = None
    
    def add_child(self, item):
        if item not in self.children:
            self.children[item] = Node(item, 0)
        self.children[item].count += 1
        return self.children[item]

# Step 5: Mine the FP-tree to find frequent itemsets
def mine_fp_tree(node, header_table, prefix, frequent_itemsets, min_support, transactions, min_support_threshold):
    if node.item != 'null':  # Skip the root node
        # Create the itemset with the current node's item
        frequent_itemsets.append((prefix + [node.item], node.count))

    # Traverse the children
    for child in node.children.values():
        mine_fp_tree(child, header_table, prefix + [child.item], frequent_itemsets, min_support, transactions, min_support_threshold)

    # Use the link to visit the next nodes for the same item
    if node.link:
        mine_fp_tree(node.link, header_table, prefix, frequent_itemsets, min_support, transactions, min_support_threshold)

# Updated FP-Growth Algorithm
def fp_growth(data, min_support_threshold):
    transactions = encode(data)
    header_table = build_header_table(transactions, min_support_threshold)
    
    if not header_table:
        return []

    # Build FP-tree
    fp_tree = build_fp_tree(transactions, header_table)

    # Mine the FP-tree to get frequent itemsets
    frequent_itemsets = []
    for item, count in header_table.items():
        # Create a conditional pattern base for each item
        conditional_pattern_base = []
        node = fp_tree.children.get(item)
        
        if node is None:
            continue

        while node:
            conditional_pattern_base.append([item] * node.count)
            node = node.link
        
        # Mine the conditional pattern base
        if conditional_pattern_base:
            mine_fp_tree(fp_tree, header_table, [item], frequent_itemsets, min_support_threshold, transactions, min_support_threshold)
    
    return frequent_itemsets

# Run FP-Growth
data = load_csv("new_dataset.csv")  # Load the dataset
min_support = 0.2  # Minimum support threshold (20%)

frequent_itemsets = fp_growth(data, min_support)

# Output results
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets:
    print(f"{set(itemset)}: Support = {count / len(data)}")
