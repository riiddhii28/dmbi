

## Algorithm Overview

| **Algorithm**                       | **Tab to Go**    | **Select From**        |
|-------------------------------------|------------------|------------------------|
| **Decision Tree (J48)**             | Classify         | Trees → J48            |
| **Bayesian Classifier (Naive Bayes)**| Classify         | Bayes → NaiveBayes     |
| **SVM (SMO)**                       | Classify         | Functions → SMO        |
| **Random Forest**                   | Classify         | Trees → RandomForest   |
| **Adaboost (LogitBoost)**           | Classify         | Meta → LogitBoost      |
| **Backpropagation (Multilayer Perceptron)** | Classify | Functions → MultilayerPerceptron |
| **K-Means Clustering**              | Cluster          | Clusterer → SimpleKMeans |
| **BIRCH Clustering**                | Cluster          | Clusterer → BIRCH      |
| **DBSCAN Clustering**               | Cluster          | Clusterer → DBSCAN     |
| **CLIQUE Clustering**               | Cluster          | Clusterer → CLIQUE     |
| **Apriori (Association Rules)**     | Associate        | Associate → Apriori    |
| **FP-Growth (Frequent Pattern Mining)** | Associate    | Associate → FP-Growth  |


## Notes

- The **Classify** tab is for classification tasks (e.g., J48, NaiveBayes, SMO).
- The **Cluster** tab is for clustering tasks (e.g., KMeans, DBSCAN, BIRCH).
- The **Associate** tab is for association rule mining and frequent pattern mining (e.g., Apriori, FP-Growth).
  
## Dataset Used

- For all classification and clustering tasks, **dataset.csv** is used. (a simple dataset with columns: ID, Age, Income, Student, CreditRating, BuysComputer).
- For **Apriori** and **FP-Growth**, **new_dataset.csv** dataset containing transactional data for association rule mining is used.
