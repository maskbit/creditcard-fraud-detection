# Machine Learning for Credit Card Fraud Detection

# Introduction

Credit card fraud has become one of the most rapidly evolving financial crimes in the world today, presenting significant challenges for individuals and businesses globally. The increasing use of credit cards for online and offline transactions has created an opportunity for fraudsters to exploit vulnerabilities in payment systems, using highly sophisticated methods. 

The aim of this project is to construct machine learning models employing various prediction algorithms and assess their outcomes. The primary goal is to create models capable of effectively detecting fraudulent transactions and rejecting them. Utilizing machine learning algorithms such as Logistic Regression, Support Vector Machines (SVM), and Random Forest, the results will be analyzed to determine the most effective model. Additionally, an artificial neural network model will be developed and contrasted with the conventional models to evaluate its performance and accuracy.

# Dataset

The dataset utilized in this project is derived from IBM's synthetic credit card data, which can be accessed [here.](https://ibm.box.com/v/tabformer-data) The dataset is extensive, comprising over 24 million credit card transactions. However, for the purpose of this project, only a sample of 50,000 transactions was selected from the original dataset.


**Exploratory data analysis:**

The dataset consists of 11 features alongside a target feature, which is binary indicating whether a transaction is fraudulent. Two numeric features were initially unidentified and retained as such. Upon initial examination, null values were observed in 'Merchant State', 'Zip', and 'Errors?' features. The 'Amount' feature underwent conversion to a numeric field by removing the dollar symbol, followed by the application of a logarithmic function. The features 'Year', 'Month', 'Day', and 'Time' were merged into a Date field using the Pandas datetime function. Subsequent data analysis led to the removal of 'Card', 'Error', 'Merchant City', and 'Merchant State' features as they were deemed irrelevant. However, in practical scenarios, the card number, when encrypted, could serve as a unique and potentially valuable feature. Additionally, features like 'Merchant State', 'City', or other geo-locations could offer insights into transaction origins. For this project, solely the zip code sufficed as a location feature. After the dataset underwent cleaning, it appeared as follows:

| Feature          | Description                                          |
|------------------|------------------------------------------------------|
| _Unnamed: 0.1_   | unnamed feature #1                                   |
| _Unnamed: 0_     | _unnamed feature #2_                                 |
| _User_           | _User id _                                           |
| _Amount_         | _Dollar Amount of the transaction_                   |
| _Use Chip_       | _Type of the credit card transaction_                |
| _Merchant Name_  | _Name of the Merchant who initiated the transaction_ |
| _Zip_            | _Zip code where the transaction was initiated_       |
| _MCC_            | _Merchant Categorization Code_                       |
| Date             | Date of the transaction                              |
| Fraud            | Target ( Fradulent or not)                           |


**Data Visualization:**

![Types of Transactions](https://github.com/maskbit/creditcard-fraud-detection/blob/main/Images/ChipSwipeOnlineTransactions.png)

When analyzing solely in-person transactions and disregarding online ones, it becomes evident that California, Texas, Florida, New York, and Ohio stand out with the highest transaction volumes. Moreover, it is noteworthy that these states exhibit a notable concentration on specific MCCs (Merchant Category Codes), including but not limited to 5411, 5499, 5541, 5812, 5912, and 5300. These MCCs predominantly represent various essential establishments such as grocery stores or supermarkets, specialty markets, convenience stores, gas/service stations, restaurants, pharmacies, and wholesale clubs.

![Transactions by State](https://github.com/maskbit/creditcard-fraud-detection/blob/main/Images/TotalTransactions.png)

# Models Evaluation and Comparison

After additional research and market analysis, various machine learning models such as random forest, logistic regression, and support vector machines were evaluated for credit card fraud detection. To determine the optimal hyperparameters for these models, the dataset underwent grid search cross-validation. The best estimator and corresponding scores were obtained. Additionally, the dataset was also processed through an artificial neural network for comparison. Below, the results of the models, including the best estimator parameters and scores, are tabulated.

**Random Forest:**

Random Forest was tested, and the optimal hyperparameters identified were criterion: entropy, max_depth: 40, max_features: sqrt, min_samples_leaf: 2, min_samples_split: 5, and n_estimators: 200. The highest ROC AUC score obtained was 0.92.

**Logistic Regression:**

Logistic Regression was experimented with, and the optimal hyperparameters identified were C: 29.76, max_iter: 100, and penalty: l2. The highest ROC AUC score obtained was 0.86.

**Support Vector Machines:**

Support Vector Machines were experimented with, and the optimal hyperparameters identified were C: 0.1, degree: 0, gamma: 0.1, and kernel: linear. The highest ROC AUC score achieved was 0.82. Because of the substantial computational demands associated with SVM for kernels such as 'RBF' and 'Poly', it was constrained to employing only the linear kernel.

**Artificial Neural Network:**

Ultimately, the Artificial Neural Network with 100 hidden layers utilizing ReLU activation, one output layer employing Sigmoid activation, 20 epochs, a batch size of 10, and the Adam optimizer surpassed all alternative models, achieving the highest ROC AUC score of 0.99.

![ANN Model AUC](https://github.com/maskbit/creditcard-fraud-detection/blob/main/Images/ANNmodelAUC.png)


| MODEL | Hyper Parameters |  Score |
|:---:|:---:|:---:|
| Logistic Regression | C: 29.76, max_iter: 100, penalty: l2 | 0.86 |
| Support Vector Machines | C: 0.1, degree: 0, gamma: 0.1, kernel: linear | 0.82 |
| Random Forest | criterion: entropy, max_depth: 40, max_features: sqrt, min_samples_leaf: 2, min_samples_split: 5, n_estimators: 200 | 0.92 |
| Artificial Neural Network | Hidden Layer(100) with Relu, Output Layer(1) with Sigmoid, epochs: 20, batch_size: 10, optimizer: adam | 0.99 |


# Summary of Findings

Among the conventional machine learning models, Random Forest exhibited the strongest performance with an ROC AUC score of 0.92. Nevertheless, the artificial neural network significantly outperformed it, achieving a notably higher score of 0.99. Notably, the Keras tuner wasn't employed to refine the hyperparameters, which could have potentially optimized factors such as the number of neurons, epochs, and batch size even further.

# Notebook

[Fraud-ML Notebook](/fraud-ml.ipynb)

# Citations and References:
 
 1. Credit Card Fraud Detection using Machine Learning Algorithms published in ScienceDirect: [https://www.sciencedirect.com/science/article/pii/S187705092030065X](https://www.sciencedirect.com/science/article/pii/S187705092030065X)
    
 2. Stripe's guide on machine learning for fraud detection: [https://stripe.com/in/guides/primer-on-machine-learning-for-fraud-protection](https://stripe.com/in/guides/primer-on-machine-learning-for-fraud-protection)
    
 3. Review of Machine Learning Approach on Credit Card Fraud Detection" available on Springer: [https://link.springer.com/article/10.1007/s44230-022-00004-0](https://link.springer.com/article/10.1007/s44230-022-00004-0)
