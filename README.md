# creditcard-fraud-detection
Machine Learning for Credit Card Fraud Detection

# Introduction

Credit card fraud has become one of the most rapidly evolving financial crimes in the world today, presenting significant challenges for individuals and businesses globally. The increasing use of credit cards for online and offline transactions has created an opportunity for fraudsters to exploit vulnerabilities in payment systems, using highly sophisticated methods. 

The aim of this project is to construct machine learning models employing various prediction algorithms and assess their outcomes. The primary goal is to create models capable of effectively detecting fraudulent transactions and rejecting them. Utilizing machine learning algorithms such as Logistic Regression, Support Vector Machines (SVM), and Random Forest, the results will be analyzed to determine the most effective model. Additionally, an artificial neural network model will be developed and contrasted with the conventional models to evaluate its performance and accuracy.

# Dataset
The dataset for this project is sourced from IBM's synthetic credit card data. 

**Exploratory data analysis:**


# Models Evaluation and Comparison

**Random Forest:**

Random Forest was tested, and the optimal hyperparameters identified were criterion: entropy, max_depth: 40, max_features: sqrt, min_samples_leaf: 2, min_samples_split: 5, and n_estimators: 200. The highest ROC score obtained was 0.92.

**Logistic Regression:**

Logistic Regression was experimented with, and the optimal hyperparameters identified were C: 29.76, max_iter: 100, and penalty: l2. The highest ROC score obtained was 0.86.

**Support Vector Machines:**

Support Vector Machines were experimented with, and the optimal hyperparameters identified were C: 0.1, degree: 0, gamma: 0.1, and kernel: linear. The highest ROC score achieved was 0.82.

**Artificial Neural Network:**

Ultimately, the Artificial Neural Network with 100 hidden layers utilizing ReLU activation, one output layer employing Sigmoid activation, 20 epochs, a batch size of 10, and the Adam optimizer surpassed all alternative models, achieving the highest ROC score of 0.99.


| MODEL | Hyper Parameters | ROC Score |
|:---:|:---:|:---:|
| Logistic Regression | C: 29.76, max_iter: 100, penalty: l2 | 0.86 |
| Support Vector Machines | C: 0.1, degree: 0, gamma: 0.1, kernel: linear | 0.82 |
| Random Forest | criterion: entropy, max_depth: 40, max_features: sqrt, min_samples_leaf: 2, min_samples_split: 5, n_estimators: 200 | 0.92 |
| Artificial Neural Network | Hidden Layer(100) with Relu, Output Layer(1) with Sigmoid, epochs: 20, batch_size: 10, optimizer: adam | 0.99 |

# Summary of Findings

Among the conventional machine learning models, Random Forest exhibited the strongest performance with a ROC score of 0.92. Nevertheless, the artificial neural network significantly outperformed it, achieving a notably higher score of 0.99. Notably, Keras tuner wasn't employed to refine the hyperparameters, which could have potentially optimized factors such as the number of neurons, epochs, and batch size even further.

# Citations and References:
 
 1. Credit Card Fraud Detection using Machine Learning Algorithms published in ScienceDirect: [https://www.sciencedirect.com/science/article/pii/S187705092030065X](https://www.sciencedirect.com/science/article/pii/S187705092030065X)
    
 2. Stripe's guide on machine learning for fraud detection: [https://stripe.com/in/guides/primer-on-machine-learning-for-fraud-protection](https://stripe.com/in/guides/primer-on-machine-learning-for-fraud-protection)
    
 3. Review of Machine Learning Approach on Credit Card Fraud Detection" available on Springer: [https://link.springer.com/article/10.1007/s44230-022-00004-0](https://link.springer.com/article/10.1007/s44230-022-00004-0)
