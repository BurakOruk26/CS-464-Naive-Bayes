# About the Homework
## Dataset
The dataset consists of 100,000 space observations collected by the Sloan Digital Sky Survey (SDSS). Each observation is described by 9 feature columns and a class column that identifies whether the object is a galaxy or not. The dataset has been preprocessed and adapted for the homework. It is divided into two subsets: 80,000 observations for training and 17,576 for testing.

## Task
To train a Naive Bayes model on the training set and evaluate the model on the test set given. Please note that it is needed to implement Naive Bayes from scratch

# About the Code
## Prerequisites
Following Python libraries are needed to run these codes:
- pandas        (for some csv operations)
- matplotlib    (for data visualization)
- numpy         (for data visualization)
- seaborn       (for data visualization)

## How to run
There are two steps to obtain results for question 3:
1. Run train_model.py to create model.json.
2. Run evaluate_model.py to do predictions on the test dataset and then receive the confusion matrix.
