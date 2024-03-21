# Term-paper-code

This repository is for the term paper code for CS 777 Spring 2024 created by Murong Li &amp; Chuao Peng

The data resource is from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Summary of dataset:

1. The transaction amount is relatively small. The mean of all the mounts made is approximately USD 88.
2. There are no "Null" values, so we don't have to work on ways to replace values.
3. Most of the transactions were Non-Fraud (99.83%) of the time, while Fraud transactions occurred (0.17%) of the time
   in the dataframe.
4. Except for the transaction and amount, we don't know what the other columns are (due to privacy reasons). The only
   thing we know is that those unknown columns have already been scaled.

# Environment and Libraries

1. You need to install pyspark package in order to run the code and process data in HDFS.
2. You need to install sklearn and imblearn in oreder to use stratified-K-Fold on imbalanced dataset.

# Sample files:

These two code samples are Jupyter notebook files. You can download and run them simply. But remember to download the
dataset from the URL above, then change the path to your local one.