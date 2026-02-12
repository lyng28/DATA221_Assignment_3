import pandas as pd
from sklearn.model_selection import train_test_split

file = pd.read_csv('kidney_disease.csv')

feature_matrix = file.drop(columns=['classification']) # Drop the CKD column
target = file['classification'] # Create label vector y

# Perform splitting
feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix, target, test_size=0.30, random_state=42)

# Question 1: Why we should not train and test a model on the same data
''' We should not train and test a model on the same data because the model would memorize the
training data, which will give high accuracy if we use the same data as the testing data.
So when we will not know if it will have the same accuracy on an unseen set of datas. '''

# What the purpose of the testing set is
''' Testing set is used to evaluate how well the training set performs on training the machine. '''