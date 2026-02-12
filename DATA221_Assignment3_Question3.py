import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Call function for easier access for Question 4 and 5
def get_train_test_split():
    file = pd.read_csv('kidney_disease.csv')

    # Identify numeric/category columns:
    # Code learn from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
    numeric_columns = file.select_dtypes(include=['int64', 'float64']).columns
    category_columns = file.select_dtypes(include=['string', 'object']).columns

    # Fill numeric columns with mean values
    # Code learn from https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    numeric_fill = SimpleImputer(strategy='mean')
    file[numeric_columns] = numeric_fill.fit_transform(file[numeric_columns])

    # Fill category column with most frequent values
    category_fill = SimpleImputer(strategy='most_frequent')
    file[category_columns] = category_fill.fit_transform(file[category_columns])

    feature_matrix = file.drop(columns=['classification']) # Drop the Classification column
    target = file['classification'] # Create label vector y

    # Convert categorial datas to number for training purpose
    # Code learn from https://www.geeksforgeeks.org/pandas/python-pandas-get_dummies-method/
    category_columns = feature_matrix.select_dtypes(include=['object', 'string']).columns
    feature_matrix = pd.get_dummies(feature_matrix, columns=category_columns)

    # Perform splitting
    feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix, target, test_size=0.30, random_state=42)

    return feature_train, feature_test, target_train, target_test

# Question 1: Why we should not train and test a model on the same data
''' We should not train and test a model on the same data because the model would memorize the
training data, which will give high accuracy if we use the same data as the testing data.
So when we will not know if it will have the same accuracy on an unseen set of datas. '''

# What the purpose of the testing set is
''' Testing set is used to evaluate how well the training set performs on training the machine. '''