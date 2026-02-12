import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from DATA221_Assignment3_Question3 import get_train_test_split
from sklearn.metrics import accuracy_score

# Get train/test data from Question 3
feature_train, feature_test, label_train, label_test = get_train_test_split()

# Set multiple k, train the model and calculate the accuracy score
k = [1, 3, 5, 7, 9]
accuracy_score_list = []
for num_neighbors in k:
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
    trained_knn_model = knn_model.fit(feature_train, label_train)
    predicted_label = knn_model.predict(feature_test)
    accuracy = accuracy_score(label_test, predicted_label)
    accuracy_score_list.append(accuracy)

# Create dictionary that sore k value and accuracy score for each model
data = {'k value': k, 'Accuracy score': accuracy_score_list}

# Create dataframe from the dictionary
accuracy_table = pd.DataFrame(data)
print(accuracy_table)
print('The highest accuracy score is: ', max(accuracy_score_list))

# Question 1: How changing k affects the behavior of the model
''' Changing k value will affect how the model will make predictions. '''

# Question 2: Why very small values of k may cause overfitting
''' Very small values of k, the model will predict exactly the label of the nearest neighbor, 
which means the model might perform poorly on unseen data. '''

# Question 3: Why very large values of k may cause underfitting
''' Very large k will cause underfitting, the model will have to consider too many neighbor,
which makes it too smooth and misses the real patterns in the data. '''
