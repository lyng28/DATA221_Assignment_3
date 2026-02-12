import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from DATA221_Assignment3_Question3 import get_train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Get train/test data from Question 3
feature_train, feature_test, label_train, label_test = get_train_test_split()

# Set KNN to 5 and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
trained_knn_model = knn_model.fit(feature_train, label_train)

# Predict the labels of test data
predicted_labels = trained_knn_model.predict(feature_test)

# Calculation
confusion_matrix_result = confusion_matrix(label_test, predicted_labels)
confusion_matrix_table = pd.DataFrame(confusion_matrix_result, index=['Actual Negative', 'Actual Positive'],
                                      columns=['Predicted Negative', 'Predicted Positive'])
accuracy = accuracy_score(label_test, predicted_labels)
precision = precision_score(label_test, predicted_labels, pos_label='ckd')
recall = recall_score(label_test, predicted_labels, pos_label='ckd')
f1 = f1_score(label_test, predicted_labels, pos_label='ckd')

print(confusion_matrix_table)
print('Accuracy score: ', round(accuracy, 2))
print('Precision score: ', round(precision, 2))
print('Recall score: ', round(recall, 2))
print('F1 score: ', round(f1, 2))

# What True Positive, True Negative, False Positive, and False Negative mean in the context
# of kidney disease prediction
''' TP is when the model correctly predicts the patient has ckd (Positive Class). 
TN is when the model correctly predicts the patient does not have ckd (notckd/Negative Class).
FP is when the model predicts the patient has ckd but the patient is not having ckd.
FN is when the model predicts the patient does not have ckd but the patient actually has ckd.'''

# Why accuracy alone may not be enough to evaluate a classification model
''' The accuracy score alone may not be enough to evaluate a model because it only use the correct 
predictions to evaluate. 
In a dataset where there are more healthy cases than ckd cases, the model will not be trained very well.'''

# Which metric is most important if missing a kidney disease case is very serious, and why
''' The most important metric is the recall score because it measure how many patients that are ckd positive
actually identified. Higher recall score means less false negative so it will reduce the chance of missing 
someone with kidney disease. '''






