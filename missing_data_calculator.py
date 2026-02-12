import pandas as pd

file = pd.read_csv('kidney_disease.csv')

# Total missing values in the dataset
total_missing = file.isna().sum().sum()
print(f'Total missing cells in dataset: {total_missing}')

# Missing values per column
missing_per_column = file.isna().sum()
print('Missing values per column:')
print(missing_per_column)

# Number of rows with any missing values
rows_with_missing = file.isna().any(axis=1).sum()
print(f'Total rows with missing data: {rows_with_missing}')

# Total rows
print(f'Total rows in dataset: {file.shape[0]}')
