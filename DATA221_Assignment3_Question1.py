import pandas as pd
import numpy as np

file = pd.read_csv('crime1.csv')

# Filter out the violent crime rate for calculation
violent_crime_rate = file['ViolentCrimesPerPop']

# Calculation
mean = np.mean(violent_crime_rate)
median = np.median(violent_crime_rate)
standard_deviation = np.std(violent_crime_rate)
min_value = np.min(violent_crime_rate)
max_value = np.max(violent_crime_rate)

print('The mean value is: ', mean.round(2))
print('The median value is: ', median.round(2))
print('The standard deviation is: ', standard_deviation.round(2))
print('The maximum is: ', max_value)
print('The minimum is: ', min_value)


# Question 1: Compare the mean and the median. Does the distribution look symmetric or skewed? Explain briefly.
''' The mean value is 0.44 and the median is 0.39, so mean value > median value
The distribution look skewed and more possibly be right-skewed (positively skewed).
This usually means that most of the value in the data sets are on the lower value side, but there are couples of
extreme higher values that pull the mean upward. '''

# Question 2: If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why.
''' If there are extreme values, the mean is more affected as the mean uses all values to calculate,
and median only used sorted middle range values to calculate. '''






