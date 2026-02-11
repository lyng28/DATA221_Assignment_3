import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv('crime1.csv')

violent_crime_rate = file['ViolentCrimesPerPop']

# Histogram
plt.hist(violent_crime_rate, bins= 20, edgecolor= 'white')
plt.title('Violent Crime Rate Distribution')
plt.xlabel('Violent Crime Per Population')
plt.ylabel('Frequency')
plt.show()

# Boxplot
plt.boxplot(violent_crime_rate)
plt.title('Violent Crime Rate Boxplot')
plt.ylabel('Violent Crime Per Population')
plt.show()

# Question 1: What the histogram shows about how the data values are spread
''' The histogram shows how often (the frequency) of the crime rate values. We can see that most of the 
value are in the lower value sides and the bars are smaller as the value increases. Therefore we could
see that the data is not evenly spread, and it has a longer tail to the right at the maximum value of 1.00. '''

# Question 2: What the box plot shows about the median
''' The box plot shows that the median marks the middle value of the dataset, and it lies around 0.39 (exactly 
the output we calculated in Question 1), which means that most of the crime rate are < 0.39. '''

# Question 3: Whether the box plot suggests the presence of outliers
''' Yes, the boxplot shows the data beyond the whiskers so there are presence of outliers. And this would
be suggested as the maximum value is 1.00, which is far from most of the values in the dataset. '''

