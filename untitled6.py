# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mZC-xWcm6Y-eLVqwdeCWKSKSBOT8vL7S
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load the dataset
covid_data = pd.read_csv("covid csv.csv")

# Drop unnecessary columns
covid_data = covid_data[['Country/other', 'Total_cases', 'Total_recovered']]

# Replace zero recovered cases with a small value to avoid division by zero
covid_data['Total_recovered'] = covid_data['Total_recovered'].replace(0, 1)

# Calculate the ratio of affected to recovered cases
covid_data['ratio'] = covid_data['Total_cases'] / covid_data['Total_recovered']

# Check for infinite values and replace them with NaN
covid_data['ratio'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

# Drop rows with NaN values
covid_data.dropna(inplace=True)

# Normalize features using Min-Max scaling
scaler = MinMaxScaler()
covid_data['ratio_normalized'] = scaler.fit_transform(covid_data[['ratio']])

# Clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
covid_data['cluster'] = kmeans.fit_predict(covid_data[['ratio_normalized']])

# Display the clustered data
print(covid_data.head())

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
covid_data = pd.read_csv("covid csv.csv")

# Drop unnecessary columns
covid_data = covid_data[['Country/other', 'Total_cases', 'Total_recovered']]

# Replace zero recovered cases with a small value to avoid division by zero
covid_data['Total_recovered'] = covid_data['Total_recovered'].replace(0, 1)

# Calculate the ratio of affected to recovered cases
covid_data['ratio'] = covid_data['Total_cases'] / covid_data['Total_recovered']

# Check for infinite values and replace them with NaN
covid_data['ratio'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

# Drop rows with NaN values
covid_data.dropna(inplace=True)

# Normalize features using Min-Max scaling
scaler = MinMaxScaler()
# Fit the scaler on the data
scaler.fit(covid_data[['ratio']])
# Transform the 'ratio' column
covid_data['ratio_normalized'] = scaler.transform(covid_data[['ratio']])

# Clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
covid_data['cluster'] = kmeans.fit_predict(covid_data[['ratio_normalized']])

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(covid_data['Total_cases'], covid_data['Total_recovered'], c=covid_data['cluster'], cmap='viridis')
plt.xlabel('Total Cases')
plt.ylabel('Total Recovered')
plt.title('Clustering based on Ratio of Affected to Recovered Cases')
plt.colorbar(label='Cluster')
plt.show()

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
covid_data = pd.read_csv("covid csv.csv")

# Drop unnecessary columns
covid_data = covid_data[['Country/other', 'Total_cases', 'Total_recovered']]

# Replace zero recovered cases with a small value to avoid division by zero
covid_data['Total_recovered'] = covid_data['Total_recovered'].replace(0, 1)

# Calculate the ratio of affected to recovered cases
covid_data['ratio'] = covid_data['Total_cases'] / covid_data['Total_recovered']

# Check for infinite values and replace them with NaN
covid_data['ratio'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

# Drop rows with NaN values
covid_data.dropna(inplace=True)

# Normalize features using Min-Max scaling
scaler = MinMaxScaler()
# Fit the scaler on the data
scaler.fit(covid_data[['ratio']])
# Transform the 'ratio' column
covid_data['ratio_normalized'] = scaler.transform(covid_data[['ratio']])

# Clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
covid_data['cluster'] = kmeans.fit_predict(covid_data[['ratio_normalized']])

# Define colors for clusters
colors = {0: 'red', 1: 'green', 2: 'blue'}

# Plotting the clusters by countries
plt.figure(figsize=(12, 8))
plt.bar(covid_data['Country/other'], covid_data['ratio'], color=covid_data['cluster'].map(colors))
plt.xlabel('Country')
plt.ylabel('Ratio of Affected to Recovered Cases')
plt.title('Clustering by Country based on Ratio of Affected to Recovered Cases')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()