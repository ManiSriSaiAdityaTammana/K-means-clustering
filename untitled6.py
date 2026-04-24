import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

covid_data = pd.read_csv("covid csv.csv")

covid_data = covid_data[['Country/other', 'Total_cases', 'Total_recovered']]

covid_data['Total_recovered'] = covid_data['Total_recovered'].replace(0, 1)

covid_data['ratio'] = covid_data['Total_cases'] / covid_data['Total_recovered']

covid_data['ratio'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

covid_data.dropna(inplace=True)

scaler = MinMaxScaler()
scaler.fit(covid_data[['ratio']])
covid_data['ratio_normalized'] = scaler.transform(covid_data[['ratio']])

kmeans = KMeans(n_clusters=3, random_state=42)
covid_data['cluster'] = kmeans.fit_predict(covid_data[['ratio_normalized']])

print(covid_data.head())

plt.figure(figsize=(10, 6))
plt.scatter(covid_data['Total_cases'], covid_data['Total_recovered'], c=covid_data['cluster'], cmap='viridis')
plt.xlabel('Total Cases')
plt.ylabel('Total Recovered')
plt.title('Clustering based on Ratio of Affected to Recovered Cases')
plt.colorbar(label='Cluster')
plt.show()

colors = {0: 'red', 1: 'green', 2: 'blue'}

plt.figure(figsize=(12, 8))
plt.bar(covid_data['Country/other'], covid_data['ratio'], color=covid_data['cluster'].map(colors))
plt.xlabel('Country')
plt.ylabel('Ratio of Affected to Recovered Cases')
plt.title('Clustering by Country based on Ratio of Affected to Recovered Cases')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
