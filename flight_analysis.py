
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Load the data
file_path = 'flight_dataset.csv'
df = pd.read_csv(file_path)

# Data Preparation
df['Total_Duration_Minutes'] = df['Duration_hours'] * 60 + df['Duration_min']
df_cleaned = df.drop(columns=['Duration_hours', 'Duration_min'])

# Exploratory Analysis
avg_price_airline = df_cleaned.groupby('Airline')['Price'].mean().sort_values()
avg_price_stops = df_cleaned.groupby('Total_Stops')['Price'].mean()
plt.figure(figsize=(10, 6))
avg_price_airline.plot(kind='bar')
plt.title('Average Price per Airline')
plt.xlabel('Airline')
plt.ylabel('Average Price (₹)')
plt.savefig('graph0_en.png')

plt.figure(figsize=(10, 6))
avg_price_stops.plot(kind='bar')
plt.title('Impact of Number of Stops on Ticket Prices')
plt.xlabel('Number of Stops')
plt.ylabel('Average Price (₹)')
plt.savefig('graph1_en.png')

plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Total_Duration_Minutes'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Flight Durations')
plt.xlabel('Flight Duration (minutes)')
plt.ylabel('Frequency')
plt.savefig('graph2_en.png')

# Modeling
features = df_cleaned[['Airline', 'Source', 'Destination', 'Total_Stops', 'Dep_hours', 'Dep_min', 'Arrival_hours', 'Arrival_min']]
target = df_cleaned['Total_Duration_Minutes']

label_encoder = LabelEncoder()
for column in ['Airline', 'Source', 'Destination']:
    features[column] = label_encoder.fit_transform(features[column])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Advanced Analyses
correlation_matrix = df_cleaned.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('graph_correlation.png')

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
feature_names = features.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.savefig('graph_feature_importance.png')

kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(features)

plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Dep_hours'], df_cleaned['Total_Duration_Minutes'], c=df_cleaned['Cluster'], cmap='viridis')
plt.title('Clustering of Flights Based on Departure Hours and Duration')
plt.xlabel('Departure Hours')
plt.ylabel('Flight Duration (minutes)')
plt.savefig('graph_clustering.png')

avg_price_month = df_cleaned.groupby('Month')['Price'].mean()
plt.figure(figsize=(10, 6))
avg_price_month.plot(kind='line')
plt.title('Average Flight Price Over Months')
plt.xlabel('Month')
plt.ylabel('Average Price (₹)')
plt.savefig('graph_time_series.png')
