import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from utils.supabase import upload_model_to_supabase
from sklearn.metrics import silhouette_score

# Step 1: Load Data
data = pd.read_csv('data/mall_customers.csv')

# Step 2: Data Preparation (only relevant features)
features = ['age', 'annual_income', 'spending_score']
X = data[features]

# Step 3: Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method (Optimale Anzahl von Clustern)
# First visualization
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# Step 5: K-Means Clustering
optimal_k = 5 # Just imagine the elbow is k=5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)  # The model is trained here

# Step 6: Save the trained model
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
upload_model_to_supabase('kmeans_model.pkl', 'kmeans_model.pkl')
upload_model_to_supabase('scaler.pkl', 'scaler.pkl')

# Step 7: Add cluster assignments to the data
clusters = kmeans.predict(X_scaled)
data['Cluster'] = clusters

plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Visualize the results
# Second Visalization
plt.figure(figsize=(10,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering of Mall Customers')
plt.xlabel('Age (scaled)')
plt.ylabel('Annual Income (scaled)')
plt.show()

# Test the result
# Store silhouette scores for different cluster numbers
silhouette_scores = []
cluster_range = range(2, 11)  # Testing clusters from 2 to 10

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    clusters = kmeans.predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    silhouette_scores.append(sil_score)

# Step 2: Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(cluster_range)
plt.grid()
plt.show()
