import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

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

plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Schritt 5: K-Means Clustering anwenden
optimal_k = 5  # Angenommen, der Ellbogen ist bei k=5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)  # Das Modell wird hier trainiert

# Das trainierte Modell speichern
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Schritt 6: Cluster-Zuordnungen zu den Daten hinzufügen
clusters = kmeans.predict(X_scaled)
data['Cluster'] = clusters

# Ergebnisse visualisieren
# Second Visalization
plt.figure(figsize=(10,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering of Mall Customers')
plt.xlabel('Age (scaled)')
plt.ylabel('Annual Income (scaled)')
plt.show()
