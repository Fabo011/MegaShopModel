import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from utils.supabase import upload_model_to_supabase
from sklearn.metrics import silhouette_score
import plotly.graph_objs as go
from dash import Dash, dcc, html

# Step 1: Load Data
data = pd.read_csv('data/mall_customers.csv')

# Step 2: Data Preparation
features = ['age', 'annual_income', 'spending_score']
X = data[features]

# Step 3: Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Step 5: K-Means Clustering
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Step 6: Save the trained model
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 7: Add cluster assignments to the data
clusters = kmeans.predict(X_scaled)
data['Cluster'] = clusters

# Calculate silhouette scores
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    clusters = kmeans.predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    silhouette_scores.append(sil_score)

# Step 8: Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("K-Means Clustering Visualization"),

    dcc.Graph(
        id='elbow-method',
        figure=go.Figure(
            data=go.Scatter(
                x=list(range(1, 11)),
                y=wcss,
                mode='lines+markers',
                name='WCSS',
            )
        ).update_layout(
            title='Elbow Method for Optimal k',
            xaxis_title='Number of Clusters',
            yaxis_title='WCSS'
        )
    ),

    dcc.Graph(
        id='clustering',
        figure=go.Figure(
            data=go.Scatter(
                x=X_scaled[:, 0],
                y=X_scaled[:, 1],
                mode='markers',
                marker=dict(color=clusters, colorscale='Viridis', showscale=True),
                text=clusters,
            )
        ).update_layout(
            title='K-Means Clustering of Mall Customers',
            xaxis_title='Age (scaled)',
            yaxis_title='Annual Income (scaled)'
        )
    ),

    dcc.Graph(
        id='silhouette-scores',
        figure=go.Figure(
            data=go.Scatter(
                x=list(cluster_range),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
            )
        ).update_layout(
            title='Silhouette Scores for Different Number of Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score'
        )
    ),
], style={'text-align': 'center'})

upload_model_to_supabase('kmeans_model.pkl', 'kmeans_model.pkl')
upload_model_to_supabase('scaler.pkl', 'scaler.pkl')

if __name__ == '__main__':
    app.run_server(debug=True)

