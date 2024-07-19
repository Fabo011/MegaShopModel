import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from utils.supabase import upload_model_to_supabase
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from dash import Dash, dcc, html, dash_table
from matplotlib import cm

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
silhouette_scores = []
for i in range(2, 9):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=51, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Step 5: K-Means Clustering with optimal_k
optimal_k = 4  # Based on the elbow method and silhouette scores, change this as needed
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=51, random_state=42)
kmeans.fit(X_scaled)

# Step 6: Save the trained model
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 7: Add cluster assignments to the data
clusters = kmeans.predict(X_scaled)
data['Cluster'] = clusters

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate mean values of features for each cluster
cluster_means = data.groupby('Cluster')[features].mean().reset_index()

# Assign colors to clusters
num_clusters = len(cluster_means)
colors = cm.rainbow(np.linspace(0, 1, num_clusters))
color_map = {i: f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for i, c in enumerate(colors)}
data['Color'] = data['Cluster'].map(color_map)
cluster_means['Color'] = cluster_means['Cluster'].map(color_map)

# Step 8: Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("K-Means Clustering Visualization"),

    dcc.Graph(
        id='elbow-method',
        figure=go.Figure(
            data=go.Scatter(
                x=list(range(2, 9)),
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
        id='silhouette-score',
        figure=go.Figure(
            data=go.Scatter(
                x=list(range(2, 9)),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
            )
        ).update_layout(
            title='Silhouette Score for Optimal k',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score'
        )
    ),

    dcc.Graph(
        id='clustering',
        figure=go.Figure(
            data=go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode='markers',
                marker=dict(color=data['Color'], showscale=True),
                text=data['Cluster'],
            )
        ).update_layout(
            title='K-Means Clustering of Mall Customers',
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2'
        )
    ),
    
    dash_table.DataTable(
        id='cluster-means',
        columns=[
            {"name": col, "id": col} for col in cluster_means.columns if col != 'Color'
        ],
        data=cluster_means.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{{Cluster}} = {}'.format(i)},
                'backgroundColor': color_map[i],
                'color': 'black'
            } for i in range(num_clusters)
        ]
    ),
    
    dash_table.DataTable(
        id='customer-clusters',
        columns=[{"name": col, "id": col} for col in ['customer_id', 'Cluster']],
        data=data[['customer_id', 'Cluster']].to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center'},
        page_size=10  # Display 10 rows per page
    ),


], style={'text-align': 'center'})

upload_model_to_supabase('kmeans_model.pkl', 'kmeans_model.pkl')
upload_model_to_supabase('scaler.pkl', 'scaler.pkl')

if __name__ == '__main__':
    app.run_server(debug=True)
