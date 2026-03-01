
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import plotly.express as px
import plotly.figure_factory as ff
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(layout="wide", page_title="Wholesale Customer Segmentation")

# Custom CSS for premium look
components.html(
    """
    <style>
        #canvas {
            position: fixed;
            left: 0;
            top: 0;
            z-index: -1;
            width: 100vw;
            height: 100vh;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: none;
        }
        [data-testid="stAppViewContainer"] {
            background: none;
        }
        [data-testid="stHeader"] {
            background: none;
        }
        [data-testid="stToolbar"] {
            background: none;
        }
        [data-testid="stBlock"] {
            background: none;
        }
        .st-sidebar {
            background-color: #252526;
        }
        h1, h2, h3 {
            color: #00BFFF; /* Deep Sky Blue */
        }
        .stButton>button {
            color: #00BFFF;
            border-radius:12px;
            border:1px solid #00BFFF;
            background-color: transparent;
        }
        .stButton>button:hover {
            color: #FFFFFF;
            background-color: #00BFFF;
        }
    </style>
    <canvas id="canvas"></canvas>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>
    <script>
        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );
        var renderer = new THREE.WebGLRenderer({canvas: document.getElementById("canvas"), alpha: true});
        renderer.setSize( window.innerWidth, window.innerHeight );
        document.body.appendChild( renderer.domElement );
        var geometry = new THREE.TorusGeometry( 10, 3, 16, 100 );
        var material = new THREE.PointsMaterial( { color: 0x00BFFF, size: 0.02 } );
        var torus = new THREE.Points( geometry, material );
        scene.add( torus );
        camera.position.z = 20;
        var animate = function () {
            requestAnimationFrame( animate );
            torus.rotation.x += 0.001;
            torus.rotation.y += 0.001;
            renderer.render( scene, camera );
        };
        animate();
        window.addEventListener('resize', function() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize( window.innerWidth, window.innerHeight );
        });
    </script>
    """,
    height=0,
    width=0,
)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('Wholesale customers data.csv')
        return data
    except FileNotFoundError:
        st.error("The file 'Wholesale customers data.csv' was not found. Please make sure it's in the same directory as the app.")
        return None

def preprocess_data(data):
    # Drop non-numeric columns for this analysis
    data_numeric = data.drop(['Channel', 'Region'], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_numeric)
    return scaled_data, data_numeric.columns

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters

def agglomerative_clustering(data, n_clusters):
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agg_cluster.fit_predict(data)
    return clusters

# Main app
st.title("welcome to ,")
st.markdown("### SGRAR analytics")

# Load data
data = load_data()

if data is not None:
    # Sidebar for options
    st.sidebar.header("Clustering Options")
    
    # Preprocess data
    scaled_data, feature_names = preprocess_data(data.copy())
    
    # --- K-Means Clustering ---
    st.header("world of  Clustering")
    
    k = st.sidebar.slider("Number of clusters (K)", 2, 10, 3)

    if st.sidebar.button("Run K-Means"):
        kmeans_clusters = kmeans_clustering(scaled_data, k)
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = kmeans_clusters
        
        st.subheader(f"K-Means Results with {k} clusters")
        
        # 3D Scatter plot
        fig_3d = px.scatter_3d(data_with_clusters, x='Fresh', y='Milk', z='Grocery',
                               color='Cluster', symbol='Cluster',
                               title=f'3D Scatter Plot of Clusters (K-Means)',
                               labels={'Fresh': 'Fresh', 'Milk': 'Milk', 'Grocery': 'Grocery'},
                               color_continuous_scale=px.colors.sequential.Viridis)
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="rgb(200, 200, 230)"),
                yaxis=dict(backgroundcolor="rgb(230, 200, 200)"),
                zaxis=dict(backgroundcolor="rgb(200, 230, 200)"),
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- Hierarchical Clustering ---
    st.header("Hierarchical Clustering")
    
    n_clusters_agg = st.sidebar.slider("Number of clusters (Hierarchical)", 2, 10, 3)

    if st.sidebar.button("Run Hierarchical Clustering"):
        st.subheader(f"Hierarchical Clustering Results with {n_clusters_agg} clusters")
        
        # Dendrogram
        st.write("### Dendrogram")
        st.write("Due to performance, the dendrogram is generated on the first 100 samples.")
        fig_dendro = ff.create_dendrogram(scaled_data[:100], orientation='bottom', labels=data.index[:100].tolist())
        fig_dendro.update_layout(width=800, height=500, title="Dendrogram for first 100 samples")
        st.plotly_chart(fig_dendro, use_container_width=True)

        # Agglomerative clustering and 3D plot
        agg_clusters = agglomerative_clustering(scaled_data, n_clusters_agg)
        data_with_clusters_agg = data.copy()
        data_with_clusters_agg['Cluster'] = agg_clusters
        
        fig_3d_agg = px.scatter_3d(data_with_clusters_agg, x='Fresh', y='Milk', z='Grocery',
                                   color='Cluster', symbol='Cluster',
                                   title=f'3D Scatter Plot of Clusters (Hierarchical)',
                                   labels={'Fresh': 'Fresh', 'Milk': 'Milk', 'Grocery': 'Grocery'},
                                   color_continuous_scale=px.colors.sequential.Plasma)
        fig_3d_agg.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="rgb(200, 200, 230)"),
                yaxis=dict(backgroundcolor="rgb(230, 200, 200)"),
                zaxis=dict(backgroundcolor="rgb(200, 230, 200)"),
            )
        )
        st.plotly_chart(fig_3d_agg, use_container_width=True)

    # --- Data Preview ---
    st.header("Data Preview")
    st.dataframe(data.head())

    # --- Developed By ---
    st.sidebar.markdown("---")
    st.sidebar.header("Developed by")
    st.sidebar.markdown("- SMARAK DAS")
    st.sidebar.markdown("- GAGAN KUMAR")
    st.sidebar.markdown("- RUDRANARAYAN TRIPATHY")
    st.sidebar.markdown("- ANSHUMAN PANIGRAHI")
    st.sidebar.markdown("- RUDRANARAYAN DEBATA")

else:
    st.warning("Could not load data. Please ensure the CSV file is present.")
