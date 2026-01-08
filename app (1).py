import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Page config
st.set_page_config(page_title="Hierarchical Clustering", layout="wide")

st.title("🔗 Hierarchical Clustering Demo")
st.write("This app demonstrates Hierarchical Clustering using the Wine dataset.")

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Sidebar controls
st.sidebar.header("⚙️ Clustering Settings")
num_clusters = st.sidebar.slider("Select number of clusters", 2, 6, 3)
linkage_method = st.sidebar.selectbox(
    "Select linkage method",
    ["ward", "complete", "average", "single"]
)

# Dendrogram
st.subheader("🌳 Dendrogram")
linked = linkage(scaled_data, method=linkage_method)

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(linked, truncate_mode="lastp", p=30, ax=ax)
ax.set_title("Hierarchical Clustering Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")
st.pyplot(fig)

# Agglomerative Clustering
cluster = AgglomerativeClustering(
    n_clusters=num_clusters,
    linkage=linkage_method
)
labels = cluster.fit_predict(scaled_data)

df["Cluster"] = labels

st.subheader("📌 Clustered Data (First 10 rows)")
st.dataframe(df.head(10))

# Visualization
st.subheader("🎨 Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=df.iloc[:, 0],
    y=df.iloc[:, 1],
    hue=df["Cluster"],
    palette="Set2",
    ax=ax2
)
ax2.set_xlabel(df.columns[0])
ax2.set_ylabel(df.columns[1])
ax2.set_title("Cluster Visualization (2 Features)")
st.pyplot(fig2)

st.success("✅ Hierarchical Clustering completed successfully!")
