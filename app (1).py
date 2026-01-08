import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="Wine Clustering App", layout="centered")

st.title("🍷 Wine Clustering using DBSCAN")
st.write("Clustering on built-in Wine dataset (No CSV upload required)")

# --------------------------------
# Load Built-in Dataset
# --------------------------------
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

# --------------------------------
# Standardization
# --------------------------------
st.subheader("⚙️ Standardization")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

st.write("Standardized Data (Preview)")
st.dataframe(scaled_df.head())

# --------------------------------
# DBSCAN Parameters
# --------------------------------
st.subheader("🔧 DBSCAN Parameters")

eps = st.slider("Select eps value", 0.1, 5.0, 2.0, 0.1)
min_samples = st.slider("Select min_samples", 1, 10, 2)

# --------------------------------
# Apply DBSCAN
# --------------------------------
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(scaled_df)

df["Cluster"] = clusters

st.subheader("📌 Cluster Results")
st.write(df["Cluster"].value_counts())

st.dataframe(df.head())

# --------------------------------
# Visualization
# --------------------------------
st.subheader("📈 Cluster Visualization")

x_feature = st.selectbox("Select X-axis feature", wine.feature_names)
y_feature = st.selectbox("Select Y-axis feature", wine.feature_names, index=1)

fig, ax = plt.subplots()
sns.scatterplot(
    x=df[x_feature],
    y=df[y_feature],
    hue=df["Cluster"],
    palette="tab10",
    ax=ax
)

ax.set_title("DBSCAN Clustering Result")
st.pyplot(fig)
