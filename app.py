import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
from src.model import NeuralCollaborativeFiltering

st.set_page_config(page_title="NCF Movie Recommender", layout="wide")
st.title("Neural Collaborative Filtering (NCF) Recommender")

@st.cache_data
def load_movies():
    local_path = os.path.join("data", "u.item")
    
    # 1. Agar local data folder mein file maujood hai
    if os.path.exists(local_path):
        movies = pd.read_csv(local_path, sep="|", encoding="latin-1", header=None, usecols=[0, 1])
        movies.columns = ["item_id", "title"]
        return movies
    
    # 2. Agar local nahi hai toh official GroupLens zip se live extract karein
    os.makedirs("data", exist_ok=True)
    zip_path = os.path.join("data", "ml-100k.zip")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extract("ml-100k/u.item", "data")
    
    extracted_item = os.path.join("data", "ml-100k", "u.item")
    movies = pd.read_csv(extracted_item, sep="|", encoding="latin-1", header=None, usecols=[0, 1])
    movies.columns = ["item_id", "title"]
    return movies

movies_df = load_movies()

@st.cache_resource
def load_model():
    model = NeuralCollaborativeFiltering()
    model.load_state_dict(torch.load("checkpoints/ncf_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.sidebar.header("Benchmark Metrics")
st.sidebar.metric("Hit Rate@10 (HR@10)", "0.3924")
st.sidebar.metric("NDCG@10", "0.2170")

user_id = st.slider("Select User ID:", 0, 942, 42)
top_k = st.slider("Top-K Movies:", 3, 10, 5)

if st.button("Generate Recommendations", type="primary"):
    all_items = torch.arange(1682)
    user_tensor = torch.tensor([user_id] * 1682)
    with torch.no_grad():
        scores = model(user_tensor, all_items).numpy()
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    recs = movies_df[movies_df["item_id"].isin(top_indices + 1)]
    
    st.subheader(f"Top Recommendations for User {user_id}")
    for idx, (_, row) in enumerate(recs.iterrows(), start=1):
        st.write(f"**{idx}.** {row['title']}")