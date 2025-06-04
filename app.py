import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering

# --- Load scaler and original data ---
@st.cache_data
def load_model_and_data():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    user_features = pd.read_csv('user_features.csv')
    features = ['activity_frequency', 'category_count', 'event_type_count',
                'unique_products', 'unique_brands', 'average_spend']
    return user_features[features], scaler, features

# --- Load recommendations data ---
@st.cache_data
def load_recommendations():
    df = pd.read_csv('merged_data.csv')
    filtered_df = df[df['event_type'] == 'view']
    cluster_recs = (
        filtered_df
        .groupby(['final_cluster', 'main_category', 'sub_category', 'brand', 'product_id'])
        .size()
        .reset_index(name='count')
    )
    return cluster_recs

# --- Page Config ---
st.set_page_config(page_title="🧠 User Cluster Assignment & Product Recommendations", layout="centered")
st.title("🧠 User Cluster Assignment & Product Recommendations")
st.markdown("Simulate user behavior and receive tailored product recommendations based on behavioral clustering.")

# --- Load Data and Scaler ---
user_features, scaler, features = load_model_and_data()
cluster_recs = load_recommendations()

# --- User Input Form ---
with st.form("user_input_form"):
    st.subheader("📊 Simulate User Behavior")
    col1, col2 = st.columns(2)

    with col1:
        activity = st.slider("🔄 Activity Frequency", 0, 6, 1)
        event_types = st.slider("🎯 Event Type Count", 0, 2, 1)
        unique_products = st.slider("🛍️ Unique Products", 0, 6, 1)

    with col2:
        category_count = st.slider("📚 Category Count", 0, 2, 1)
        unique_brands = st.slider("🏷️ Unique Brands", 0, 4, 1)
        average_spend = st.slider("💰 Average Spend", 0.0, 1000.0, 100.0, step=10.0)

    submitted = st.form_submit_button("🚀 Assign Cluster & Get Recommendations")

# --- Nested Category Filtering ---
st.subheader("🔍 Filter Recommendations")

main_categories = sorted(cluster_recs['main_category'].dropna().unique().tolist())
selected_main = st.selectbox("Select Main Category", ["All"] + main_categories)

if selected_main != "All":
    sub_options = sorted(cluster_recs[cluster_recs['main_category'] == selected_main]['sub_category'].dropna().unique().tolist())
else:
    sub_options = sorted(cluster_recs['sub_category'].dropna().unique().tolist())

selected_sub = st.selectbox("Select Sub Category", ["All"] + sub_options)

if selected_main != "All":
    main_mask = cluster_recs['main_category'] == selected_main
else:
    main_mask = pd.Series([True] * len(cluster_recs))

if selected_sub != "All":
    sub_mask = cluster_recs['sub_category'] == selected_sub
else:
    sub_mask = pd.Series([True] * len(cluster_recs))

brand_options = sorted(
    cluster_recs[main_mask & sub_mask]['brand'].dropna().unique().tolist()
)

selected_brand = st.selectbox("Select Brand", ["All"] + brand_options)

# --- Process Submission ---
if submitted:
    new_user_df = pd.DataFrame([{
        'activity_frequency': activity,
        'category_count': category_count,
        'event_type_count': event_types,
        'unique_products': unique_products,
        'unique_brands': unique_brands,
        'average_spend': average_spend
    }])

    combined_df = pd.concat([user_features, new_user_df], ignore_index=True)
    scaled_data = scaler.transform(combined_df)

    model = AgglomerativeClustering(n_clusters=3, linkage='average', metric='euclidean')
    labels = model.fit_predict(scaled_data)
    new_user_cluster = labels[-1]

    interpretations = {
        0: "💼 Cluster 0 – Occasional Explorers",
        1: "🛒 Cluster 1 – Frequent Shoppers",
        2: "💳 Cluster 2 – High Spenders"
    }

    st.success(f"✅ Assigned to Cluster {new_user_cluster}")
    st.info(f"📌 Interpretation: {interpretations.get(new_user_cluster, 'Unknown Cluster')}")

    # --- Filter Recommendations Based on Selection ---
    filtered = cluster_recs[cluster_recs['final_cluster'] == new_user_cluster]

    if selected_main != "All":
        filtered = filtered[filtered['main_category'] == selected_main]
    if selected_sub != "All":
        filtered = filtered[filtered['sub_category'] == selected_sub]
    if selected_brand != "All":
        filtered = filtered[filtered['brand'] == selected_brand]

    top_products_df = (
        filtered
        .sort_values('count', ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    if not top_products_df.empty:
        st.subheader("🛍️ Top 5 Recommended Products for You")
        display_df = top_products_df[['product_id', 'main_category', 'sub_category', 'brand', 'count']]
        display_df.columns = ['Product ID', 'Main Category', 'Sub Category', 'Brand', 'Popularity']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No products found matching your selection and cluster.")
