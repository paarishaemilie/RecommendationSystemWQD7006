# User Cluster Assignment & Product Recommendations

This Streamlit app allows you to simulate a new user's behavior, assign them to a user cluster using hierarchical clustering (Agglomerative Clustering), and get personalized product recommendations based on their cluster and selected product categories.

## Features

- Input sliders to simulate user behavior features (activity frequency, category count, unique products, average spend, etc.)
- Assigns new user to one of three clusters:
  - 💼 **Cluster 0** – Occasional Explorers
  - 🛒 **Cluster 1** – Frequent Shoppers
  - 💳 **Cluster 2** – High Spenders
- Nested dropdown filters for Main Category, Sub Category, and Brand
- Displays top 10 recommended products for the assigned cluster and selected filters
- Clean, responsive UI with product recommendations shown in a table format

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- pandas
- numpy
- scikit-learn

### Installation

1. Clone this repository:

```bash
git clone https://github.com/paarishaemilie/RecommendationSystemWQD7006.git
cd RecommendationSystemWQD7006
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Place your data files (scaler.pkl, user_features.csv, and merged_data.csv) in the project directory.

### Running the App

Run the Streamlit app locally:

```bash
streamlit run app.py
```
The app will open in your default browser at http://localhost:8501.

### Deployment

To deploy on Streamlit Cloud:
1. Push your project (including app.py, data files, and requirements.txt) to your GitHub repository.
2. Create a new app on Streamlit Cloud linked to your GitHub repo.
3. Streamlit Cloud will automatically install dependencies and deploy the app.

### Project Structure

```bash
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── scaler.pkl           # Pickled scaler for feature scaling
├── user_features.csv    # Dataset of user features for clustering
└── merged_data.csv      # Dataset containing product and cluster info
```
