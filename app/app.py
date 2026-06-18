import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

model_path  = os.path.join(ROOT_DIR, 'models', 'kmeans_model.pkl')
scaler_path = os.path.join(ROOT_DIR, 'models', 'scaler.pkl')

# ── Load Model & Scaler ────────────────────────────────
kmeans = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ── Cluster Labels ─────────────────────────────────────
cluster_labels = {
    0: 'Champions',
    1: 'Occasional Buyers',
    2: 'Loyal Customers',
    3: 'Lost Customers'
}

cluster_colors = {
    'Champions':        '#2ecc71',
    'Loyal Customers':  '#3498db',
    'Occasional Buyers':'#f39c12',
    'Lost Customers':   '#e74c3c'
}

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Upload your transaction data to segment customers using K-Means clustering.")

# ── File Upload ────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ File uploaded! Shape: {df.shape}")

    # ── Clean & RFM ────────────────────────────────────
    df = df.dropna(subset=['CustomerID', 'Description'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency = ('InvoiceNo',   'nunique'),
        Monetary  = ('TotalPrice',  'sum')
    ).reset_index()

    # ── Scale & Predict ────────────────────────────────
    rfm_scaled = scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
    rfm['Cluster'] = kmeans.predict(rfm_scaled)
    rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

    # ── Metrics ────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(rfm))
    col2.metric("🏆 Champions",      len(rfm[rfm['Segment'] == 'Champions']))
    col3.metric("⭐ Loyal",           len(rfm[rfm['Segment'] == 'Loyal Customers']))
    col4.metric("💤 Lost",            len(rfm[rfm['Segment'] == 'Lost Customers']))

    # ── Pie Chart ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Segment Distribution")
    fig_pie = px.pie(
        rfm,
        names='Segment',
        title='Customer Segments',
        color='Segment',
        color_discrete_map=cluster_colors
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── RFM Table ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Customer RFM Table")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']])

    # ── Download ───────────────────────────────────────
    csv = rfm.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Segmented Data", csv, "segments.csv", "text/csv")