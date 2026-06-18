# 🛍️ Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)
![ML](https://img.shields.io/badge/ML-K--Means%20%7C%20DBSCAN%20%7C%20PCA-orange)

A machine learning project that segments customers into meaningful business groups using **RFM Analysis** and **K-Means Clustering** on real-world retail transaction data.

🔗 **Live Demo:** https://customer-segmentation-ewol5krtehwtrjasfqvlqr.streamlit.app/

---

## 📌 Problem Statement

Businesses struggle to treat all customers the same way. This project identifies **distinct customer segments** from transaction data so businesses can target each group with the right strategy.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Online Retail Dataset
- **Size:** 5,41,909 transactions | 8 columns
- **Period:** 2010–2011 | UK-based online retail store

---

## 🧠 Approach

### 1. Data Cleaning
- Removed missing CustomerIDs (1,35,080 rows)
- Removed cancelled orders (InvoiceNo starting with 'C')
- Removed negative quantity and zero price entries

### 2. RFM Feature Engineering
| Feature | Description |
|---|---|
| **Recency** | Days since last purchase |
| **Frequency** | Number of unique orders |
| **Monetary** | Total amount spent |

### 3. Clustering
- **Elbow Method** → identified optimal K=4
- **Silhouette Score** → validated cluster quality (0.4277)
- **DBSCAN** → confirmed density-based comparison
- **PCA** → 3D visualization with 100% variance explained

---

## 👥 Customer Segments

| Segment | Customers | Strategy |
|---|---|---|
| 🏆 Champions | 673 | Reward and retain |
| ⭐ Loyal Customers | 833 | Upsell opportunities |
| 💛 Occasional Buyers | 1,752 | Re-engagement campaigns |
| 💤 Lost Customers | 1,080 | Win-back offers |

---

## 🚀 Live App Features

- Upload any retail transaction Excel file
- Automatic RFM calculation
- Interactive pie chart with segment distribution
- Full RFM table with segment labels
- Download segmented data as CSV

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Scikit-learn** — K-Means, DBSCAN, PCA
- **Pandas & NumPy** — Data processing
- **Plotly** — Interactive visualizations
- **Streamlit** — Web application
- **Joblib** — Model persistence

---

## 📁 Project Structurecustomer-segmentation/

├── app/

│   └── app.py                  # Streamlit application

├── src/

│   ├── rfm_features.py         # RFM feature engineering

│   ├── clustering.py           # K-Means & DBSCAN logic

│   └── visualize.py            # Plotly visualization functions

├── notebook/

│   └── eda_segmentation.ipynb  # Full ML pipeline

├── models/

│   ├── kmeans_model.pkl        # Trained K-Means model

│   └── scaler.pkl              # Fitted StandardScaler

├── data/                       # Dataset (gitignored)

├── requirements.txt

└── README.md

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Satya-05/customer-segmentation.git
cd customer-segmentation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```
