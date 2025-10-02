import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🛒 Advanced E-commerce Data Dashboard")

# Upload file
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # ---- Data Cleaning ----
    st.subheader("🧹 Data Cleaning")
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    st.write("✅ Duplicates removed & empty rows dropped.")

    # ---- Data Preview ----
    st.subheader("👀 Dataset Preview")
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())
    st.write("**Last 5 Rows:**")
    st.dataframe(df.tail())

    # ---- Data Info ----
    st.subheader("📋 Dataset Information")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    st.write("Summary Statistics:")
    st.write(df.describe(include="all"))

    # ---- Missing Values ----
    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    # ---- Correlation Heatmap ----
    st.subheader("🔥 Correlation Heatmap (Numerical Features)")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns for correlation heatmap.")

    # ---- Revenue Calculations ----
    if 'Revenue' in df.columns:
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # 📈 Daily Revenue Trend
        if 'Date' in df.columns:
            st.subheader("📈 Daily Revenue Trend")
            daily_revenue = df.groupby('Date')['Revenue'].sum()
            fig, ax = plt.subplots(figsize=(10, 4))
            daily_revenue.plot(ax=ax)
            ax.set_title("Daily Revenue Trend")
            ax.set_ylabel("Revenue")
            st.pyplot(fig)

        # 💰 Revenue by Category
        if 'Category' in df.columns:
            st.subheader("💰 Revenue by Category")
            category_revenue = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            category_revenue.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Revenue by Category")
            st.pyplot(fig)

        # 🌍 Region vs Category Revenue Heatmap
        if 'Region' in df.columns and 'Category' in df.columns:
            st.subheader("🌍 Region vs. Category Revenue (Heatmap)")
            pivot_table = df.pivot_table(values='Revenue', index='Region', columns='Category', aggfunc='sum', fill_value=0)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
            st.pyplot(fig)

        # 💳 Payment Method Distribution
        if 'Payment_Method' in df.columns:
            st.subheader("💳 Payment Method Distribution")
            payment_counts = df['Payment_Method'].value_counts()
            fig, ax = plt.subplots()
            payment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
            ax.set_ylabel("")
            ax.set_title("Payment Methods")
            st.pyplot(fig)

        # 🏆 Top 10 Products by Revenue
        if 'Product' in df.columns:
            st.subheader("🏆 Top 10 Products by Revenue")
            top_products = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            top_products.plot(kind='bar', ax=ax, color='orange')
            ax.set_title("Top 10 Products by Revenue")
            st.pyplot(fig)

    else:
        st.warning("⚠️ 'Revenue' column not found in dataset. Please make sure your dataset contains Revenue, Category, Product, Date, Region, and Payment_Method columns.")

else:
    st.info("👆 Upload a CSV file to start exploring your data.")

  

  

