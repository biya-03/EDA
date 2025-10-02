import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("📊 E-commerce Data Exploration App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # ---- Data Cleaning ----
    st.subheader("🧹 Data Cleaning")
    st.write("Dropping duplicates and handling empty rows...")
    df = df.drop_duplicates()
    df = df.dropna(how="all")

    # ---- Preview Data ----
    st.subheader("👀 Data Preview")
    st.write("First 5 Rows of Data:")
    st.dataframe(df.head())

    st.write("Last 5 Rows of Data:")
    st.dataframe(df.tail())

    # ---- Shape of Dataset ----
    st.subheader("📐 Shape of Dataset")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # ---- Data Types ----
    st.subheader("🔠 Data Types of Columns")
    st.write(df.dtypes)

    # ---- Missing Values ----
    st.subheader("❌ Missing Values Check")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    st.write(pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct}))

    # ---- Duplicate Records ----
    st.subheader("📋 Duplicate Records Check")
    dup_count = df.duplicated().sum()
    st.write(f"Duplicate Rows: {dup_count}")

    # ---- Unique Values ----
    st.subheader("🔍 Unique Values per Column")
    unique_counts = df.nunique()
    st.write(unique_counts)

    # ---- Summary Statistics ----
    st.subheader("📈 Summary Statistics")
    st.write("Standard Describe:")
    st.write(df.describe(include="all"))

    st.write("Extended Summary (Mean, Median, Mode, Std, Min, Max, Quantiles):")
    summary_stats = pd.DataFrame({
        "mean": df.mean(numeric_only=True),
        "median": df.median(numeric_only=True),
        "mode": df.mode().iloc[0],
        "std": df.std(numeric_only=True),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
        "25%": df.quantile(0.25, numeric_only=True),
        "50%": df.quantile(0.50, numeric_only=True),
        "75%": df.quantile(0.75, numeric_only=True),
    })
    st.write(summary_stats)

    # ---- Correlation Heatmap ----
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for correlation heatmap.")

else:
    st.warning("👆 Please upload a CSV file to begin.")
