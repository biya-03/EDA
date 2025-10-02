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

    # ---- Data Info ----
    st.subheader("ℹ️ Dataset Information")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Columns:**", df.columns.tolist())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include="all"))

    # ---- Missing Values ----
    st.subheader("❌ Missing Values Check")
    st.write(df.isnull().sum())

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






   
    

      
