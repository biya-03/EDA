import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto EDA App", layout="wide")

st.title("📊 Automated EDA App")
st.write("Upload any CSV file and explore it instantly!")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    st.subheader("📐 Shape of Dataset")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("📝 Column Info")
    buffer = []
    df.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # -------------------------------
    # Missing values
    # -------------------------------
    st.subheader("🚨 Missing Values")
    st.write(df.isnull().sum())

    # -------------------------------
    # Categorical columns
    # -------------------------------
    st.subheader("📂 Categorical Features Distribution")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        st.write(f"**{col}**")
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # Numerical columns
    # -------------------------------
    st.subheader("📈 Numerical Features Distribution")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        st.write(f"**{col}**")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # Correlation heatmap
    # -------------------------------
    if len(num_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to start EDA.")

  
