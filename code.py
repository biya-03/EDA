import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from wordcloud import WordCloud

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.title("📊 Advanced E-commerce EDA Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # ---- Sidebar Options ----
    st.sidebar.header("⚙️ Data Cleaning Options")
    if st.sidebar.checkbox("Drop Duplicates"):
        df = df.drop_duplicates()

    missing_option = st.sidebar.selectbox("Handle Missing Values", 
                                          ["Do Nothing", "Drop Rows", "Fill with Mean", "Fill with Mode"])

    if missing_option == "Drop Rows":
        df = df.dropna()
    elif missing_option == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_option == "Fill with Mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ---- Dataset Preview ----
    st.subheader("👀 Data Preview")
    st.write("First 5 Rows:")
    st.dataframe(df.head())
    st.write("Last 5 Rows:")
    st.dataframe(df.tail())

    # ---- Data Info ----
    st.subheader("ℹ️ Dataset Information")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {df.columns.tolist()}")
    st.write("**Summary Statistics:**")
    st.write(df.describe(include="all"))

    # ---- Missing Values ----
    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    # ---- Column Explorer ----
    st.subheader("🔍 Explore a Column")
    column = st.selectbox("Select Column", df.columns)

    if df[column].dtype == "object":
        st.write(df[column].value_counts().head(10))
        fig, ax = plt.subplots()
        df[column].value_counts().head(10).plot(kind="bar", ax=ax)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

    # ---- Correlation Heatmap ----
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for correlation heatmap.")

    # ---- Pairplot ----
    if st.checkbox("Show Pairplot (Numeric Features Only)"):
        sns.pairplot(numeric_df)
        st.pyplot()

    # ---- Word Cloud ----
    st.subheader("☁️ Word Cloud (Text Data)")
    text_columns = df.select_dtypes(include="object").columns
    if len(text_columns) > 0:
        text_col = st.selectbox("Select a text column", text_columns)
        text_data = " ".join(str(val) for val in df[text_col].dropna())

        if text_data.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Selected column is empty, cannot generate Word Cloud.")
    else:
        st.info("No text columns available for Word Cloud.")

    # ---- Download Cleaned Dataset ----
    st.subheader("💾 Download Cleaned Dataset")
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button("Download CSV", buffer.getvalue(), "cleaned_dataset.csv", "text/csv")

else:
    st.warning("👆 Please upload a CSV file to begin.")
