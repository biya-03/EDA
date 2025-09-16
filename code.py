
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

st.set_page_config(page_title="Universal EDA", layout="wide")
st.title("📊 Universal EDA — Upload any dataset (CSV / Excel / JSON)")

uploaded_file = st.file_uploader("Upload a file", type=['csv', 'xlsx', 'xls', 'json', 'txt'])

if uploaded_file is None:
    st.info("Upload a CSV, Excel, or JSON file to start exploring.")
    st.stop()

# ---------- Read file robustly ----------
try:
    uploaded_file.seek(0)
    fname = uploaded_file.name.lower()
    if fname.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif fname.endswith('.json'):
        df = pd.read_json(uploaded_file)
    else:
        # Try automatic delimiter detection (engine='python' with sep=None)
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            # Try common encodings as fallback
            uploaded_file.seek(0)
            df = None
            for enc in ("utf-8", "latin1", "cp1252"):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except Exception:
                    continue
            if df is None:
                # Last resort: decode to text and read
                uploaded_file.seek(0)
                text = uploaded_file.read().decode('utf-8', errors='replace')
                df = pd.read_csv(io.StringIO(text))
    # ensure column names are strings
    df.columns = df.columns.map(str)
except Exception as e:
    st.error("Couldn't read the uploaded file. See error below:")
    st.exception(e)
    st.stop()

# ---------- Basic overview ----------
st.subheader("Dataset preview")
st.write(df.head())

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Memory (MB)", round(df.memory_usage(deep=True).sum() / (1024**2), 2))

st.subheader("Column info")
buf = io.StringIO()
df.info(buf=buf)
st.text(buf.getvalue())

# Optional: show describe
if st.checkbox("Show summary statistics (describe)"):
    st.subheader("Summary statistics")
    st.write(df.describe(include='all').transpose())

# ---------- Missing values ----------
st.subheader("Missing values")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
st.dataframe(missing_df.sort_values("missing_count", ascending=False))

if st.checkbox("Show missing-values heatmap (sample if large)"):
    # sample if huge
    max_cells = 5000  # threshold to avoid insane heatmaps
    if df.shape[0] * df.shape[1] > max_cells:
        sample_n = min(1000, df.shape[0])
        df_sample = df.sample(sample_n, random_state=1)
        st.caption(f"Large dataset — heatmap shown for a sample of {sample_n} rows.")
    else:
        df_sample = df
    fig, ax = plt.subplots(figsize=(12, max(2, df_sample.shape[1] / 3)))
    sns.heatmap(df_sample.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_xlabel("Columns")
    st.pyplot(fig)

# ---------- Duplicates ----------
dup_count = int(df.duplicated().sum())
st.write(f"Duplicate rows: **{dup_count}**")
if dup_count > 0 and st.button("Show duplicate rows"):
    st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)))

# ---------- Column categories ----------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

st.sidebar.header("Plot & Analysis Options")
plot_type = st.sidebar.selectbox("Choose action", [
    "— select —",
    "Univariate analysis",
    "Bivariate analysis",
    "Correlation heatmap",
    "Pairplot (sampled)"
])

# ---------- Univariate ----------
if plot_type == "Univariate analysis":
    col = st.sidebar.selectbox("Select column", df.columns.tolist())
    st.subheader(f"Univariate: {col}")
    if pd.api.types.is_numeric_dtype(df[col]):
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[col], ax=ax2)
        st.pyplot(fig2)
        st.write(df[col].describe())
    else:
        vc = df[col].value_counts().nlargest(30)
        fig, ax = plt.subplots(figsize=(8, min(12, 0.35 * len(vc))))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_xlabel("count")
        st.pyplot(fig)
        st.write("Unique values:", df[col].nunique())

# ---------- Bivariate ----------
elif plot_type == "Bivariate analysis":
    st.subheader("Bivariate Analysis")
    x = st.sidebar.selectbox("X column", df.columns.tolist(), index=0)
    y = st.sidebar.selectbox("Y column", df.columns.tolist(), index=1 if df.shape[1] > 1 else 0)
    kind = st.sidebar.selectbox("Plot kind", ["scatter", "regplot", "box", "violin", "countplot"])
    if kind in ("scatter", "regplot"):
        if x in num_cols and y in num_cols:
            fig, ax = plt.subplots()
            if kind == "scatter":
                ax.scatter(df[x], df[y], alpha=0.6, s=20)
            else:
                sns.regplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Scatter/regplot require numeric columns for both X and Y.")
    elif kind in ("box", "violin"):
        fig, ax = plt.subplots(figsize=(8, 6))
        if kind == "box":
            sns.boxplot(x=df[x], y=df[y], ax=ax)
        else:
            sns.violinplot(x=df[x], y=df[y], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:  # countplot
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            sns.countplot(x=df[x], hue=df[y], data=df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error("Could not draw countplot (likely too many categories).")
            st.exception(e)

# ---------- Correlation heatmap ----------
elif plot_type == "Correlation heatmap":
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for correlation.")
    else:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, max(6, len(num_cols) / 2)))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

# ---------- Pairplot ----------
elif plot_type == "Pairplot (sampled)":
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for pairplot.")
    else:
        max_cols = min(6, len(num_cols))
        n = st.sidebar.slider("Number of numeric columns to include", 2, max_cols, value=min(4, max_cols))
        selected = st.sidebar.multiselect("Select numeric columns", num_cols, default=num_cols[:n])
        sample_n = min(500, len(df))
        st.caption(f"Pairplot uses a random sample of {sample_n} rows for speed.")
        pairgrid = sns.pairplot(df[selected].sample(sample_n, random_state=1))
        st.pyplot(pairgrid.fig)

# ---------- Dataframe view ----------
if st.checkbox("Show full dataframe (paginated)"):
    st.dataframe(df)

st.sidebar.header("Columns")
st.sidebar.write(f"Numeric ({len(num_cols)})")
st.sidebar.write(num_cols)
st.sidebar.write(f"Categorical ({len(cat_cols)})")
st.sidebar.write(cat_cols)

st.success("EDA ready — use the sidebar to explore different plots and checks 🚀")
