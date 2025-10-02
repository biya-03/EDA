# 📊 Comprehensive E-commerce Dataset EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ---------- Load Dataset ----------
file_path = "ecommerce_dataset.csv"   # change path if needed
df = pd.read_csv(file_path)

# ---------- Initial Exploration ----------
print("🔎 Dataset Shape:", df.shape)
print("\n📌 First 5 rows:")
print(df.head())
print("\n📌 Last 5 rows:")
print(df.tail())
print("\n📌 Data Info:")
print(df.info())
print("\n📌 Summary Statistics:")
print(df.describe(include="all").transpose())

# ---------- Data Cleaning ----------
# Remove duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\n🧹 Removed {before - after} duplicate rows.")

# Handle missing values (basic check)
print("\n❓ Missing Values per Column:")
print(df.isnull().sum())

# Example cleaning: fill NA in categorical with "Unknown"
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# Fill numeric NA with mean (if any)
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# Convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# Compute revenue
df['revenue'] = df['quantity'] * df['price'] * (1 - df['discount'])

# Final check
print("\n✅ Cleaned Data Info:")
print(df.info())

# ---------- Visualization 1: Sales Trend ----------
sales_trend = df.groupby(df['order_date'].dt.date)['revenue'].sum()

plt.figure(figsize=(12,5))
sales_trend.plot(color="teal")
plt.title("📈 Daily Revenue Trend")
plt.ylabel("Revenue")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# ---------- Visualization 2: Revenue by Category ----------
category_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=category_rev.values, y=category_rev.index, palette="viridis")
plt.title("💰 Revenue by Category")
plt.xlabel("Revenue")
plt.tight_layout()
plt.show()

# ---------- Visualization 3: Region vs Category Revenue ----------
region_cat_rev = df.pivot_table(values="revenue", index="region", columns="category", aggfunc="sum")

plt.figure(figsize=(10,6))
sns.heatmap(region_cat_rev, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("🌍 Region vs Category Revenue")
plt.tight_layout()
plt.show()

# ---------- Visualization 4: Payment Method Preferences ----------
payment_counts = df['payment_method'].value_counts()

plt.figure(figsize=(6,6))
payment_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap="Set3")
plt.title("💳 Payment Method Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ---------- Visualization 5: Top 10 Products by Revenue ----------
product_rev = df.groupby('product_id')['revenue'].sum().nlargest(10)

plt.figure(figsize=(10,5))
sns.barplot(x=product_rev.index.astype(str), y=product_rev.values, palette="coolwarm")
plt.title("🏆 Top 10 Products by Revenue")
plt.xlabel("Product ID")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()


    
               

        

   
    

      
