import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_correlation(df, save_path="assets/corr_heatmap.png"):
    """Plot correlation heatmap for numeric columns"""
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation analysis")
    
    # Remove columns with constant values (std = 0)
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        raise ValueError("Insufficient numeric columns with variance for correlation analysis")
    
    plt.figure(figsize=(max(10, len(numeric_df.columns)), max(6, len(numeric_df.columns) * 0.8)))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt='.2f', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_sales_by_brand(df, save_path="assets/sales_by_brand.png"):
    """Plot sales by brand"""
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Find brand and sales columns
    brand_col = None
    sales_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if brand_col is None and ('brand' in col_lower or 'manufacturer' in col_lower):
            brand_col = col
        if sales_col is None and ('sales' in col_lower or 'quantity' in col_lower or 'units' in col_lower):
            sales_col = col
    
    if brand_col is None or sales_col is None:
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}. Need 'brand' and 'sales' columns.")
    
    # Aggregate sales by brand
    brand_sales = df.groupby(brand_col)[sales_col].sum().reset_index()
    brand_sales.columns = ['brand', 'sales']
    brand_sales = brand_sales.sort_values('sales', ascending=False)
    
    # Limit to top 20 brands for readability
    if len(brand_sales) > 20:
        brand_sales = brand_sales.head(20)
    
    plt.figure(figsize=(max(12, len(brand_sales) * 0.6), 6))
    sns.barplot(data=brand_sales, x="brand", y="sales", palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Sales by Brand", fontsize=14, pad=20)
    plt.xlabel("Brand", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_price_distribution(df, save_path="assets/price_dist.png"):
    """Plot price distribution"""
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Find price column
    price_col = None
    for col in df.columns:
        col_lower = col.lower()
        if price_col is None and ('price' in col_lower):
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"Missing required column: 'price'. Found columns: {df.columns.tolist()}")
    
    # Remove missing prices and outliers
    prices = df[price_col].dropna()
    prices = prices[prices > 0]  # Remove zero or negative prices
    
    if prices.empty:
        raise ValueError("No valid price data found")
    
    # Remove extreme outliers (beyond 3 standard deviations)
    mean_price = prices.mean()
    std_price = prices.std()
    if std_price > 0:
        prices = prices[(prices >= mean_price - 3*std_price) & (prices <= mean_price + 3*std_price)]
    
    if prices.empty:
        raise ValueError("No valid price data after outlier removal")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(prices, bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title("Price Distribution", fontsize=14, pad=20)
    plt.xlabel("Price", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
