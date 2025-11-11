import pandas as pd

def _find_column(df, possible_names):
    """Helper function to find a column by trying multiple possible names"""
    for name in possible_names:
        if name in df.columns:
            return name
    # Try case-insensitive search
    df_cols_lower = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_cols_lower:
            return df_cols_lower[name.lower()]
    return None

def chatbot(df, query):
    """Simple rule-based chatbot for EV queries"""
    if df.empty:
        return "No data available to answer your question."
    
    query_lower = query.lower()
    
    try:
        if "average price" in query_lower or "mean price" in query_lower:
            price_col = _find_column(df, ['price', 'Price', 'PRICE', 'price_usd', 'Price (USD)'])
            if price_col:
                prices = df[price_col].dropna()
                prices = prices[prices > 0]  # Remove invalid prices
                if not prices.empty:
                    avg_price = prices.mean()
                    return f"The average EV price is ${avg_price:,.2f}"
                else:
                    return "No valid price data available."
            else:
                return "Price data is not available in the dataset."
        
        elif "highest sales" in query_lower or "top sales" in query_lower:
            model_col = _find_column(df, ['model', 'Model', 'MODEL'])
            sales_col = _find_column(df, ['sales', 'Sales', 'SALES', 'quantity', 'units'])
            brand_col = _find_column(df, ['brand', 'Brand', 'BRAND', 'manufacturer'])
            
            if model_col and sales_col:
                sales_data = df.groupby(model_col)[sales_col].sum()
                sales_data = sales_data[sales_data > 0]
                if not sales_data.empty:
                    top_model = sales_data.idxmax()
                    top_sales = sales_data.max()
                    return f"The model with highest sales is {top_model} with {top_sales:,.0f} units sold."
            
            if brand_col and sales_col:
                sales_data = df.groupby(brand_col)[sales_col].sum()
                sales_data = sales_data[sales_data > 0]
                if not sales_data.empty:
                    top_brand = sales_data.idxmax()
                    top_sales = sales_data.max()
                    return f"The brand with highest sales is {top_brand} with {top_sales:,.0f} units sold."
            
            return "Sales data is not available in the dataset."
        
        elif "forecast" in query_lower or "future sales" in query_lower:
            year_col = _find_column(df, ['year', 'Year', 'YEAR'])
            sales_col = _find_column(df, ['sales', 'Sales', 'SALES', 'quantity', 'units'])
            
            if year_col and sales_col:
                sales_yearly = df.groupby(year_col)[sales_col].sum().reset_index()
                sales_yearly = sales_yearly[sales_yearly[sales_col] > 0]
                if not sales_yearly.empty:
                    latest_year = sales_yearly[year_col].max()
                    last_year_sales = sales_yearly[sales_yearly[year_col] == latest_year][sales_col].values[0]
                    return f"Latest year ({int(latest_year)}) total sales: {last_year_sales:,.0f} units"
                else:
                    return "Insufficient sales data for forecasting."
            else:
                return "Year or sales data is not available in the dataset."
        
        elif "brand" in query_lower:
            brand_col = _find_column(df, ['brand', 'Brand', 'BRAND', 'manufacturer', 'Manufacturer'])
            if brand_col:
                brands = df[brand_col].dropna().unique()
                brands = [str(b) for b in brands if str(b) != 'nan']
                if brands:
                    brand_list = ', '.join(brands[:10])
                    more_text = f" and {len(brands)-10} more." if len(brands) > 10 else "."
                    return f"Available brands in the dataset: {brand_list}{more_text}"
                else:
                    return "No brand information found in the dataset."
            else:
                return "Brand information is not available in the dataset."
        
        elif "model" in query_lower:
            model_col = _find_column(df, ['model', 'Model', 'MODEL'])
            if model_col:
                models = df[model_col].dropna().unique()
                models = [str(m) for m in models if str(m) != 'nan']
                if models:
                    model_list = ', '.join(models[:5])
                    return f"Total models in dataset: {len(models)}. Some examples: {model_list}"
                else:
                    return "No model information found in the dataset."
            else:
                return "Model information is not available in the dataset."
        
        else:
            return "I can help you with questions about average prices, highest sales, sales forecasts, brands, and models. Please try rephrasing your question."
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"