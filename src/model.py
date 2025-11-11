from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def train_price_model(df):
    # Check required columns - try different possible column name variations
    column_mapping = {
        'battery_kwh': ['battery_kwh', 'battery', 'Battery (kWh)', 'battery_kWh'],
        'range_km': ['range_km', 'range', 'Range (km)', 'range_KM'],
        'year': ['year', 'Year', 'YEAR'],
        'acceleration': ['acceleration', 'accel', 'Acceleration', '0-100 km/h'],
        'brand': ['brand', 'Brand', 'BRAND', 'manufacturer', 'Manufacturer'],
        'price': ['price', 'Price', 'PRICE', 'price_usd', 'Price (USD)']
    }
    
    # Find actual column names
    actual_cols = {}
    for key, possible_names in column_mapping.items():
        found = False
        for name in possible_names:
            if name in df.columns:
                actual_cols[key] = name
                found = True
                break
        if not found:
            raise ValueError(f"Missing required column: {key}. Tried: {possible_names}")
    
    # Features for price prediction
    feature_cols = [actual_cols['battery_kwh'], actual_cols['range_km'], 
                    actual_cols['year'], actual_cols['acceleration'], actual_cols['brand']]
    X = df[feature_cols].copy()
    
    # Rename columns for consistency
    X.columns = ['battery_kwh', 'range_km', 'year', 'acceleration', 'brand']
    
    # Handle missing values in features
    numeric_features = ['battery_kwh', 'range_km', 'year', 'acceleration']
    for col in numeric_features:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.notna(median_val):
                X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(0)
    
    # Encode categorical variables
    X = pd.get_dummies(X, columns=['brand'], drop_first=True, dtype=int)
    
    # Target variable
    y = df[actual_cols['price']].copy()
    if y.isna().any():
        median_price = y.median()
        if pd.notna(median_price):
            y = y.fillna(median_price)
        else:
            y = y.fillna(0)
    
    # Remove rows where target is still missing or zero
    valid_idx = ~y.isna() & (y > 0)
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) == 0:
        raise ValueError("No valid data for training after preprocessing")
    
    if len(X) < 10:
        raise ValueError(f"Insufficient data for training. Only {len(X)} valid samples available.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    # Calculate RMSE manually (square root of MSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return model, rmse

def forecast_sales(df):
    # Check required columns - try different possible column name variations
    year_col = None
    sales_col = None
    date_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if year_col is None and ('year' in col_lower or col_lower == 'y'):
            year_col = col
        if sales_col is None and ('sales' in col_lower or 'quantity' in col_lower or 'units' in col_lower):
            sales_col = col
        if date_col is None and col_lower == 'date':
            date_col = col
    
    if year_col is None or sales_col is None:
        raise ValueError(f"Missing required columns. Found columns: {df.columns.tolist()}. Need 'year' and 'sales' columns.")
    
    # If we have a Date column with monthly data, use that for better forecasting
    if date_col and date_col in df.columns:
        try:
            # Parse date and extract year-month for time series
            df_temp = df.copy()
            df_temp['date_parsed'] = pd.to_datetime(df_temp[date_col], format='%Y-%m', errors='coerce')
            df_temp = df_temp.dropna(subset=['date_parsed'])
            df_temp['year_month'] = df_temp['date_parsed'].dt.to_period('M')
            
            # Aggregate by year-month
            sales_monthly = df_temp.groupby('year_month')[sales_col].sum().reset_index()
            sales_monthly = sales_monthly.sort_values('year_month')
            
            # Convert to numeric for forecasting (months since start)
            sales_monthly['month_num'] = range(len(sales_monthly))
            
            if len(sales_monthly) >= 3:  # Need at least 3 months
                X = sales_monthly[['month_num']].values
                y = sales_monthly[sales_col].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast next 12 months (1 year ahead)
                last_month_num = sales_monthly['month_num'].max()
                future_months = np.array([last_month_num + i + 1 for i in range(12)]).reshape(-1, 1)
                forecast_monthly = model.predict(future_months)
                forecast_monthly = np.maximum(forecast_monthly, 0)
                
                # Aggregate to yearly forecast
                forecast_yearly_sum = forecast_monthly.sum()
                # Estimate next year as average of forecasted months * 12
                forecast_next_year = forecast_monthly.mean() * 12
                # Estimate year after as trend continuation
                forecast_year_after = forecast_next_year * 1.1  # Assume 10% growth
                
                # Create yearly summary for display
                max_year = int(df_temp['date_parsed'].dt.year.max())
                sales_yearly = df_temp.groupby(df_temp['date_parsed'].dt.year)[sales_col].sum().reset_index()
                sales_yearly.columns = ['year', 'sales']
                
                future_years = np.array([max_year + 1, max_year + 2])
                forecast = np.array([forecast_next_year, forecast_year_after])
                
                return sales_yearly, future_years, forecast
        except Exception as e:
            # Fall back to yearly aggregation if monthly parsing fails
            pass
    
    # Fallback: Aggregate yearly sales
    sales_yearly = df.groupby(year_col)[sales_col].sum().reset_index()
    sales_yearly.columns = ['year', 'sales']
    
    # Remove rows with missing or invalid data
    sales_yearly = sales_yearly.dropna()
    sales_yearly = sales_yearly[sales_yearly['sales'] > 0]
    
    if len(sales_yearly) < 1:
        raise ValueError(f"Insufficient data for forecasting. Found {len(sales_yearly)} valid years.")
    
    # Sort by year
    sales_yearly = sales_yearly.sort_values('year')
    
    # If only one year of data, use average monthly sales * 12 for next year
    if len(sales_yearly) == 1:
        current_year_sales = sales_yearly['sales'].iloc[0]
        # Estimate next year with 10% growth assumption
        forecast_next = current_year_sales * 1.1
        forecast_after = forecast_next * 1.1
        
        max_year = int(sales_yearly['year'].max())
        future_years = np.array([max_year + 1, max_year + 2])
        forecast = np.array([forecast_next, forecast_after])
        
        return sales_yearly, future_years, forecast
    
    # Multiple years: use linear regression
    X = sales_yearly[['year']].values
    y = sales_yearly['sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast for next 2 years
    max_year = int(sales_yearly['year'].max())
    future_years = np.array([max_year + 1, max_year + 2]).reshape(-1, 1)
    forecast = model.predict(future_years)
    
    # Ensure forecast values are non-negative
    forecast = np.maximum(forecast, 0)
    
    return sales_yearly, future_years.flatten(), forecast
