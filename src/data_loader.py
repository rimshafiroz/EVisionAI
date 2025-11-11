import pandas as pd
import os
import numpy as np

def load_data(file_path="data/train.csv"):
    """
    Load EV sales dataset from train.csv
    """
    # Try train.csv first, then fall back to ev_sales_adoption.csv
    if not os.path.exists(file_path):
        # Try alternative file
        alt_path = "data/ev_sales_adoption.csv"
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(
                f"Data file not found at {file_path} or {alt_path}.\n"
                "Please ensure the dataset file exists in the data/ directory."
            )
    
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Data file at {file_path} is empty or has no valid columns.\n"
            "Please ensure the CSV file contains data."
        )
    except Exception as e:
        raise ValueError(
            f"Error reading data file: {str(e)}\n"
            "Please ensure the file is a valid CSV file with data."
        )
    
    if df.empty:
        raise ValueError(
            f"Data file at {file_path} is empty.\n"
            "Please ensure the CSV file contains data."
        )
    
    return df

def preprocess_data(df):
    """
    Preprocess EV sales data: transform columns to match expected format
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Transform train.csv format to expected format
    # Check if this is the train.csv format (has Date, Battery_Capacity_kWh, Units_Sold, Revenue)
    if 'Date' in df.columns and 'Battery_Capacity_kWh' in df.columns:
        # Extract year from Date (format: "2023-07" or "2023-01")
        if 'year' not in df.columns:
            df['year'] = pd.to_datetime(df['Date'], format='%Y-%m', errors='coerce').dt.year
            # If parsing fails, try extracting first 4 characters
            if df['year'].isna().any():
                df['year'] = df['Date'].str[:4].astype(int, errors='ignore')
        
        # Rename Battery_Capacity_kWh to battery_kwh
        if 'Battery_Capacity_kWh' in df.columns and 'battery_kwh' not in df.columns:
            df['battery_kwh'] = df['Battery_Capacity_kWh']
        
        # Rename Brand to brand (if needed)
        if 'Brand' in df.columns and 'brand' not in df.columns:
            df['brand'] = df['Brand']
        
        # Rename Units_Sold to sales (if needed)
        if 'Units_Sold' in df.columns and 'sales' not in df.columns:
            df['sales'] = df['Units_Sold']
        
        # Calculate price from Revenue/Units_Sold (average price per unit)
        if 'Revenue' in df.columns and 'Units_Sold' in df.columns:
            if 'price' not in df.columns:
                df['price'] = df['Revenue'] / df['Units_Sold']
                # Remove any infinite or invalid prices
                df['price'] = df['price'].replace([np.inf, -np.inf], np.nan)
        
        # Estimate range_km based on battery capacity if not present
        # Typical EV: ~5-7 km per kWh
        if 'range_km' not in df.columns and 'battery_kwh' in df.columns:
            # Use 6 km per kWh as average, with some variation
            df['range_km'] = df['battery_kwh'] * 6
            # Add some realistic variation (±20%)
            np.random.seed(42)  # For reproducibility
            variation = np.random.uniform(0.8, 1.2, len(df))
            df['range_km'] = (df['range_km'] * variation).astype(int)
        
        # Estimate acceleration based on battery capacity if not present
        # Larger batteries often in more powerful cars (faster acceleration)
        # Typical range: 3-12 seconds for 0-100 km/h
        if 'acceleration' not in df.columns and 'battery_kwh' in df.columns:
            # Inverse relationship: larger battery = faster (lower acceleration time)
            # Formula: acceleration = 12 - (battery_kwh - 40) / 10, clamped between 3 and 12
            battery_normalized = (df['battery_kwh'] - 40) / 10
            df['acceleration'] = 12 - battery_normalized
            df['acceleration'] = df['acceleration'].clip(lower=3, upper=12)
            # Add some variation
            np.random.seed(42)
            variation = np.random.uniform(0.9, 1.1, len(df))
            df['acceleration'] = (df['acceleration'] * variation).round(1)
    
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
            else:
                # If all values are NaN, fill with 0
                df[col] = df[col].fillna(0)
    
    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna('Unknown')
    
    return df