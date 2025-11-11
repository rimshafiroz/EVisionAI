"""
Manual Data Download Helper
This script provides instructions and helps verify the download
"""
import os
import pandas as pd
import webbrowser
import sys

def check_data_file():
    """Check if data file exists and is valid"""
    file_path = "data/ev_sales_adoption.csv"
    
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # Less than 100 bytes is likely empty
            return False, "File is too small (likely empty)"
        
        # Try to read the file
        df = pd.read_csv(file_path)
        if df.empty:
            return False, "File exists but is empty"
        
        return True, f"File is valid! Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    
    except pd.errors.EmptyDataError:
        return False, "File has no valid columns"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def main():
    print("=" * 60)
    print("EV Sales Dataset Download Helper")
    print("=" * 60)
    print()
    
    # Check current status
    is_valid, message = check_data_file()
    
    if is_valid:
        print(f"✅ {message}")
        print("\nYour dataset is ready to use!")
        return
    
    print(f"❌ {message}")
    print()
    print("=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("Step 1: Open the Kaggle dataset page")
    print("   URL: https://www.kaggle.com/datasets/rameezmeerasahib/electric-vehicle-ev-sales-and-adoption")
    print()
    print("Step 2: Download the dataset")
    print("   - Click the 'Download' button (you may need to accept terms first)")
    print("   - This will download a ZIP file")
    print()
    print("Step 3: Extract and place the CSV file")
    print("   - Extract the ZIP file")
    print("   - Find the CSV file (usually named something like 'ev_sales.csv' or 'electric_vehicle_sales.csv')")
    print("   - Copy it to: data/ev_sales_adoption.csv")
    print("   - Replace the existing empty file if prompted")
    print()
    print("Step 4: Verify the download")
    print("   - Run this script again: python download_data_manual.py")
    print()
    
    # Ask if user wants to open browser
    response = input("Would you like to open the Kaggle dataset page in your browser? (y/n): ")
    if response.lower() == 'y':
        url = "https://www.kaggle.com/datasets/rameezmeerasahib/electric-vehicle-ev-sales-and-adoption"
        print(f"\nOpening {url} in your browser...")
        webbrowser.open(url)
        print("\nAfter downloading, place the CSV file in: data/ev_sales_adoption.csv")
        print("Then run this script again to verify.")

if __name__ == "__main__":
    main()

