"""
Script to download EV Sales and Adoption dataset from Kaggle
"""
import os
import sys
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

def download_kaggle_dataset():
    """
    Download the EV Sales and Adoption dataset from Kaggle
    Dataset: rameezmeerasahib/electric-vehicle-ev-sales-and-adoption
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Dataset information
    dataset = "rameezmeerasahib/electric-vehicle-ev-sales-and-adoption"
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading dataset: {dataset}")
    print(f"Destination: {data_dir}/")
    
    try:
        # Download dataset files
        api.dataset_download_files(dataset, path=data_dir, unzip=True)
        
        # Find the CSV file(s) in the data directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if csv_files:
            # If multiple CSV files, find the main one or use the first
            # Look for common names
            main_csv = None
            for name in ['ev_sales_adoption.csv', 'electric_vehicle_sales.csv', 'EV_Sales.csv']:
                if name in csv_files:
                    main_csv = name
                    break
            
            # If no specific match, use the first CSV file
            if main_csv is None:
                main_csv = csv_files[0]
            
            # If the file is not already named ev_sales_adoption.csv, rename it
            if main_csv != 'ev_sales_adoption.csv':
                source_path = os.path.join(data_dir, main_csv)
                target_path = os.path.join(data_dir, 'ev_sales_adoption.csv')
                
                # Remove old file if it exists
                if os.path.exists(target_path):
                    os.remove(target_path)
                
                # Rename the downloaded file
                os.rename(source_path, target_path)
                print(f"Renamed {main_csv} to ev_sales_adoption.csv")
            
            # Remove other CSV files if multiple were downloaded
            for csv_file in csv_files:
                if csv_file != 'ev_sales_adoption.csv':
                    file_path = os.path.join(data_dir, csv_file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Removed duplicate file: {csv_file}")
            
            # Remove zip file if it exists
            zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
            for zip_file in zip_files:
                zip_path = os.path.join(data_dir, zip_file)
                os.remove(zip_path)
                print(f"Removed zip file: {zip_file}")
            
            # Remove any subdirectories that might have been created
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and item != '.' and item != '..':
                    # Check if there are CSV files in the subdirectory
                    sub_csv_files = [f for f in os.listdir(item_path) if f.endswith('.csv')]
                    if sub_csv_files:
                        # Move CSV files to main data directory
                        for csv_file in sub_csv_files:
                            src = os.path.join(item_path, csv_file)
                            dst = os.path.join(data_dir, csv_file)
                            if not os.path.exists(dst) or csv_file == 'ev_sales_adoption.csv':
                                shutil.move(src, dst)
                    # Remove the subdirectory
                    shutil.rmtree(item_path)
                    print(f"Removed subdirectory: {item}")
            
            print(f"\n✅ Dataset downloaded successfully!")
            print(f"📁 File location: {os.path.join(data_dir, 'ev_sales_adoption.csv')}")
            
            # Verify the file
            import pandas as pd
            df = pd.read_csv(os.path.join(data_dir, 'ev_sales_adoption.csv'))
            print(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"📋 Columns: {', '.join(df.columns.tolist()[:10])}")
            if len(df.columns) > 10:
                print(f"    ... and {len(df.columns) - 10} more columns")
            
        else:
            print("⚠️ Warning: No CSV files found in the downloaded dataset")
            
    except OSError as e:
        if "Could not find kaggle.json" in str(e):
            print(f"❌ {str(e)}")
            print("\n" + "="*60)
            print("KAGGLE API CREDENTIALS NOT SET UP")
            print("="*60)
            print("\nOption 1: Set up Kaggle API (for automatic download)")
            print("  1. Go to: https://www.kaggle.com/settings")
            print("  2. Scroll to 'API' section → Click 'Create New Token'")
            print("  3. This downloads kaggle.json to your Downloads folder")
            print("  4. Create folder: C:\\Users\\YourUsername\\.kaggle")
            print("  5. Copy kaggle.json to that folder")
            print("  6. Run this script again")
            print("\nOption 2: Manual Download (Easier!)")
            print("  1. Visit: https://www.kaggle.com/datasets/rameezmeerasahib/electric-vehicle-ev-sales-and-adoption")
            print("  2. Click 'Download' button (accept terms if prompted)")
            print("  3. Extract the ZIP file")
            print("  4. Find the CSV file inside")
            print("  5. Copy it to: data/ev_sales_adoption.csv")
            print("  6. Replace the existing empty file")
            print("\n" + "="*60)
            print("\n💡 TIP: Run 'python download_data_manual.py' for interactive help")
            sys.exit(1)
        else:
            print(f"❌ Error downloading dataset: {str(e)}")
            raise
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("\nPlease make sure:")
        print("1. You have Kaggle API credentials set up")
        print("2. You have accepted the dataset terms on Kaggle")
        print("3. Your internet connection is working")
        print("\nOr download manually from:")
        print("https://www.kaggle.com/datasets/rameezmeerasahib/electric-vehicle-ev-sales-and-adoption")
        raise

if __name__ == "__main__":
    download_kaggle_dataset()

