import pandas as pd
from pathlib import Path

def verify_datasets():
    project_root = Path(r"C:\Users\arun9\OneDrive\Desktop\Edge Ai")
    data_dir = project_root / 'data'
    
    # The file we want to check
    target_file = data_dir / 'merged_features.csv'

    print("\n" + "="*60)
    print(" [DIAGNOSTIC] DATASET VERIFICATION ENGINE")
    print("="*60)

    if not target_file.exists():
        print(f"[ERROR] File not found at: {target_file}")
        # Let's list what IS in the directory to help find it
        print(f"[INFO] Files available in /data/: {[f.name for f in data_dir.glob('*.csv')]}")
        return

    try:
        # Load just the first 5 rows to see the structure
        df_sample = pd.read_csv(target_file, nrows=5)
        
        print(f"✅ File Found: {target_file.name}")
        print(f"📊 Total Columns: {len(df_sample.columns)}")
        print(f"📝 Column Names: \n{list(df_sample.columns)}")
        print("\n🔎 First 2 Rows of Data:")
        print(df_sample.head(2).to_string(index=False))
        
        # Check for non-numeric columns that might cause errors
        non_numeric = df_sample.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric:
            print(f"\n⚠️ Non-Numeric Columns Found: {non_numeric} (These must be dropped or converted)")
        else:
            print("\n✅ All columns are numeric. Ready for ML.")

    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")

if __name__ == "__main__":
    verify_datasets()