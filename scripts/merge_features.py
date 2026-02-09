import os
import pandas as pd
from pathlib import Path

def merge_features():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    print("[LOG] Starting safe merge of feature files...")
    
    feature_files = list(data_dir.glob('features_*.csv'))
    if not feature_files:
        print("[ERROR] No features_*.csv files found in data/")
        return
    
    print(f"[LOG] Found {len(feature_files)} feature files:")
    for f in feature_files:
        print(f"  - {f.name}")
    
    output_path = data_dir / 'merged_features.csv'
    
    first_file = True
    total_rows = 0
    
    # Columns to keep (adjust if you want more/less)
    keep_cols = [
        'timestamp', 'can_id', 'dlc', 'delta_t', 'entropy',
        'hamming_dist', 'delta_t_mean_10', 'delta_t_std_10',
        'attack_type'
        # Add any data_* if you want them; skip flag/raw payload for now
    ]
    
    for file_path in feature_files:
        print(f"[LOG] Merging: {file_path.name}")
        
        chunksize = 100000
        for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=lambda c: c in keep_cols):
            # Optional: memory optimizations
            if 'can_id' in chunk.columns:
                chunk['can_id'] = chunk['can_id'].astype('category')
            if 'attack_type' in chunk.columns:
                chunk['attack_type'] = chunk['attack_type'].astype('category')
            
            # Append to output
            mode = 'w' if first_file else 'a'
            header = first_file
            chunk.to_csv(output_path, mode=mode, header=header, index=False)
            total_rows += len(chunk)
            print(f"[LOG]   Added {len(chunk)} rows (total so far: {total_rows:,})")
            
            first_file = False
        
        print(f"[LOG] Finished {file_path.name}")
    
    print(f"\n[LOG] Merge complete!")
    print(f"[LOG] Merged dataset saved to: {output_path}")
    print(f"[LOG] Total rows: {total_rows:,}")
    
    # Quick summary stats
    print("\n[LOG] Quick stats from merged file (first chunk):")
    df_sample = pd.read_csv(output_path, nrows=1000)
    print(df_sample.describe(include='all'))
    print("\nMissing values:")
    print(df_sample.isna().sum())

if __name__ == "__main__":
    merge_features()