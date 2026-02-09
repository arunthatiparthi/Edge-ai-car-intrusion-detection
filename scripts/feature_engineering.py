import os
import pandas as pd
import numpy as np
from pathlib import Path

def hex_to_int(hex_str):
    if pd.isna(hex_str):
        return 0  # Use 0 for missing bytes (common in CAN padding)
    try:
        return int(hex_str, 16)
    except:
        return 0

def compute_entropy_vectorized(bytes_array):
    """Vectorized entropy on uint8 array (rows = messages, cols = bytes)."""
    # bytes_array shape: (n_rows, 8), uint8
    # Compute per row
    entropy_vals = np.zeros(bytes_array.shape[0], dtype=np.float32)
    for i in range(bytes_array.shape[0]):
        row = bytes_array[i]
        counts = np.bincount(row, minlength=256)
        total = counts.sum()
        if total == 0:
            entropy_vals[i] = 0.0
            continue
        probs = counts / total
        probs = probs[probs > 0]
        entropy_vals[i] = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy_vals

def add_features(df_chunk):
    df = df_chunk.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Time delta per CAN ID
    df['delta_t'] = df.groupby('can_id')['timestamp'].diff()
    
    # Byte columns
    byte_cols = [f'data_{i}' for i in range(8)]
    
    # Convert hex to int, fill NaN with 0
    for col in byte_cols:
        df[col] = df[col].apply(hex_to_int).astype(np.uint8)
    
    # Previous payload per ID (as tuple for shift)
    df['payload_tuple'] = list(zip(*[df[col] for col in byte_cols]))
    df['prev_payload'] = df.groupby('can_id')['payload_tuple'].shift(1)
    
    # Hamming distance
    def calc_hamming(row):
        if pd.isna(row['prev_payload']):
            return 0
        curr = np.array(row[byte_cols], dtype=np.uint8)
        prev = np.array(row['prev_payload'], dtype=np.uint8)
        min_len = min(len(prev), len(curr))
        return np.sum(prev[:min_len] != curr[:min_len])
    
    df['hamming_dist'] = df.apply(calc_hamming, axis=1)
    
    # Entropy (vectorized over the uint8 matrix)
    bytes_matrix = df[byte_cols].to_numpy(dtype=np.uint8)
    df['entropy'] = compute_entropy_vectorized(bytes_matrix)
    
    # Rolling stats on delta_t
    df['delta_t_mean_10'] = df.groupby('can_id')['delta_t'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    df['delta_t_std_10'] = df.groupby('can_id')['delta_t'].transform(
        lambda x: x.rolling(10, min_periods=1).std()
    )
    
    # Clean up temp columns
    df = df.drop(columns=['payload_tuple', 'prev_payload'], errors='ignore')
    
    # Drop rows with NaN delta_t (first message per ID)
    df = df.dropna(subset=['delta_t'])
    
    # Memory optimization
    df['can_id'] = df['can_id'].astype('category')
    df['attack_type'] = df['attack_type'].astype('category')
    df['flag'] = df['flag'].astype('category')
    for col in byte_cols:
        df[col] = df[col].astype(np.uint8)
    
    return df

def process_file(input_path, output_path, chunksize=50000):
    print(f"[LOG] Processing: {input_path.name} → {output_path.name}")
    
    first_chunk = True
    chunk_num = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        print(f"[LOG]   Chunk {chunk_num}: {len(chunk)} rows")
        
        try:
            enriched = add_features(chunk)
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            enriched.to_csv(output_path, mode=mode, header=header, index=False)
            first_chunk = False
            del enriched, chunk
            print(f"[LOG]   Chunk {chunk_num} saved")
        except Exception as e:
            print(f"[ERROR] Failed in chunk {chunk_num}: {str(e)}")
            return False
    
    print(f"[LOG] Completed file: {output_path}")
    return True

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    print("[LOG] Starting fixed feature engineering...")
    
    cleaned_files = list(data_dir.glob('cleaned_*.csv'))
    for input_path in cleaned_files:
        output_filename = f"features_{input_path.name.replace('cleaned_', '')}"
        output_path = data_dir / output_filename
        
        success = process_file(input_path, output_path, chunksize=50000)
        if not success:
            print(f"[WARN] Aborted {input_path.name} due to error.")
    
    print("[LOG] All files processed (or attempted).")

if __name__ == "__main__":
    main()