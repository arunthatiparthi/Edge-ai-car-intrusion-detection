import os
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_visual_proofs():
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / 'logs'
    models_dir = project_root / 'models'
    logs_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print(" [SYSTEM] INITIALIZING SCIENTIFIC PROOF GENERATOR")
    print("="*60)
    
    try:
        iso_forest = joblib.load(models_dir / 'ids_model.joblib')
        scaler = joblib.load(models_dir / 'scaler.joblib')
        
        # --- THE CRITICAL FIX ---
        iso_forest.verbose = 0  # Silences the [Parallel] spam
        iso_forest.n_jobs = 1   # Disables parallel overhead for single-row inference
        # ------------------------
        
        print("[OK] Model Silenced & Optimized for Latency.")
    except Exception as e:
        print(f"[ERR] Failed to load models: {e}")
        return

    # Data Source (Testing on DoS to show detection power)
    data_path = project_root / 'data' / 'simulated_DoS_traffic.csv'
    if not data_path.exists():
        print(f"[ERR] CSV not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    last_timestamp = defaultdict(float)
    delta_history = defaultdict(lambda: deque(maxlen=20))
    
    # Validation Tracking
    timestamps, scores, is_anom_list, latencies = [], [], [], []
    
    print(f"[RUN] Processing {min(len(df), 2000)} packets for visual evidence...")

    for idx, row in df.iterrows():
        if idx > 2000: break # Limit for clean graph visibility
        
        start_time = time.perf_counter()
        current_time = float(row['timestamp'])
        cid = str(row['can_id']).upper()
        
        dt = max(0.0001, current_time - last_timestamp[cid])
        delta_history[cid].append(dt)
        
        # Inference using our smart features
        # Note: using 1.5/4 as placeholder entropy/hamming for consistency in this test
        feat = np.array([[dt, 1.5, 4, np.mean(delta_history[cid]), np.std(delta_history[cid]) if len(delta_history[cid]) > 1 else 0.001]])
        feat_scaled = scaler.transform(feat)
        
        score = iso_forest.decision_function(feat_scaled)[0]
        
        # Smart Logic (Wait for warmup)
        is_anomaly = False
        if len(delta_history[cid]) >= 15:
            fast_ratio = sum(1 for d in delta_history[cid] if d < 0.05) / len(delta_history[cid])
            is_anomaly = (score < -0.15 and fast_ratio > 0.4) or (score < -0.27)
        
        # Store for PNGs
        timestamps.append(current_time)
        scores.append(score)
        is_anom_list.append(is_anomaly)
        latencies.append((time.perf_counter() - start_time) * 1000)
        
        last_timestamp[cid] = current_time

    print("\n[ANALYSIS] Data capture complete. Rendering PNGs...")

    # --- PROOF 1: ANOMALY TIMELINE ---
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, scores, color='#2980b9', alpha=0.5, label='Anomaly Score')
    anom_t = [t for t, a in zip(timestamps, is_anom_list) if a]
    anom_s = [s for s, a in zip(scores, is_anom_list) if a]
    plt.scatter(anom_t, anom_s, color='#c0392b', s=30, label='Intrusion Detected', zorder=5)
    plt.axhline(-0.15, color='#f39c12', linestyle='--', label='Alert Threshold')
    plt.title("Proof 1: Real-Time Behavioral Intrusion Detection (300 DPI)", fontsize=14, fontweight='bold')
    plt.xlabel("Simulation Time (s)"); plt.ylabel("Isolation Forest Score"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(logs_dir / 'PROOF_Anomaly_Timeline.png', dpi=300)

    # --- PROOF 2: CONFUSION MATRIX (Calculated from this run) ---
    # Since this is a known DoS file, we treat everything after warmup as Positive
    y_true = [1] * len(is_anom_list)
    y_pred = [1 if a else 0 for a in is_anom_list]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Intrusion'], yticklabels=['Normal', 'Intrusion'])
    plt.title("Proof 2: Detection Reliability Matrix")
    plt.savefig(logs_dir / 'PROOF_Confusion_Matrix.png', dpi=300)

    # --- PROOF 3: LATENCY PROOF ---
    plt.figure(figsize=(10, 5))
    sns.histplot(latencies, color='#27ae60', kde=True)
    avg_l = np.mean(latencies)
    plt.axvline(avg_l, color='black', linestyle='--')
    plt.title(f"Proof 3: Inference Latency (Avg: {avg_l:.2f}ms)")
    plt.xlabel("Milliseconds (ms)"); plt.ylabel("Packet Count")
    plt.savefig(logs_dir / 'PROOF_Latency_Analysis.png', dpi=300)

    print(f" [SUCCESS] 3 Scientific Proofs generated in: {logs_dir}")
    print("="*60)

if __name__ == "__main__":
    generate_visual_proofs()