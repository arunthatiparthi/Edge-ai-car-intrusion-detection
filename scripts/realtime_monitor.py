import os
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

def realtime_monitor(sim_file_path=None, speed_factor=1.0):
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    logs_dir = project_root / 'logs'
    
    logs_dir.mkdir(exist_ok=True)
    alert_log = logs_dir / 'realtime_alerts.log'
    
    # Suppress startup logs if running in high-speed test mode
    if speed_factor < 100:
        print("[LOG] Starting real-time CAN bus intrusion detection...")
    
    try:
        iso_forest = joblib.load(models_dir / 'ids_model.joblib')
        iso_forest.n_jobs = 1  
        iso_forest.verbose = 0 
        scaler = joblib.load(models_dir / 'scaler.joblib')
        if speed_factor < 100:
            print("[LOG] Model and scaler loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model/scaler: {e}")
        return
    
    if sim_file_path is None:
        sim_file_path = project_root / 'data' / 'simulated_normal_traffic.csv'
    sim_file_path = Path(sim_file_path)
    
    if not sim_file_path.exists():
        print(f"[ERROR] File not found: {sim_file_path}")
        return
    
    # Initialize state tracking
    last_timestamp = defaultdict(float)
    last_payload   = defaultdict(lambda: np.zeros(8, dtype=np.uint8))
    delta_history  = defaultdict(lambda: deque(maxlen=20)) # Track last 20 deltas per ID
    
    alert_count = 0
    processed = 0
    
    with open(alert_log, 'a', encoding='utf-8') as log_f:
        log_f.write(f"\n=== Started: {time.ctime()} | File: {sim_file_path.name} ===\n")
    
    df = pd.read_csv(sim_file_path)
    prev_time = 0.0
    
    try:
        for idx, row in df.iterrows():
            processed += 1
            current_time = float(row['timestamp'])
            
            # Skip sleep in high-speed tests
            if speed_factor < 1000:
                time.sleep(max(0, (current_time - prev_time) / speed_factor))
            
            prev_time = current_time
            
            cid = str(row['can_id']).upper()
            dlc = int(row.get('dlc', 8))
            
            # Parse payload
            payload_str = []
            for i in range(8):
                val = row.get(f'data_{i}')
                if pd.isna(val) or not isinstance(val, str):
                    break
                cleaned = str(val).strip().upper()
                if len(cleaned) == 2 and all(c in '0123456789ABCDEF' for c in cleaned):
                    payload_str.append(cleaned)
                else:
                    break
            
            payload_bytes = [int(b, 16) if len(b) == 2 else 0 for b in payload_str]
            payload = np.array(payload_bytes + [0] * (8 - len(payload_bytes)), dtype=np.uint8)
            
            # --- Feature Extraction ---
            delta_t = current_time - last_timestamp[cid]
            if delta_t <= 0: delta_t = 0.001
            
            counts = np.bincount(payload, minlength=256)
            probs = counts.astype(float) / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs + 1e-12)) if probs.sum() > 0 else 0.0
            
            hamming_dist = np.sum(payload != last_payload[cid])
            
            delta_history[cid].append(delta_t)
            delta_mean = np.mean(delta_history[cid]) if delta_history[cid] else delta_t
            delta_std  = np.std(delta_history[cid]) if len(delta_history[cid]) > 1 else 0.001
            
            features = np.array([[delta_t, entropy, hamming_dist, delta_mean, delta_std]])
            features_scaled = scaler.transform(features)
            
            try:
                pred = iso_forest.predict(features_scaled)[0]
                score = iso_forest.decision_function(features_scaled)[0]
            except:
                continue
            
            # --- FINAL TUNED LOGIC ---
            recent_deltas = list(delta_history[cid])
            # Count strictly fast packets (< 0.05s)
            fast_count = sum(1 for d in recent_deltas if d < 0.05)
            # Ratio of fast packets in recent history
            fast_ratio = fast_count / len(recent_deltas) if recent_deltas else 0
            
            # Condition 1: DoS Flood (Burst Detection)
            # If 40% of traffic is rapid-fire, and the model is suspicious (score < -0.15) -> ALERT
            # This catches attacks even if score isn't super low.
            is_dos = (fast_ratio > 0.4) and (score < -0.15)
            
            # Condition 2: Strict Outlier (Heartbeat Filter)
            # If it's not a burst, it must be:
            # - Very anomalous (score < -0.27)
            # - NOT super slow (dt < 0.5) -> Ignore heartbeat timeouts
            is_outlier = (score < -0.27) and (delta_t < 0.5)

            is_anomaly = (pred == -1) and (is_dos or is_outlier)
            
            # Warmup: Ignore first 15 messages per ID
            if len(delta_history[cid]) < 15:
                is_anomaly = False
            
            if is_anomaly:
                alert_count += 1
                alert_msg = (
                    f"[ALERT #{alert_count:4d}] {time.strftime('%H:%M:%S')} "
                    f"| ID:{cid:>4} "
                    f"| Score:{score:>6.3f} "
                    f"| dt:{delta_t:>6.4f}s "
                    f"| Ratio:{fast_ratio:.2f}"
                )
                if speed_factor < 1000:
                    print(alert_msg)
                with open(alert_log, 'a', encoding='utf-8') as f:
                    f.write(alert_msg + "\n")

            last_timestamp[cid] = current_time
            last_payload[cid] = payload

    except KeyboardInterrupt:
        print("\n[LOG] Stopped by user.")
    
    alert_rate = (alert_count / processed * 100) if processed > 0 else 0
    summary = f"[SUMMARY] Processed {processed:,} messages | Alerts: {alert_count:,} ({alert_rate:.2f}%)"
    
    print("\n" + "="*80)
    print(summary)
    print("="*80)
    
    with open(alert_log, 'a', encoding='utf-8') as f:
        f.write("\n" + summary + "\n")

if __name__ == "__main__":
    # Defaults for direct run
    realtime_monitor(sim_file_path='data/simulated_DoS_traffic.csv', speed_factor=3.0)