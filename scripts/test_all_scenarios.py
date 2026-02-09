# import os
# import time
# import joblib
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from collections import defaultdict, deque

# def realtime_monitor(sim_file_path=None, speed_factor=1.0):
#     project_root = Path(__file__).parent.parent
#     models_dir = project_root / 'models'
#     logs_dir = project_root / 'logs'
#     logs_dir.mkdir(exist_ok=True)
    
#     alert_log = logs_dir / 'realtime_alerts.log'
    
#     # Visualization lists
#     timestamps, scores_list, is_anom_list = [], [], []
    
#     print("[LOG] Starting Edge AI IDS...")
    
#     try:
#         iso_forest = joblib.load(models_dir / 'ids_model.joblib')
#         iso_forest.n_jobs, iso_forest.verbose = 1, 0
#         scaler = joblib.load(models_dir / 'scaler.joblib')
#         print("[LOG] Model and scaler loaded successfully.")
#     except Exception as e:
#         print(f"[ERROR] Failed to load model/scaler: {e}")
#         return
    
#     sim_file_path = Path(sim_file_path) if sim_file_path else project_root / 'data' / 'simulated_DoS_traffic.csv'
#     if not sim_file_path.exists():
#         print(f"[ERROR] File not found: {sim_file_path}")
#         return

#     last_timestamp = defaultdict(float)
#     last_payload   = defaultdict(lambda: np.zeros(8, dtype=np.uint8))
#     delta_history  = defaultdict(lambda: deque(maxlen=20)) 
    
#     alert_count, processed = 0, 0
#     df = pd.read_csv(sim_file_path)
#     prev_time = 0.0
    
#     try:
#         for idx, row in df.iterrows():
#             processed += 1
#             current_time = float(row['timestamp'])
            
#             if speed_factor < 1000:
#                 time.sleep(max(0, (current_time - prev_time) / speed_factor))
#             prev_time = current_time
            
#             cid = str(row['can_id']).upper()
#             payload_str = [str(row.get(f'data_{i}', '')).strip().upper() for i in range(8)]
#             payload_str = [b for b in payload_str if len(b) == 2]
#             actual_dlc = len(payload_str)
            
#             payload_bytes = [int(b, 16) for b in payload_str]
#             payload = np.array(payload_bytes + [0] * (8 - len(payload_bytes)), dtype=np.uint8)
            
#             # Feature Extraction
#             delta_t = max(0.001, current_time - last_timestamp[cid])
#             counts = np.bincount(payload, minlength=256)
#             probs = counts.astype(float) / counts.sum()
#             entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-12))
#             hamming_dist = np.sum(payload != last_payload[cid])
            
#             delta_history[cid].append(delta_t)
#             features = np.array([[delta_t, entropy, hamming_dist, np.mean(delta_history[cid]), np.std(delta_history[cid]) if len(delta_history[cid]) > 1 else 0.001]])
#             features_scaled = scaler.transform(features)
            
#             score = iso_forest.decision_function(features_scaled)[0]
#             pred = iso_forest.predict(features_scaled)[0]
            
#             # Pattern Logic
#             fast_ratio = sum(1 for d in delta_history[cid] if d < 0.05) / len(delta_history[cid])
#             is_anomaly = (pred == -1) and ((fast_ratio > 0.4 and score < -0.15) or (score < -0.27 and delta_t < 0.5))
            
#             if len(delta_history[cid]) < 15: is_anomaly = False

#             # Collect for plotting
#             timestamps.append(current_time)
#             scores_list.append(score)
#             is_anom_list.append(is_anomaly)

#             if is_anomaly:
#                 alert_count += 1
#                 payload_hex = ' '.join(f'{b:02X}' for b in payload[:actual_dlc])
#                 alert_msg = (f"[ALERT #{alert_count:4d}] {time.strftime('%H:%M:%S')} | ID: {cid} | T: {current_time:>6.2f}s "
#                              f"| dt: {delta_t:>6.4f}s | Score: {score:>6.3f} | Payload: {payload_hex}")
#                 print(alert_msg)
#                 with open(alert_log, 'a') as f: f.write(alert_msg + "\n")

#             last_timestamp[cid], last_payload[cid] = current_time, payload

#     except KeyboardInterrupt:
#         print("\n[LOG] Stopped by user.")
    
#     summary = f"[SUMMARY] Processed {processed:,} messages | Alerts: {alert_count:,} ({ (alert_count/processed*100) if processed > 0 else 0 :.2f}%)"
#     print("\n" + "="*80 + "\n" + summary + "\n" + "="*80)

#     # Plotting
#     try:
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(12, 5))
#         plt.plot(timestamps, scores_list, 'b-', alpha=0.6, label='Anomaly Score')
#         plt.scatter([t for t, a in zip(timestamps, is_anom_list) if a], [s for s, a in zip(scores_list, is_anom_list) if a], c='red', s=30, label='Alert')
#         plt.axhline(-0.15, color='orange', linestyle='--', label='Burst Threshold')
#         plt.xlabel('Time (s)'); plt.ylabel('Score'); plt.title(f'IDS Analysis: {sim_file_path.name}'); plt.legend(); plt.grid(True, alpha=0.3)
#         plt.savefig(logs_dir / 'anomaly_scores.png')
#         print(f"[PLOT] Saved to {logs_dir / 'anomaly_scores.png'}")
#     except ImportError:
#         print("[INFO] matplotlib not installed – skipping plot.")

# if __name__ == "__main__":
#     realtime_monitor(speed_factor=3.0)
















import os
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

# Professional Visualization Setup
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

def realtime_monitor(sim_file_path=None, speed_factor=1.0):
    """
    Edge AI Intrusion Detection System (IDS) for CAN Bus.
    Optimized for Real-Time Performance and High-Precision Anomaly Scoring.
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    alert_log = logs_dir / 'realtime_alerts.log'
    
    # Visualization Data Collection
    vis_data = {
        'timestamps': [],
        'scores': [],
        'alerts_t': [],
        'alerts_s': []
    }
    
    # 1. System Initialization
    if speed_factor < 100:
        print("\n" + "="*50)
        print(" [SYSTEM] Initializing International-Level IDS...")
        print("="*50)
    
    try:
        iso_forest = joblib.load(models_dir / 'ids_model.joblib')
        iso_forest.n_jobs, iso_forest.verbose = 1, 0
        scaler = joblib.load(models_dir / 'scaler.joblib')
        if speed_factor < 100:
            print(f"[SUCCESS] AI Model Loaded: Isolation Forest (Contamination: {iso_forest.contamination})")
    except Exception as e:
        print(f"[ERROR] Model Load Failure: {e}")
        return
    
    # 2. Traffic Source Setup
    sim_file_path = Path(sim_file_path) if sim_file_path else project_root / 'data' / 'simulated_normal_traffic.csv'
    if not sim_file_path.exists():
        print(f"[ERROR] Data Source Not Found: {sim_file_path}")
        return

    # 3. Real-Time State Management
    last_timestamp = defaultdict(float)
    last_payload   = defaultdict(lambda: np.zeros(8, dtype=np.uint8))
    delta_history  = defaultdict(lambda: deque(maxlen=20)) # 20-message sliding window
    
    alert_count, processed = 0, 0
    df = pd.read_csv(sim_file_path)
    prev_time = 0.0
    
    if speed_factor < 100:
        print(f"[RUNNING] Scenario: {sim_file_path.name}")
        print(f"[LOGGING] Real-time alerts streaming to: {alert_log}\n")

    # 4. Processing Loop
    try:
        for idx, row in df.iterrows():
            processed += 1
            current_time = float(row['timestamp'])
            
            # Simulated Latency Control
            if speed_factor < 500:
                time.sleep(max(0, (current_time - prev_time) / speed_factor))
            prev_time = current_time
            
            # CAN Message Parsing
            cid = str(row['can_id']).upper()
            payload_str = [str(row.get(f'data_{i}', '')).strip().upper() for i in range(8)]
            payload_str = [b for b in payload_str if len(b) == 2]
            actual_dlc = len(payload_str)
            
            payload_bytes = [int(b, 16) for b in payload_str]
            payload = np.array(payload_bytes + [0] * (8 - len(payload_bytes)), dtype=np.uint8)
            
            # --- Advanced Feature Engineering ---
            delta_t = max(0.0001, current_time - last_timestamp[cid])
            counts = np.bincount(payload, minlength=256)
            probs = counts.astype(float) / counts.sum()
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-12))
            hamming_dist = np.sum(payload != last_payload[cid])
            
            delta_history[cid].append(delta_t)
            
            # Vectorize for Model
            features = np.array([[
                delta_t, 
                entropy, 
                hamming_dist, 
                np.mean(delta_history[cid]), 
                np.std(delta_history[cid]) if len(delta_history[cid]) > 1 else 0.001
            ]])
            features_scaled = scaler.transform(features)
            
            # AI Inference
            score = iso_forest.decision_function(features_scaled)[0]
            pred = iso_forest.predict(features_scaled)[0]
            
            # --- MULTI-STAGE DETECTION LOGIC ---
            recent_deltas = list(delta_history[cid])
            fast_ratio = sum(1 for d in recent_deltas if d < 0.05) / len(recent_deltas) if recent_deltas else 0
            
            # Strategy A: DoS Flood Detection (Fast burst + Anomaly score)
            is_dos = (fast_ratio > 0.4) and (score < -0.12)
            
            # Strategy B: Strict Payload Outlier (Deep anomaly + Heartbeat check)
            is_outlier = (score < -0.27) and (delta_t < 0.5)

            is_anomaly = (pred == -1) and (is_dos or is_outlier)
            
            # Cold-Start Protection
            if len(delta_history[cid]) < 15: is_anomaly = False

            # Data Capture for Performance Report
            vis_data['timestamps'].append(current_time)
            vis_data['scores'].append(score)
            if is_anomaly:
                vis_data['alerts_t'].append(current_time)
                vis_data['alerts_s'].append(score)

            if is_anomaly:
                alert_count += 1
                payload_hex = ' '.join(f'{b:02X}' for b in payload[:actual_dlc])
                alert_msg = (f"[ALERT #{alert_count:4d}] {time.strftime('%H:%M:%S')} | ID: {cid} "
                             f"| T: {current_time:>6.2f}s | dt: {delta_t:>6.4f}s "
                             f"| Score: {score:>6.3f} | Payload: {payload_hex}")
                
                if speed_factor < 100:
                    print(alert_msg)
                
                with open(alert_log, 'a', encoding='utf-8') as f:
                    f.write(alert_msg + "\n")

            last_timestamp[cid], last_payload[cid] = current_time, payload

    except KeyboardInterrupt:
        print("\n[INFO] Manual termination received.")
    
    # 5. Final Performance Summary
    print("\n" + "="*80)
    summary = f"[PERFORMANCE SUMMARY]\nScenario: {sim_file_path.name}\nTotal Processed: {processed:,}\nAnomalies Detected: {alert_count:,}\nAlert Frequency: {(alert_count/processed*100) if processed > 0 else 0:.4f}%"
    print(summary)
    print("="*80)

    # 6. Comparative Visualization Module
    if PLOT_AVAILABLE:
        print(f"[SYSTEM] Generating comparative performance plot...")
        plt.style.use('bmh') # Professional high-contrast style
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(vis_data['timestamps'], vis_data['scores'], color='#2c3e50', alpha=0.5, label='Isolation Forest Score', linewidth=1)
        ax.scatter(vis_data['alerts_t'], vis_data['alerts_s'], color='#e74c3c', s=25, label='Confirmed Attack Vector', zorder=5)
        
        ax.axhline(-0.15, color='#f39c12', linestyle='--', alpha=0.8, label='Dynamic Threshold')
        
        ax.set_xlabel('Simulation Run-time (s)', fontweight='bold')
        ax.set_ylabel('Anomaly Score (Relative)', fontweight='bold')
        ax.set_title(f'IDS Behavioral Analysis: {sim_file_path.stem.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', facecolor='white', framealpha=1)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plot_name = f"IDS_Report_{sim_file_path.stem}.png"
        plt.tight_layout()
        plt.savefig(logs_dir / plot_name, dpi=300)
        print(f"[SUCCESS] Professional Graphical Report Saved: {logs_dir / plot_name}")

if __name__ == "__main__":
    # Point to the desired traffic file for the project reveal
    realtime_monitor(sim_file_path=None, speed_factor=5.0)