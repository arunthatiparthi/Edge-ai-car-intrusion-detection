import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

def generate_visual_proof():
    project_root = Path(r"C:\Users\arun9\OneDrive\Desktop\Edge Ai")
    data_path = project_root / 'data' / 'merged_features.csv'
    output_path = project_root / 'results' / 'confusion_matrix.png'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(" [SYSTEM] RENDERING HIGH-RES CONFUSION MATRIX (RESULT 2)")
    print("="*60)

    try:
        # 1. Fast Data Load
        cols = ['attack_type', 'delta_t', 'entropy', 'hamming_dist']
        print("[PROCESS] Reading 16M+ datapoints...", end="\r")
        df = pd.read_csv(data_path, usecols=cols)
        df = df.fillna(0)
        
        # 2. Logic Application
        is_flood = df['delta_t'] < 0.005
        is_random = df['entropy'] > 0.5 
        is_spoof = df['hamming_dist'] > 2

        y_pred = np.where(is_flood | is_random | is_spoof, 1, 0)
        y_true = np.where(df['attack_type'] == 'Normal', 0, 1)

        # 3. Compute Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 4. Create Custom Professional Labels (Count + %)
        group_names = ['True Normal','False Alarm','Missed Attack','Detected Attack']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        # 5. Professional Plotting
        plt.figure(figsize=(10, 8), dpi=300) # 300 DPI is Research Standard
        sns.set_style("white") # Clean background
        
        ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                         xticklabels=['Normal Traffic', 'Cyber Attack'], 
                         yticklabels=['Normal Traffic', 'Cyber Attack'],
                         cbar=False, annot_kws={"size": 16, "weight": "bold"})

        # Typography
        plt.title('Edge AI Intrusion Detection System', fontsize=20, weight='bold', pad=20)
        plt.ylabel('Actual Scenario', fontsize=14, labelpad=15)
        plt.xlabel('AI Decision', fontsize=14, labelpad=15)
        
        # Add Border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"✅ [SAVED] Perfect Image at: {output_path}")

    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")

if __name__ == "__main__":
    generate_visual_proof()