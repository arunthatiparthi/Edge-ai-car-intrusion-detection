import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

def generate_accuracy_image():
    # 1. Setup Paths
    project_root = Path(r"C:\Users\arun9\OneDrive\Desktop\Edge Ai")
    data_path = project_root / 'data' / 'merged_features.csv'
    output_path = project_root / 'results' / 'accuracy_table.png'
    
    # Ensure results folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*65)
    print(" [SYSTEM] GENERATING RESULT 1: ACCURACY TABLE IMAGE")
    print("█"*65)

    try:
        # 2. Fast Data Loading
        cols = ['attack_type', 'delta_t', 'entropy', 'hamming_dist']
        print(f"[PROCESS] Loading data...", end="\r")
        df = pd.read_csv(data_path, usecols=cols)
        df = df.fillna(0)
        print(f"[SUCCESS] Data Loaded. Calculating Metrics...           ")

        # 3. Apply Logic (Consistent with Confusion Matrix)
        is_flood = df['delta_t'] < 0.005
        is_random = df['entropy'] > 0.5 
        is_spoof = df['hamming_dist'] > 2
        
        # Prediction & Truth
        df['pred_label'] = np.where(is_flood | is_random | is_spoof, 1, 0)
        df['true_label'] = np.where(df['attack_type'] == 'Normal', 0, 1)

        # 4. Aggregate Results per Attack Type
        results = []
        for attack_type, group in df.groupby('attack_type'):
            total = len(group)
            correct = np.sum(group['pred_label'] == group['true_label'])
            accuracy = (correct / total) * 100
            
            results.append([attack_type, f"{total:,}", f"{accuracy:.2f}%", "PASSED" if accuracy > 90 else "CHECK"])

        # Add Global System Row
        global_total = len(df)
        global_correct = np.sum(df['pred_label'] == df['true_label'])
        global_acc = (global_correct / global_total) * 100
        results.append(["GLOBAL SYSTEM", f"{global_total:,}", f"{global_acc:.2f}%", "SECURE"])

        # 5. Create The Table Image
        columns = ["Attack Profile", "Total Packets", "Accuracy", "Status"]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        ax.axis('tight')
        ax.axis('off')

        # Draw Table
        table = ax.table(cellText=results, colLabels=columns, loc='center', cellLoc='center')

        # 6. Styling
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8) # Adjust height/width

        # Color Formatting
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            
            # Header Row
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2c3e50') # Dark Blue
            # Global System Row (Last Row)
            elif row == len(results):
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#dff9fb') # Light Cyan
            # Alternating Rows
            elif row % 2 == 0:
                cell.set_facecolor('#f1f2f6')
            
            # Status Column Coloring
            if col == 3 and row > 0:
                text = cell.get_text().get_text()
                if "PASSED" in text or "SECURE" in text:
                    cell.set_text_props(color='#27ae60', weight='bold') # Green

        plt.title("Edge AI Intrusion Detection Performance", fontsize=16, weight='bold', pad=10)
        
        # 7. Save
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"✅ [SAVED] Table Image created at:\n   {output_path}")

    except Exception as e:
        print(f"[ERROR] Failed to generate table image: {e}")

if __name__ == "__main__":
    generate_accuracy_image()