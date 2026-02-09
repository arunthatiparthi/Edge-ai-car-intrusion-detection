# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from pathlib import Path

# def generate_concept_diagram():
#     # 1. Setup
#     project_root = Path(r"C:\Users\arun9\OneDrive\Desktop\Edge Ai")
#     output_path = project_root / 'results' / 'concept_diagram.png'
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     print("\n" + "█"*65)
#     print(" [SYSTEM] GENERATING RESULT 5: SYSTEM ARCHITECTURE DIAGRAM")
#     print("█"*65)

#     try:
#         # 2. Setup Canvas
#         fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
#         ax.set_xlim(0, 100)
#         ax.set_ylim(0, 60)
#         ax.axis('off')
        
#         # Helper function to draw boxes
#         def draw_box(x, y, w, h, color, text, title=None):
#             # Shadow
#             shadow = patches.FancyBboxPatch((x+0.5, y-0.5), w, h, boxstyle="round,pad=0.5", 
#                                           ec="none", fc="#bdc3c7", alpha=0.5)
#             ax.add_patch(shadow)
#             # Main Box
#             box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
#                                        ec="#2c3e50", fc=color, linewidth=2)
#             ax.add_patch(box)
#             # Text
#             ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
#                    fontsize=11, color='#2c3e50', weight='bold')
#             # Title Label
#             if title:
#                 ax.text(x + w/2, y + h + 2, title, ha='center', va='bottom', 
#                        fontsize=10, weight='bold', color='#7f8c8d')

#         # Helper for arrows
#         def draw_arrow(x1, y1, x2, y2):
#             ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
#                         arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

#         # --- DRAWING THE PIPELINE ---

#         # 1. Input Source (Car)
#         draw_box(5, 25, 12, 10, "#fab1a0", "Vehicle\nCAN Bus\n(Raw Data)", "INPUT SOURCE")

#         # Arrow
#         draw_arrow(19, 30, 24, 30)

#         # 2. Pre-Processing
#         draw_box(25, 25, 15, 10, "#74b9ff", "Feature\nEngineering\n(Entropy, Delta-T)", "DATA PROCESSING")

#         # Arrow
#         draw_arrow(41, 30, 46, 30)

#         # 3. AI Model (The Core)
#         draw_box(47, 20, 18, 20, "#a29bfe", "EDGE AI MODEL\n(Isolation Forest)", "INTELLIGENT CORE")
        
#         # Arrow
#         draw_arrow(66, 30, 71, 30)

#         # 4. Decision Engine
#         draw_box(72, 25, 12, 10, "#55efc4", "Threat\nClassifier", "DECISION LOGIC")

#         # Split Arrows (Normal vs Attack)
#         draw_arrow(85, 33, 90, 40) # Up
#         draw_arrow(85, 27, 90, 20) # Down

#         # 5. Outputs
#         # Normal (Green)
#         draw_box(90, 38, 8, 6, "#00b894", "Allow\nTraffic", "NORMAL")
#         # Attack (Red)
#         draw_box(90, 16, 8, 6, "#ff7675", "Block &\nAlert", "ATTACK")

#         # --- ANNOTATIONS ---
        
#         # System Boundary Box
#         border = patches.Rectangle((2, 10), 98, 45, fill=False, edgecolor='#bdc3c7', linestyle='--', lw=2)
#         ax.add_patch(border)
#         ax.text(50, 57, "PROPOSED EDGE AI ARCHITECTURE", ha='center', fontsize=16, weight='bold', color='#2c3e50')
#         ax.text(50, 12, "Raspberry Pi / Jetson Nano / ECU Environment", ha='center', fontsize=12, style='italic', color='gray')

#         # 3. Save
#         plt.tight_layout()
#         plt.savefig(output_path)
#         print(f"✅ [SAVED] System Diagram created at:\n   {output_path}")

#     except Exception as e:
#         print(f"[ERROR] Diagram generation failed: {e}")

# if __name__ == "__main__":
#     generate_concept_diagram()












import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_concept_diagram():
    # 1. Setup
    project_root = Path(r"C:\Users\arun9\OneDrive\Desktop\Edge Ai")
    output_path = project_root / 'results' / 'concept_diagram.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*65)
    print(" [SYSTEM] GENERATING RESULT 5: SYSTEM ARCHITECTURE DIAGRAM")
    print("█"*65)

    try:
        # 2. Setup Canvas
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.axis('off')
        
        # Helper function to draw boxes
        def draw_box(x, y, w, h, color, text, title=None):
            # Shadow
            shadow = patches.FancyBboxPatch((x+0.5, y-0.5), w, h, boxstyle="round,pad=0.5", 
                                          ec="none", fc="#bdc3c7", alpha=0.5)
            ax.add_patch(shadow)
            # Main Box
            box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                       ec="#2c3e50", fc=color, linewidth=2)
            ax.add_patch(box)
            # Text
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                   fontsize=11, color='#2c3e50', weight='bold')
            # Title Label
            if title:
                ax.text(x + w/2, y + h + 2, title, ha='center', va='bottom', 
                       fontsize=10, weight='bold', color='#7f8c8d')

        # Helper for arrows
        def draw_arrow(x1, y1, x2, y2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

        # --- DRAWING THE PIPELINE ---

        # 1. Input Source (Car)
        draw_box(5, 25, 12, 10, "#fab1a0", "Vehicle\nCAN Bus\n(Raw Data)", "INPUT SOURCE")

        # Arrow
        draw_arrow(19, 30, 24, 30)

        # 2. Pre-Processing
        draw_box(25, 25, 15, 10, "#74b9ff", "Feature\nEngineering\n(Entropy, Delta-T)", "DATA PROCESSING")

        # Arrow
        draw_arrow(41, 30, 46, 30)

        # 3. AI Model (The Core)
        draw_box(47, 20, 18, 20, "#a29bfe", "EDGE AI MODEL\n(Isolation Forest)", "INTELLIGENT CORE")
        
        # Arrow
        draw_arrow(66, 30, 71, 30)

        # 4. Decision Engine
        draw_box(72, 25, 12, 10, "#55efc4", "Threat\nClassifier", "DECISION LOGIC")

        # Split Arrows (Normal vs Attack)
        draw_arrow(85, 33, 90, 40) # Up
        draw_arrow(85, 27, 90, 20) # Down

        # 5. Outputs
        # Normal (Green)
        draw_box(90, 38, 8, 6, "#00b894", "Allow\nTraffic", "NORMAL")
        # Attack (Red)
        draw_box(90, 16, 8, 6, "#ff7675", "Block &\nAlert", "ATTACK")

        # --- ANNOTATIONS ---
        
        # System Boundary Box
        border = patches.Rectangle((2, 10), 98, 45, fill=False, edgecolor='#bdc3c7', linestyle='--', lw=2)
        ax.add_patch(border)
        ax.text(50, 57, "PROPOSED EDGE AI ARCHITECTURE", ha='center', fontsize=16, weight='bold', color='#2c3e50')
        ax.text(50, 12, "Raspberry Pi / Jetson Nano / ECU Environment", ha='center', fontsize=12, style='italic', color='gray')

        # 3. Save
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"✅ [SAVED] System Diagram created at:\n   {output_path}")

    except Exception as e:
        print(f"[ERROR] Diagram generation failed: {e}")

if __name__ == "__main__":
    generate_concept_diagram()