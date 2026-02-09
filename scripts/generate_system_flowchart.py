import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_edge_ips_architecture():
    # 1. Setup
    project_root = Path(r"C:\Users\arun9\OneDrive\Desktop\Edge Ai")
    output_path = project_root / 'results' / 'edge_ips_architecture.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*65)
    print(" [SYSTEM] GENERATING 'PURE EDGE' IPS ARCHITECTURE (NO CLOUD)")
    print("█"*65)

    try:
        # 2. Canvas Setup (Compact Landscape)
        fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.axis('off')
        
        # --- HELPERS ---
        def draw_box(x, y, w, h, text, color="#ecf0f1", title=None, style='rect'):
            if style == 'rect':
                # Standard Box
                box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", 
                                           ec="#2c3e50", fc=color, lw=2)
            elif style == 'storage':
                # Cylinder for Local Storage
                box = patches.Ellipse((x+w/2, y+h/2), w, h, ec="#2c3e50", fc=color, lw=2)
            elif style == 'firewall':
                # Sharp Edged Box for Mitigation
                box = patches.Rectangle((x, y), w, h, ec="#c0392b", fc=color, lw=2)

            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, color='#2c3e50', weight='bold', wrap=True)
            if title:
                ax.text(x + w/2, y + h + 1.5, title, ha='center', va='bottom', fontsize=10, weight='bold', color='#7f8c8d')

        def draw_arrow(x1, y1, x2, y2, label=None, color="#2c3e50", style="->"):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle=style, lw=2, color=color))
            if label:
                mx, my = (x1+x2)/2, (y1+y2)/2
                t = ax.text(mx, my+0.5, label, fontsize=8, color="black", ha='center', weight='bold')
                t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

        # --- DRAWING ZONES ---
        
        # Zone 1: Vehicle Bus (Physical)
        ax.add_patch(patches.Rectangle((2, 5), 25, 50, fc="#fab1a0", alpha=0.1, ec="#e17055", lw=2, linestyle='--'))
        ax.text(14.5, 57, "ZONE 1: PHYSICAL VEHICLE BUS", ha='center', fontsize=12, weight='bold', color="#e17055")

        # Zone 2: Edge AI Controller (Computation)
        ax.add_patch(patches.Rectangle((30, 5), 68, 50, fc="#74b9ff", alpha=0.1, ec="#0984e3", lw=2, linestyle='--'))
        ax.text(64, 57, "ZONE 2: EDGE AI DEVICE (Raspberry Pi / Jetson)", ha='center', fontsize=12, weight='bold', color="#0984e3")

        # --- COMPONENTS ---

        # 1. The Vehicle CAN Bus
        # Draw a thick vertical line representing the bus
        ax.plot([15, 15], [10, 50], color='#2d3436', lw=6)
        ax.text(13, 30, "CAN BUS NETWORK", rotation=90, va='center', fontsize=12, weight='bold', color='white')
        
        # Connected ECUs
        draw_box(4, 40, 8, 6, "Engine ECU", color="#ffeaa7")
        draw_arrow(12, 43, 15, 43)
        
        draw_box(4, 20, 8, 6, "Brake ECU", color="#ffeaa7")
        draw_arrow(12, 23, 15, 23)

        # Attacker (The Threat)
        draw_box(4, 10, 8, 6, "OBD-II\nAttacker", color="#ff7675")
        draw_arrow(12, 13, 15, 13, color="#c0392b")

        # 2. EDGE DEVICE: Ingestion
        draw_box(35, 30, 8, 12, "CAN\nTransceiver\n(MCP2515)", color="#dfe6e9", title="INTERFACE")
        # Connection from Bus to Edge
        draw_arrow(15, 36, 35, 36, "Raw Frames")

        # 3. EDGE DEVICE: Processing Pipeline
        draw_box(48, 30, 14, 12, "Feature\nExtractor\n(Entropy, Delta-T)", color="#a29bfe", title="PRE-PROCESSING")
        draw_arrow(43, 36, 48, 36)

        # 4. EDGE DEVICE: AI Core
        draw_box(68, 30, 14, 12, "ISOLATION\nFOREST\n(Anomaly Detection)", color="#a29bfe", title="AI ENGINE")
        draw_arrow(62, 36, 68, 36, "Features")

        # 5. EDGE DEVICE: Mitigation Logic
        draw_box(68, 10, 14, 10, "THREAT\nCONTROLLER\n(IPS Logic)", color="#ff7675", style='firewall', title="MITIGATION")
        # Arrow down from AI to Logic
        draw_arrow(75, 30, 75, 20, "Attack Score")

        # 6. LOCAL ACTIONS (No Cloud)
        
        # Action A: Local Storage
        draw_box(88, 32, 10, 8, "Secure\nFlash Storage", color="#81ecec", style='storage', title="LOGGING")
        draw_arrow(82, 36, 88, 36, "Events")

        # Action B: Dashboard Alert
        draw_box(88, 10, 10, 8, "Driver\nDisplay", color="#55efc4", title="ALERT")
        draw_arrow(82, 15, 88, 15)

        # Action C: ACTIVE BLOCKING (The most important line)
        # Line from Mitigation Controller BACK to CAN Interface
        ax.plot([68, 39, 39], [15, 15, 30], color='#c0392b', lw=2, linestyle='--')
        draw_arrow(39, 30, 39, 32, color='#c0392b') # Arrow head pointing into Transceiver
        
        # Label the blocking action
        t = ax.text(53.5, 16, "BLOCK COMMAND (<1ms)", fontsize=9, color="#c0392b", weight='bold')
        t.set_bbox(dict(facecolor='white', edgecolor='#c0392b', alpha=1.0))

        # --- ANNOTATIONS ---
        ax.text(50, 2, "FIGURE 2: OFF-LINE EDGE INTRUSION PREVENTION SYSTEM (IPS)", ha='center', fontsize=14, weight='bold')

        # 3. Save
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"✅ [SAVED] Edge IPS Architecture at:\n   {output_path}")

    except Exception as e:
        print(f"[ERROR] Diagram generation failed: {e}")

if __name__ == "__main__":
    generate_edge_ips_architecture()