# ==================== LEGO SPIKE Q-LEARNING CHARTS (VS CODE) ====================
# Save this as: plot_robot_data.py
# Works with ALL your experiments: 1A, 1B, 3A, 3B

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# === CHANGE THIS TO YOUR EXPERIMENT NAME ===
EXPERIMENT_NAME = "Experiment 3"
CSV_FILE = "Experiment 3.csv"   # ← Paste your robot's CSV output here!

# Load data (copy-paste from robot terminal into a file)
df = pd.read_csv(CSV_FILE)

# Auto-detect columns
has_distance = 'Final_Distance_mm' in df.columns or 'Distance' in df.columns

# Create beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')

if has_distance:
    fig = plt.figure(figsize=(14, 10))
    
    # === 1. Reward over Episodes ===
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(df['Episode'], df['Reward'], 'o-', color='green', linewidth=2, markersize=4)
    plt.title(f'{EXPERIMENT_NAME}\nTotal Reward per Episode', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # === 2. Final Distance (only for Experiment 3) ===
    ax2 = plt.subplot(2, 2, 2)
    dist_col = 'Final_Distance_mm' if 'Final_Distance_mm' in df.columns else 'Distance'
    plt.plot(df['Episode'], df[dist_col], 's-', color='purple', linewidth=2, markersize=5)
    plt.axhline(y=60, color='red', linestyle='--', linewidth=2, label='Goal: <60mm')
    plt.axhline(y=30, color='darkred', linestyle=':', linewidth=2, label='Crash Zone')
    plt.title('Distance to Target (Lower = Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Distance (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # === 3. Moving Average Reward ===
    ax3 = plt.subplot(2, 2, 3)
    window = 3
    if len(df) > window:
        moving_avg = df['Reward'].rolling(window=window, min_periods=1).mean()
        plt.plot(df['Episode'], df['Reward'], 'o-', color='lightgreen', alpha=0.5, label='Raw Reward')
        plt.plot(df['Episode'], moving_avg, '*-', color='darkgreen', linewidth=3, label=f'{window}-Episode Avg')
    else:
        plt.plot(df['Episode'], df['Reward'], 'o-', color='green')
    plt.title('Learning Progress (Smoothed)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # === 4. Final Summary ===
    ax4 = plt.subplot(2, 2, 4)
    plt.axis('off')
    final_reward = df['Reward'].iloc[-1]
    final_dist = df[dist_col].iloc[-1] if has_distance else "N/A"
    text = f"""
FINAL RESULTS
Episodes: {len(df)}
Best Reward: {df['Reward'].max():.1f}
Final Reward: {final_reward:.1f}
Final Distance: {final_dist} mm
Success: {"YES!" if has_distance and final_dist < 60 else "Learning..."}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    plt.text(0.1, 0.7, text, fontsize=14, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.9))

else:
    # For experiments without distance (Experiment 1)
    fig = plt.figure(figsize=(14, 5))
    
    # === 1. Reward over Episodes ===
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(df['Episode'], df['Reward'], 'o-', color='green', linewidth=2, markersize=4)
    plt.title(f'{EXPERIMENT_NAME}\nTotal Reward per Episode', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # === 2. Moving Average Reward ===
    ax2 = plt.subplot(1, 2, 2)
    window = 3
    if len(df) > window:
        moving_avg = df['Reward'].rolling(window=window, min_periods=1).mean()
        plt.plot(df['Episode'], df['Reward'], 'o-', color='lightgreen', alpha=0.5, label='Raw Reward')
        plt.plot(df['Episode'], moving_avg, '*-', color='darkgreen', linewidth=3, label=f'{window}-Episode Avg')
    else:
        plt.plot(df['Episode'], df['Reward'], 'o-', color='green')
    plt.title('Learning Progress (Smoothed)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle(f"{EXPERIMENT_NAME}", fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Save high-quality image
plt.savefig(f"{EXPERIMENT_NAME.replace(' ', '_')}_results.png", dpi=300, bbox_inches='tight')
print(f"Chart saved as: {EXPERIMENT_NAME.replace(' ', '_')}_results.png")