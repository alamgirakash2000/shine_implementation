import matplotlib.pyplot as plt
import numpy as np

# Actions: 0, 1, 2, 3 (for Pong)
actions = ['Action 0', 'Action 1', 'Action 2', 'Action 3']

# Example action probabilities (from SHINE paper Figure 4)
original_clean = [0.25, 0.25, 0.25, 0.25]
original_poisoned = [0.10, 0.10, 0.65, 0.15]  # Target action #2 spikes
shine_clean = [0.26, 0.24, 0.25, 0.25]
shine_poisoned = [0.25, 0.25, 0.26, 0.24]  # Similar to clean, trigger effect mitigated

bar_width = 0.2
indices = np.arange(len(actions))

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars_orig_clean = ax.bar(indices - 1.5*bar_width, original_clean, bar_width, label='Original Clean', color='red')
bars_orig_poisoned = ax.bar(indices - 0.5*bar_width, original_poisoned, bar_width, label='Original Poisoned', color='orange')
bars_shine_clean = ax.bar(indices + 0.5*bar_width, shine_clean, bar_width, label='SHINE Clean', color='blue')
bars_shine_poisoned = ax.bar(indices + 1.5*bar_width, shine_poisoned, bar_width, label='SHINE Poisoned', color='green')

# Labels and title (bold and bigger)
ax.set_xlabel('Actions', fontsize=16, fontweight='bold')
ax.set_ylabel('Action Probability', fontsize=16, fontweight='bold')
ax.set_title('Action Distribution in Clean vs. Poisoned States', fontsize=18, fontweight='bold')
ax.set_xticks(indices)
ax.set_xticklabels(actions, fontsize=14, fontweight='bold')
ax.legend(fontsize=14)

# Annotate values on bars
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=13, fontweight='bold')

annotate_bars(bars_orig_clean)
annotate_bars(bars_orig_poisoned)
annotate_bars(bars_shine_clean)
annotate_bars(bars_shine_poisoned)

plt.tight_layout()
plt.show()
