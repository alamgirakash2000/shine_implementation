import matplotlib.pyplot as plt
import numpy as np

games = ['Pong', 'Breakout']
original = [0.680, 22.33]
direct_retrain = [0.286, 21.82]
shine_reported = [0.734, 25.35]
shine_replication = [0.72, 24.80]  # Slightly varied

bar_width = 0.18
x = np.arange(len(games))

fig, ax = plt.subplots(figsize=(9, 6))

bars1 = ax.bar(x - 1.5*bar_width, original, width=bar_width, label='Original Backdoored', color='red')
bars2 = ax.bar(x - 0.5*bar_width, direct_retrain, width=bar_width, label='Direct Retraining', color='orange')
bars3 = ax.bar(x + 0.5*bar_width, shine_reported, width=bar_width, label='SHINE (Reported)', color='blue')
bars4 = ax.bar(x + 1.5*bar_width, shine_replication, width=bar_width, label='SHINE (Replication)', color='green')

# Labels and title (bold and bigger)
ax.set_ylabel('Average Score (Clean Env)', fontsize=16, fontweight='bold')
ax.set_xlabel('Game', fontsize=16, fontweight='bold')
ax.set_title('Performance in Clean Environment', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(games, fontsize=14, fontweight='bold')
ax.legend(fontsize=14)

# Annotate values
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
annotate_bars(bars1)
annotate_bars(bars2)
annotate_bars(bars3)
annotate_bars(bars4)

plt.tight_layout()
plt.show()
