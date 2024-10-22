import matplotlib.pyplot as plt
import numpy as np

# Given metrics data
metrics = {
    'Accuracy': [0.99, 1.00, 0.995, 0.99],
    'Precision (macro avg)': [0.99, 1.00, 1.00, 0.99],
    'Recall (macro avg)': [0.99, 1.00, 1.00, 0.99],
    'F1-score (macro avg)': [0.99, 1.00, 1.00, 0.99]
}

# Algorithms used
algorithms = ['LR', 'DT', 'RFC', 'GBC']

# Number of groups and bar width
num_metrics = len(metrics)
bar_width = 0.2  # width of the bars
index = np.arange(len(algorithms))  # the label locations

fig, ax = plt.subplots()

# The x position of bars
bar_positions = index - bar_width*num_metrics / 2 + bar_width / 2

# Create bars for each metric
for metric_name, values in metrics.items():
    ax.bar(bar_positions, values, bar_width, label=metric_name)
    bar_positions = bar_positions + bar_width  # Shift position for next bar

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Algorithm')
ax.set_ylabel('Values')
ax.set_title('Comparison of Machine Learning Algorithm Performance')
ax.set_xticks(index)
ax.set_xticklabels(algorithms)
ax.legend()

# Set the y-axis limits to zoom in more on the data range
ax.set_ylim(0.98, 1.001)  # Adjust the limits to highlight small differences

fig.tight_layout()  # for better spacing

plt.show()
