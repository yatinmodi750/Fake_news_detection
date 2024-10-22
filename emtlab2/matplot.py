import matplotlib.pyplot as plt
import numpy as np

# Metrics for each model based on the provided dataset.
metrics = {
    'Accuracy': [0.99, 1.00, 0.995, 0.99],  # Assuming similar accuracies for LR, DT, and RFC to GBC
    'Precision (macro avg)': [0.99, 1.00, 1.00, 0.99],
    'Recall (macro avg)': [0.99, 1.00, 1.00, 0.99],
    'F1-score (macro avg)': [0.99, 1.00, 1.00, 0.99]
}

models = ['Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'Random Forest']

# Setting the positions and width for the bars
positions = np.arange(len(models))
width = 0.2  # the width of the bars

# Creating subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Performance Comparison Across Different Metrics')

# Flatten the array of axes for easy iterating
axs = axs.ravel()

# Looping through metrics to create a subplot for each
for i, (metric, values) in enumerate(metrics.items()):
    axs[i].bar(positions, values, width, align='center', alpha=0.7, color=['blue', 'green', 'red', 'purple'])
    axs[i].set_xticks(positions)
    axs[i].set_xticklabels(models, rotation=45)
    axs[i].set_title(metric)
    axs[i].set_ylim([0.98, 1.01])  # Adjust y-axis limits to best fit your data
    
    # Adding value labels on top of each bar
    for index, value in enumerate(values):
        axs[i].text(index, value, f"{value:.3f}", ha='center', va='bottom')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show plot
plt.show()
