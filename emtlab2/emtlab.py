import matplotlib.pyplot as plt

# Set the style
plt.style.use('seaborn-darkgrid')

# Create figure and axes
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Titles for each subplot
titles = ['Root Mean Square Error (RMSE)', 'R Squared (R^2)', 'Accuracy']

# Metrics to plot
metrics = ['RMSE', 'RSquare', 'Accuarcy']

# Generate each bar plot
for i, metric in enumerate(metrics):
    ax[i].bar(data['ModelName'], data[metric], color='skyblue')
    ax[i].set_title(titles[i])
    ax[i].set_xlabel('Model Name')
    ax[i].set_ylabel(metric)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/mnt/data/model_performance_charts.png')

# Show the plot
plt.show()

# Provide path to the saved image
'/mnt/data/model_performance_charts.png'
