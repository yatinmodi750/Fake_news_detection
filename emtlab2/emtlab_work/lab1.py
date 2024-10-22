import pandas as pd
import matplotlib.pyplot as plt

# print(plt.style.available)

# Load the data from your Excel file
file_path = 'Result.xlsx'  # Make sure to update this to your actual file path
data = pd.read_excel(file_path)

# Set the style
plt.style.use('seaborn-v0_8-notebook')

# Create figure and axes for 3 subplots
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

# Adjust layout for better spacing
plt.tight_layout()

# Optionally, save the figure to a file
plt.savefig('model_performance_charts.eps', format='eps')
plt.savefig('model_performance_charts.png')

# Display the plot
plt.show()
