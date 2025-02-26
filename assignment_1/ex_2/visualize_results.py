import pandas as pd
import matplotlib.pyplot as plt

# Get CSV file path
csv_path = "model_performance.csv"

# Load the data
df = pd.read_csv(csv_path)

# Keep only the rows corresponding to the last epoch for each model type
df_last_epoch = df.loc[df.groupby('model_name')['epoch'].idxmax()]

# Extract clean model names
df_last_epoch['model_name'] = df_last_epoch['model_name'].apply(lambda x: "size_" + x.split('models/')[1].split('.pth')[0])

# Create dot plot for test accuracy
plt.figure(figsize=(8, 6))
plt.scatter(df_last_epoch['model_name'], df_last_epoch['test_accuracy'], s=100, marker='o')

plt.ylabel('Test Accuracy')
plt.title('Test accuracy using model weights with highest validation score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()