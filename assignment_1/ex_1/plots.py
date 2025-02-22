import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load results from JSON file
RESULTS_FILE = "results/all_results.json"
OUTPUT_PLOT = "results/validation_accuracy.png"

with open(RESULTS_FILE, "r") as f:
    results_data = json.load(f)

# Prepare data for visualization
data = []
model_labels = {}

for model_id, model_data in results_data.items():
    config = model_data["config"]
    model_label = f"embed: {config['embed_dim']}, heads: {config['num_heads']}, layers: {config['num_layers']}"
    
    if model_label not in model_labels:
        model_labels[model_label] = []
    
    model_labels[model_label].append(model_data["accuracies"])

# Create a DataFrame for Seaborn
df = []

for model_label, runs in model_labels.items():
    num_epochs = len(runs[0])  # Assume all runs have the same epochs
    for epoch in range(num_epochs):
        epoch_accuracies = [run[epoch] for run in runs]  # Collect all runs' accuracy at this epoch
        mean_acc = sum(epoch_accuracies) / len(epoch_accuracies)  # Average accuracy
        std_acc = (sum((x - mean_acc) ** 2 for x in epoch_accuracies) / len(epoch_accuracies)) ** 0.5  # Standard deviation
        
        df.append({"Model": model_label, "Epoch": epoch + 1, "Accuracy": mean_acc, "Std": std_acc})

df = pd.DataFrame(df)

# Plot settings
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Plot each model with confidence intervals
for model_label in df["Model"].unique():
    subset = df[df["Model"] == model_label]
    plt.plot(subset["Epoch"], subset["Accuracy"], marker="o", label=model_label, linewidth=2.5)
    plt.fill_between(subset["Epoch"], subset["Accuracy"] - subset["Std"], subset["Accuracy"] + subset["Std"], alpha=0.2)

# Titles & Labels
plt.title("Validation Accuracy\nShowing Mean ± Std for Each Model", fontsize=14)
plt.xlabel("Epoch")
plt.xticks(df["Epoch"].unique())  # Force x-axis to show only integer epochs
plt.ylabel("Accuracy [%]")
plt.ylim(0.4, 0.8)  # Adjust for clarity
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

# Save figure
os.makedirs("results", exist_ok=True)
plt.savefig(OUTPUT_PLOT, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot saved to {OUTPUT_PLOT}")





# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # File paths
# results_folder = "results"
# results_file = os.path.join(results_folder, "model_results.txt")
# plot_file = os.path.join(results_folder, "final_accuracy_plot.png")

# # Ensure the results folder exists
# os.makedirs(results_folder, exist_ok=True)

# # Dictionary to store final model results
# results = {}

# with open(results_file, "r") as f:
#     current_model = None
#     for line in f:
#         line = line.strip()
#         # Parse model header line
#         model_match = re.match(r"Model (\d+): (.*)", line)
#         if model_match:
#             current_model = int(model_match.group(1))
#             config = model_match.group(2)
#             results[current_model] = {"config": config}
#         # Parse the final accuracy line
#         final_match = re.match(r"Final Accuracy over \d+ runs: Mean = ([\d.]+), Std = ([\d.]+)", line)
#         if final_match and current_model is not None:
#             mean_acc = float(final_match.group(1))
#             std_acc = float(final_match.group(2))
#             results[current_model]["mean"] = mean_acc
#             results[current_model]["std"] = std_acc
#         # Parse the confidence interval line
#         ci_match = re.match(r"95% Confidence Interval: \[([\d.]+), ([\d.]+)\]", line)
#         if ci_match and current_model is not None:
#             ci_lower = float(ci_match.group(1))
#             ci_upper = float(ci_match.group(2))
#             results[current_model]["ci_lower"] = ci_lower
#             results[current_model]["ci_upper"] = ci_upper

# # Prepare data sorted by model number
# models_sorted = sorted(results.keys())
# model_labels = [f"Model {m}" for m in models_sorted]
# mean_accuracies = [results[m]["mean"] for m in models_sorted]
# # Calculate asymmetric error bars: differences between mean and CI bounds
# lower_errors = [results[m]["mean"] - results[m]["ci_lower"] for m in models_sorted]
# upper_errors = [results[m]["ci_upper"] - results[m]["mean"] for m in models_sorted]
# asymmetric_errors = np.array([lower_errors, upper_errors])

# # Set Seaborn theme for a polished look
# sns.set_theme(style="whitegrid", context="talk")

# plt.figure(figsize=(12, 8))
# # Use a Viridis gradient palette for the bars
# palette = sns.color_palette("viridis", len(models_sorted))

# bars = plt.bar(
#     model_labels,
#     mean_accuracies,
#     yerr=asymmetric_errors,
#     capsize=8,
#     color=palette,
#     alpha=0.85,
#     edgecolor="k"
# )

# plt.xlabel("Model", fontsize=16, labelpad=10)
# plt.ylabel("Final Accuracy", fontsize=16, labelpad=10)
# plt.title("Final Accuracy with 95% Confidence Intervals", fontsize=20, fontweight="bold", pad=15)
# plt.ylim(0.65, 0.85)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# plt.tight_layout()
# plt.savefig(plot_file, dpi=300, bbox_inches="tight")
# print(f"Plot saved to {plot_file}")
