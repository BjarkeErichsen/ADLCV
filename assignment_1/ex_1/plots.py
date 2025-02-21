import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
results_folder = "results"
results_file = os.path.join(results_folder, "model_results.txt")
plot_file = os.path.join(results_folder, "final_accuracy_plot.png")

# Ensure the results folder exists
os.makedirs(results_folder, exist_ok=True)

# Dictionary to store final model results
results = {}

with open(results_file, "r") as f:
    current_model = None
    for line in f:
        line = line.strip()
        # Parse model header line
        model_match = re.match(r"Model (\d+): (.*)", line)
        if model_match:
            current_model = int(model_match.group(1))
            config = model_match.group(2)
            results[current_model] = {"config": config}
        # Parse the final accuracy line
        final_match = re.match(r"Final Accuracy over \d+ runs: Mean = ([\d.]+), Std = ([\d.]+)", line)
        if final_match and current_model is not None:
            mean_acc = float(final_match.group(1))
            std_acc = float(final_match.group(2))
            results[current_model]["mean"] = mean_acc
            results[current_model]["std"] = std_acc
        # Parse the confidence interval line
        ci_match = re.match(r"95% Confidence Interval: \[([\d.]+), ([\d.]+)\]", line)
        if ci_match and current_model is not None:
            ci_lower = float(ci_match.group(1))
            ci_upper = float(ci_match.group(2))
            results[current_model]["ci_lower"] = ci_lower
            results[current_model]["ci_upper"] = ci_upper

# Prepare data sorted by model number
models_sorted = sorted(results.keys())
model_labels = [f"Model {m}" for m in models_sorted]
mean_accuracies = [results[m]["mean"] for m in models_sorted]
# Calculate asymmetric error bars: differences between mean and CI bounds
lower_errors = [results[m]["mean"] - results[m]["ci_lower"] for m in models_sorted]
upper_errors = [results[m]["ci_upper"] - results[m]["mean"] for m in models_sorted]
asymmetric_errors = np.array([lower_errors, upper_errors])

# Set Seaborn theme for a polished look
sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(12, 8))
# Use a Viridis gradient palette for the bars
palette = sns.color_palette("viridis", len(models_sorted))

bars = plt.bar(
    model_labels,
    mean_accuracies,
    yerr=asymmetric_errors,
    capsize=8,
    color=palette,
    alpha=0.85,
    edgecolor="k"
)

plt.xlabel("Model", fontsize=16, labelpad=10)
plt.ylabel("Final Accuracy", fontsize=16, labelpad=10)
plt.title("Final Accuracy with 95% Confidence Intervals", fontsize=20, fontweight="bold", pad=15)
plt.ylim(0.65, 0.85)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to {plot_file}")
