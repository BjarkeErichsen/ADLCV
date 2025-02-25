import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load results from JSON file
RESULTS_FILE = "results/all_resultsauto_update3.json"
OUTPUT_PLOT = "results/validation_accuracy3.png"

with open(RESULTS_FILE, "r") as f:
    results_data = json.load(f)

# Prepare data for visualization: group runs by model configuration.
# We also keep a list of model_ids if needed later.
model_data_dict = {}
for model_id, model_data in results_data.items():
    config = model_data["config"]
    # Include the model id in the label for clarity
    # If several runs share the same config, we will add a unique index later.
    base_label = f"embed: {config['embed_dim']}, heads: {config['num_heads']}, layers: {config['num_layers']}"
    if base_label not in model_data_dict:
        model_data_dict[base_label] = []
    model_data_dict[base_label].append(model_data["accuracies"])

# Create a DataFrame for Seaborn plotting.
records = []
for model_label, runs in model_data_dict.items():
    num_epochs = len(runs[0])  # assume all runs have same number of epochs
    for epoch in range(num_epochs):
        epoch_accuracies = [run[epoch] for run in runs]
        mean_acc = sum(epoch_accuracies) / len(epoch_accuracies)
        std_acc = (sum((x - mean_acc) ** 2 for x in epoch_accuracies) / len(epoch_accuracies)) ** 0.5
        records.append({"Model": model_label, "Epoch": epoch + 1, "Accuracy": mean_acc, "Std": std_acc})
df = pd.DataFrame(records)

# Set up the plotting style and figure size.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Create a palette with a distinct color for each model.
models = df["Model"].unique()
palette = sns.color_palette("tab10", len(models))

# Plot each model's validation accuracy with its confidence interval.
for i, model_label in enumerate(models):
    subset = df[df["Model"] == model_label]
    # Compute final accuracy (last epoch) and convert to percentage.
    final_acc = subset["Accuracy"].iloc[-1] * 100
    # Create a label with a model number and its final accuracy.
    label_with_info = f"Model {i+1} w. acc: {final_acc:.1f}%"
    plt.plot(
        subset["Epoch"],
        subset["Accuracy"],
        marker="o",
        label=label_with_info,
        linewidth=2.5,
        color=palette[i]
    )
    plt.fill_between(
        subset["Epoch"],
        subset["Accuracy"] - subset["Std"],
        subset["Accuracy"] + subset["Std"],
        alpha=0.2,
        color=palette[i]
    )

# Titles & labels
plt.title("Validation Accuracy\nShowing Mean ± Std for Each Model", fontsize=14)
plt.xlabel("Epoch")
plt.xticks(df["Epoch"].unique())  # Show only integer epochs on the x-axis
plt.ylabel("Accuracy [%]")
plt.ylim(0.54, 0.85)  # Adjust the y-axis for clarity
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

# Save and close the plot
os.makedirs("results", exist_ok=True)
plt.savefig(OUTPUT_PLOT, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot saved to {OUTPUT_PLOT}")
