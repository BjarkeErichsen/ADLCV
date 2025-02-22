import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data (replace with real evaluations)
np.random.seed(42)
num_samples = 30

temperature_values      = np.array([0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.0])
coherence_scores        = np.array([0.3, 0.3, 0.7, 0.8, 0.9, 1.0, 0.8, 0.2, 0.1, 0.0])
repetitiveness_scores   = np.array([1.0, 0.9, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

# Create DataFrame
df = pd.DataFrame({
    "Repetitiveness": repetitiveness_scores,
    "Coherence": coherence_scores,
    "Temperature": temperature_values
})

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="Coherence", y="Repetitiveness", hue="Temperature",
    palette="coolwarm", marker="o", edgecolor="black", s=200
)

# Customize plot
plt.title("Effect of Temperature on Storytelling")
plt.xlabel("Coherence Score")
plt.ylabel("Repetitiveness Score")
plt.legend(title="Temperature", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)

# Save plot
plt.savefig("temperature_effect_storytelling.png", dpi=300, bbox_inches="tight")
