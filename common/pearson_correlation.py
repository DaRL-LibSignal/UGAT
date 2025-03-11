import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read log file
log_file = "/Users/khoavo2003/PycharmProjects/UGAT/data/output_data/2025_03_09-18_21_24_BRF.log"

# Refined regular expressions for extracting values
sim_pattern = re.compile(
    r"Sim Rollout episode:(\d+)/\d+, sim avg travel time:([\d\.]+), rewards:([-]?\d+\.\d+), queue:([\d\.]+), delay:([\d\.]+), throughput:(\d+)"
)
real_pattern = re.compile(
    r"Real rollout step:(\d+)/\d+, travel time:([\d\.]+), rewards:([-]?\d+\.\d+), queue:([\d\.]+), delay:([\d\.]+), throughput:(\d+)"
)

# Store extracted data
data = []

with open(log_file, "r") as f:
    lines = f.readlines()
    for i in range(len(lines) - 1):  # Iterate over lines
        sim_match = sim_pattern.search(lines[i])
        real_match = real_pattern.search(lines[i + 1])

        if sim_match and real_match:
            step = int(sim_match.group(1))
            sim_values = list(map(float, sim_match.groups()[1:]))  # Convert values to floats
            real_values = list(map(float, real_match.groups()[1:]))  # Convert values to floats

            data.append([step] + sim_values + real_values)

# Ensure data is extracted correctly
if not data:
    raise ValueError("No valid data extracted. Check log file formatting.")

# Create DataFrame
columns = [
    "step",
    "sim_travel_time",
    "sim_rewards",
    "sim_queue",
    "sim_delay",
    "sim_throughput",
    "real_travel_time",
    "real_rewards",
    "real_queue",
    "real_delay",
    "real_throughput",
]
df = pd.DataFrame(data, columns=columns)

# Compute Pearson correlation
correlations = df.iloc[:, 1:].corr(method="pearson")

# Plot graphs
metrics = ["travel_time", "rewards", "queue", "delay", "throughput"]

for metric in metrics:
    plt.figure(figsize=(5, 5))  # Set square figure size
    ax = plt.gca()
    sns.regplot(x=df[f"sim_{metric}"], y=df[f"real_{metric}"], scatter_kws={'alpha': 0.6}, fit_reg=False)

    # Add diagonal line (y = x)
    min_val = min(df[f"sim_{metric}"].min(), df[f"real_{metric}"].min())
    max_val = max(df[f"sim_{metric}"].max(), df[f"real_{metric}"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='dotted', color='red', label="y = x")

    # Set aspect ratio to 1 (square)
    ax.set_aspect('equal', adjustable='datalim')

    corr_value = correlations.loc[f"sim_{metric}", f"real_{metric}"]
    plt.title(f"{metric.capitalize()} Correlation: {corr_value:.3f}")
    plt.xlabel(f"Simulation {metric}")
    plt.ylabel(f"Real {metric}")
    plt.legend()

    plt.savefig(f"correlation_{metric}.png")
    plt.close()

print("Plots saved successfully!")