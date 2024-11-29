import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

class DataVisualization:
    def __init__(self, json_path):
        # Load data from JSON
        with open(json_path, "r") as f:
            self.data = pd.DataFrame(json.load(f))
        self.output_folder = "data/visualizations/"
        os.makedirs(self.output_folder, exist_ok=True)

    def plot_time_series(self, feature):
        """Plots time series trends for a given feature."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=self.data,
            x="ICULOS",
            y=feature,
            hue="SepsisLabel",
            palette="tab10",
        )
        plt.title(f"Time Series Plot: {feature} (Sepsis vs Non-Sepsis)")
        plt.xlabel("ICULOS (Time)")
        plt.ylabel(feature)
        plt.legend(title="SepsisLabel", labels=["Non-Sepsis", "Sepsis"])
        output_path = os.path.join(self.output_folder, f"time_series_{feature}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved time series plot for {feature} to {output_path}")

    def plot_feature_distribution(self, feature):
        """Plots the distribution of a feature for Sepsis vs Non-Sepsis cases."""
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.data,
            x=feature,
            hue="SepsisLabel",
            kde=True,
            palette="tab10",
            bins=30,
        )
        plt.title(f"Distribution of {feature} (Sepsis vs Non-Sepsis)")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        output_path = os.path.join(self.output_folder, f"distribution_{feature}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved distribution plot for {feature} to {output_path}")

    def plot_correlation_heatmap(self):
        """Plots a heatmap of correlations between features."""
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.corr()
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="coolwarm",
            fmt=".2f",
            cbar=True,
        )
        plt.title("Correlation Heatmap of Features")
        output_path = os.path.join(self.output_folder, "correlation_heatmap.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved correlation heatmap to {output_path}")

    def plot_missing_data(self):
        """Plots a heatmap of missing data."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap")
        output_path = os.path.join(self.output_folder, "missing_data_heatmap.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved missing data heatmap to {output_path}")

    def plot_boxplots_by_sepsis(self, feature):
        """Plots boxplots of a feature grouped by SepsisLabel."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x="SepsisLabel", y=feature, palette="tab10")
        plt.title(f"Boxplot of {feature} by SepsisLabel")
        plt.xlabel("SepsisLabel")
        plt.ylabel(feature)
        output_path = os.path.join(self.output_folder, f"boxplot_{feature}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved boxplot for {feature} to {output_path}")


if __name__ == "__main__":
    # Path to the JSON file (e.g., train.json)
    json_file_path = "data/preprocessed_data/train.json"

    # Initialize the visualization class
    visualizer = DataVisualization(json_file_path)

    # Perform visualizations
    features_to_visualize = ["HR", "O2Sat", "Temp", "SBP", "MAP"]
    for feature in features_to_visualize:
        visualizer.plot_time_series(feature)
        visualizer.plot_feature_distribution(feature)
        visualizer.plot_boxplots_by_sepsis(feature)

    # Additional visualizations
    visualizer.plot_correlation_heatmap()
    visualizer.plot_missing_data()
