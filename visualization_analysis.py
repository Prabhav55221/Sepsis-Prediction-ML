import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import random
import logging


class DataVisualization:
    def __init__(self, json_path):
        # Set up logging
        logging.basicConfig(
            filename="./logs/visualization.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Initializing DataVisualization class.")

        # Load data from JSON
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.output_folder = "data/visualizations/"
        os.makedirs(self.output_folder, exist_ok=True)
        logging.info(f"Output folder set to {self.output_folder}.")

    def extract_flat_data(self):
        """Extract flat data for aggregation and distribution plots."""
        records = []
        for patient_id, patient_data in self.data.items():
            sepsis_label = patient_data.get("sepsis_label", 0)
            for observation in patient_data["timeseries"]:
                observation["sepsis_label"] = sepsis_label
                observation["PatientID"] = patient_id
                records.append(observation)
        return pd.DataFrame(records)

    def plot_patient_level(self, feature, num_patients=10):
        """
        Plots time series for a feature for multiple patients (sepsis vs non-sepsis) on the same graph.
        """
        # Separate patients by sepsis_label
        sepsis_patients = [pid for pid, pdata in self.data.items() if pdata.get("sepsis_label", 0) == 1 and len(pdata['timeseries']) > 100]
        non_sepsis_patients = [pid for pid, pdata in self.data.items() if pdata.get("sepsis_label", 0) == 0 and len(pdata['timeseries']) > 100]

        # Select random patients
        selected_sepsis_patients = random.sample(sepsis_patients, min(num_patients, len(sepsis_patients)))
        selected_non_sepsis_patients = random.sample(non_sepsis_patients, min(num_patients, len(non_sepsis_patients)))

        logging.info(
            f"Selected {len(selected_sepsis_patients)} sepsis patients and {len(selected_non_sepsis_patients)} non-sepsis patients."
        )

        # Plot time series for both groups
        plt.figure(figsize=(12, 8))
        for patient_id in selected_sepsis_patients:
            timeseries = pd.DataFrame(self.data[patient_id]["timeseries"])
            if feature in timeseries:
                sns.lineplot(
                    data=timeseries,
                    x="ICULOS",
                    y=feature,
                    label=f"Sepsis Patient {patient_id}",
                    alpha=0.7,
                    linestyle="--",
                    color="red",
                )

        for patient_id in selected_non_sepsis_patients:
            timeseries = pd.DataFrame(self.data[patient_id]["timeseries"])
            if feature in timeseries:
                sns.lineplot(
                    data=timeseries,
                    x="ICULOS",
                    y=feature,
                    label=f"Non-Sepsis Patient {patient_id}",
                    alpha=0.7,
                    linestyle="-",
                    color="blue",
                )

        # Finalize plot
        plt.title(f"Time Series of {feature} for Sepsis and Non-Sepsis Patients")
        plt.xlabel("ICULOS (Time)")
        plt.ylabel(feature)
        plt.legend(title="Patients", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        output_path = os.path.join(self.output_folder, f"patients_{feature}_comparison.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved combined time series plot for {feature} to {output_path}.")

    def plot_aggregated_trends(self, feature):
        """Plots aggregated trends of a feature for sepsis and non-sepsis patients."""
        flat_data = self.extract_flat_data()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=flat_data, x="ICULOS", y=feature, hue="sepsis_label", estimator="mean", ci="sd")
        plt.title(f"Aggregated Trends of {feature} (Sepsis vs Non-Sepsis)")
        plt.xlabel("ICULOS (Time)")
        plt.ylabel(feature)
        output_path = os.path.join(self.output_folder, f"aggregated_{feature}.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved aggregated trends plot for {feature} to {output_path}.")

    def plot_counts(self):
        """Plots counts of sepsis vs non-sepsis patients."""
        sepsis_counts = pd.Series({pid: pdata["sepsis_label"] for pid, pdata in self.data.items()})
        plt.figure(figsize=(8, 6))
        sns.countplot(x=sepsis_counts, palette="tab10")
        plt.title("Counts of Sepsis vs Non-Sepsis Patients")
        plt.xlabel("sepsis_label")
        plt.ylabel("Number of Patients")
        output_path = os.path.join(self.output_folder, "sepsis_counts.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved sepsis count plot to {output_path}.")

    def plot_feature_distributions(self, feature):
        """Plots feature distributions for sepsis vs non-sepsis patients."""
        flat_data = self.extract_flat_data()
        plt.figure(figsize=(10, 6))
        sns.histplot(data=flat_data, x=feature, hue="sepsis_label", kde=True, palette="tab10")
        plt.title(f"Distribution of {feature} (Sepsis vs Non-Sepsis)")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        output_path = os.path.join(self.output_folder, f"distribution_{feature}.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved distribution plot for {feature} to {output_path}.")

    def plot_class_distribution(self, col):
        """
        Plots the distribution of unique values in a column, separated by sepsis label.
        If the column is 'Age', bin it by 10-year intervals.
        """

        flat_data = self.extract_flat_data()

        # Handle 'Age' column separately by binning into 10-year intervals
        if col == "Age":
            flat_data["Age_Bin"] = pd.cut(flat_data["Age"], bins=range(0, 101, 10))
            group_col = "Age_Bin"
            xlabel = "Age Range (Years)"
        else:
            group_col = col
            xlabel = col

        # Group data by the column (or bins) and SepsisLabel
        col_sepsis_counts = (
            flat_data.groupby([group_col, "sepsis_label"])["PatientID"]
            .nunique()
            .reset_index()
            .rename(columns={"PatientID": "Count"})
        )

        # Bar plot for the column distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=col_sepsis_counts,
            x=group_col,
            y="Count",
            hue="sepsis_label",
            palette="coolwarm",
        )
        plt.title(f"Distribution of {col} (Sepsis vs Non-Sepsis)")
        plt.xlabel(xlabel)
        plt.ylabel("Number of Unique Patients")
        plt.xticks(rotation=45)
        output_path = os.path.join(self.output_folder, f"{col}_distribution_sepsis.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved {col} distribution plot (with sepsis split) to {output_path}.")

    def plot_feature_counts(self):
        """Plots the counts of non-missing values for each feature."""
        flat_data = self.extract_flat_data()
        feature_counts = flat_data.notnull().sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_counts.index, y=feature_counts.values, palette="viridis")
        plt.title("Feature Counts (Non-Missing Values)")
        plt.xlabel("Feature")
        plt.ylabel("Non-Missing Value Count")
        plt.xticks(rotation=90)
        output_path = os.path.join(self.output_folder, "feature_counts.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved feature counts plot to {output_path}.")


if __name__ == "__main__":
    # Path to the JSON file (e.g., train.json)
    json_file_path = "data/preprocessed_data/train.json"

    # Initialize the visualization class
    visualizer = DataVisualization(json_file_path)

    # Patient-level analysis (combined plots for both sepsis and non-sepsis)
    features_to_visualize = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "Platelets", "DBP", "Glucose", "Potassium", "Hct", "FiO2"]
    for feature in features_to_visualize:
        visualizer.plot_patient_level(feature, num_patients=1)

    # Aggregated trends
    for feature in features_to_visualize:
        visualizer.plot_aggregated_trends(feature)

    # Count plot
    visualizer.plot_counts()

    # Feature distributions
    for feature in features_to_visualize:
        visualizer.plot_feature_distributions(feature)

    # Age distribution
    visualizer.plot_class_distribution('Age')
    visualizer.plot_class_distribution('Unit1')
    visualizer.plot_class_distribution('Gender')

    # # Feature counts
    visualizer.plot_feature_counts()
