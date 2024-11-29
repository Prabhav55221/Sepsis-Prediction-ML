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

    def plot_patient_level(self, feature, sepsis_label, num_patients=10):
        """Plots time series for multiple patients in one plot."""
        patients = [pid for pid, pdata in self.data.items() if pdata.get("sepsis_label", 0) == sepsis_label]
        selected_patients = random.sample(patients, min(num_patients, len(patients)))
        logging.info(f"Selected {len(selected_patients)} patients for sepsis_label={sepsis_label}.")

        plt.figure(figsize=(12, 8))
        for patient_id in selected_patients:
            timeseries = pd.DataFrame(self.data[patient_id]["timeseries"])
            if feature in timeseries:
                sns.lineplot(
                    data=timeseries,
                    x="ICULOS",
                    y=feature,
                    label=f"Patient {patient_id}",
                    alpha=0.7,
                )

        plt.title(f"Time Series of {feature} (Sepsis={sepsis_label}) for {num_patients} Patients")
        plt.xlabel("ICULOS (Time)")
        plt.ylabel(feature)
        plt.legend(title="Patients", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        output_path = os.path.join(self.output_folder, f"patients_{feature}_sepsis_{sepsis_label}.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved combined time series plot for {feature} (Sepsis={sepsis_label}) to {output_path}.")

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


if __name__ == "__main__":
    # Path to the JSON file (e.g., train.json)
    json_file_path = "data/preprocessed_data/train.json"

    # Initialize the visualization class
    visualizer = DataVisualization(json_file_path)

    # Patient-level analysis (combined plots)
    features_to_visualize = ["HR", "O2Sat", "Temp", "SBP", "MAP"]
    for feature in features_to_visualize:
        visualizer.plot_patient_level(feature, sepsis_label=1, num_patients=10)  # Patients with sepsis
        visualizer.plot_patient_level(feature, sepsis_label=0, num_patients=10)  # Patients without sepsis

    # Aggregated trends
    for feature in features_to_visualize:
        visualizer.plot_aggregated_trends(feature)

    # Count plot
    visualizer.plot_counts()

    # Feature distributions
    for feature in features_to_visualize:
        visualizer.plot_feature_distributions(feature)
