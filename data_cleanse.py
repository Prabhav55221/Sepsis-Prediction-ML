import pandas as pd
import numpy as np
import json
import logging
import os

# Configure logging
logging.basicConfig(
    filename="./logs/data_processor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class DataProcessor:
    def __init__(self):
        logging.info("Initialized DataProcessor class.")

    def process_json(self, json_file):
        """
        Processes a JSON file containing patient data.

        Parameters:
        - json_file: Path to the JSON file.

        Returns:
        - processed_df: DataFrame aggregated for all patients.
        """
        logging.info(f"Loading data from {json_file}.")

        # Load JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract and process patient-level data
        processed_data = []
        for patient_id, patient_data in data.items():
            timeseries = pd.DataFrame(patient_data["timeseries"])

            # Handle missing values in the timeseries
            timeseries = timeseries.fillna(timeseries.mean())

            # Aggregate patient data
            patient_summary = {
                "PatientID": patient_id,
                "Age": timeseries["Age"].iloc[0],
                "sepsis_label": patient_data.get("sepsis_label", 0),
                "ICULOS": timeseries["ICULOS"].iloc[-1],
                "Gender": timeseries["Gender"].iloc[0],
                "HR": timeseries["HR"].mean(),
                "MAP": timeseries["MAP"].mean(),
                "O2Sat": timeseries["O2Sat"].mean(),
                "SBP": timeseries["SBP"].mean(),
                "Resp": timeseries["Resp"].mean(),
                "DBP": timeseries["DBP"].mean(),
                "Unit1": timeseries["Unit1"].mode()[0]
                if not timeseries["Unit1"].isna().all()
                else np.nan,
                "Unit2": timeseries["Unit2"].mode()[0]
                if not timeseries["Unit2"].isna().all()
                else np.nan,
                "Temp": timeseries["Temp"].mean(),
                "Glucose": timeseries["Glucose"].mean(),
                "FiO2": timeseries["FiO2"].mean(),
                "Hct": timeseries["Hct"].mean(),
                "Potassium": timeseries["Potassium"].mean(),
            }

            # Append to processed data
            processed_data.append(patient_summary)

        # Create a DataFrame
        processed_df = pd.DataFrame(processed_data)

        # Fill any remaining NaNs with dataset-level means or modes
        for col in processed_df.columns:
            if col in ['PatientID', 'Age', 'sepsis_label', 'ICULOS']:
                continue
            curr_mode = processed_df[col].mode()[0]
            for i in range(len(processed_df[col])):
                if np.isnan(processed_df[col][i]):
                    processed_df[col][i] = curr_mode

        logging.info(f"Data processing completed for {json_file}.")
        return processed_df


if __name__ == "__main__":

    processor = DataProcessor()

    # Paths to JSON files
    train_json = "data/preprocessed_data/train.json"
    val_json = "data/preprocessed_data/val.json"
    test_json = "data/preprocessed_data/test.json"

    # Process data
    train_data = processor.process_json(train_json)
    val_data = processor.process_json(val_json)
    test_data = processor.process_json(test_json)

    # Save processed data for inspection
    os.makedirs("data/processed_data", exist_ok=True)
    train_data.to_csv("data/processed_data/processed_train.csv", index=False)
    val_data.to_csv("data/processed_data/processed_val.csv", index=False)
    test_data.to_csv("data/processed_data/processed_test.csv", index=False)

    logging.info(f"All Data is cleaned and processing is complete.")