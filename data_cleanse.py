import pandas as pd
import numpy as np
import json
import logging
import os
from tqdm import tqdm

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
        for patient_id, patient_data in tqdm(data.items()):
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
                "HR_median": timeseries["HR"].median(),
                "HR_FINAL": timeseries["HR"].iloc[-1],
                "MAP": timeseries["MAP"].mean(),
                "MAP_median": timeseries["MAP"].median(),
                "MAP_FINAL": timeseries["MAP"].iloc[-1],
                "O2Sat": timeseries["O2Sat"].mean(),
                "O2Sat_median": timeseries["O2Sat"].median(),
                "O2Sat_FINAL": timeseries["O2Sat"].iloc[-1],
                "SBP": timeseries["SBP"].mean(),
                "SBP_median": timeseries["SBP"].median(),
                "SBP_FINAL": timeseries["SBP"].iloc[-1],
                "Resp": timeseries["Resp"].mean(),
                "Resp_median": timeseries["Resp"].median(),
                "Resp_FINAL": timeseries["Resp"].iloc[-1],
                "DBP": timeseries["DBP"].mean(),
                "DBP_median": timeseries["DBP"].median(),
                "DBP_FINAL": timeseries["DBP"].iloc[-1],
                "Unit1": timeseries["Unit1"].mode()[0]
                if not timeseries["Unit1"].isna().all()
                else np.nan,
                "Unit2": timeseries["Unit2"].mode()[0]
                if not timeseries["Unit2"].isna().all()
                else np.nan,
                "Temp": timeseries["Temp"].mean(),
                "Temp_median": timeseries["Temp"].median(),
                "Temp_FINAL": timeseries["Temp"].iloc[-1],
                "Glucose": timeseries["Glucose"].mean(),
                "Glucose_median": timeseries["Glucose"].median(),
                "Glucose_FINAL": timeseries["Glucose"].iloc[-1],
                "FiO2": timeseries["FiO2"].mean(),
                "FiO2_median": timeseries["FiO2"].median(),
                "FiO2_FINAL": timeseries["FiO2"].iloc[-1],
                "Hct": timeseries["Hct"].mean(),
                "Hct_median": timeseries["Hct"].median(),
                "Hct_FINAL": timeseries["Hct"].iloc[-1],
                "WBC": timeseries["WBC"].mean(),
                "WBC_median": timeseries["WBC"].median(),
                "WBC_FINAL": timeseries["WBC"].iloc[-1],
                "Calcium": timeseries["Calcium"].mean(),
                "Calcium_median": timeseries["Calcium"].median(),
                "Calcium_FINAL": timeseries["Calcium"].iloc[-1],
                "Chloride": timeseries["Chloride"].mean(),
                "Chloride_median": timeseries["Chloride"].median(),
                "Chloride_FINAL": timeseries["Chloride"].iloc[-1],
                "Magnesium": timeseries["Magnesium"].mean(),
                "Magnesium_median": timeseries["Magnesium"].median(),
                "Magnesium_FINAL": timeseries["Magnesium"].iloc[-1],
                "pH": timeseries["pH"].mean(),
                "pH_median": timeseries["pH"].median(),
                "pH_FINAL": timeseries["pH"].iloc[-1],
                "BUN": timeseries["BUN"].mean(),
                "BUN_median": timeseries["BUN"].median(),
                "BUN_FINAL": timeseries["BUN"].iloc[-1],
            }

            # Append to processed data
            processed_data.append(patient_summary)

        # Create a DataFrame
        processed_df = pd.DataFrame(processed_data)

        logging.info(f"DataFrame Created")

        # Impute!
        for col in tqdm(processed_df.columns):
            if col in ['PatientID', 'Age', 'sepsis_label', 'ICULOS']:
                continue
            mean = processed_df[col].mean()
            processed_df[col].fillna(mean, inplace=True)

        logging.info(f"Imputation Done.\n")
        logging.info(f"Data processing completed for {json_file}.")
        return processed_df

if __name__ == "__main__":

    processor = DataProcessor()

    # Paths to JSON files
    train_json = "data/preprocessed_data/train.json"
    val_json = "data/preprocessed_data/val.json"
    test_json = "data/preprocessed_data/test.json"

    # Save processed data for inspection
    os.makedirs("data/processed_data", exist_ok=True)

    # Process data
    test_data = processor.process_json(test_json)
    test_data.to_csv("data/processed_data/processed_test.csv", index=False)

    train_data = processor.process_json(train_json)
    train_data.to_csv("data/processed_data/processed_train.csv", index=False)

    val_data = processor.process_json(val_json)
    val_data.to_csv("data/processed_data/processed_val.csv", index=False)

    logging.info(f"All Data is cleaned and processing is complete.")