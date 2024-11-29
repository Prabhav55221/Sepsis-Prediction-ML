import os
import pandas as pd
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    filename="./logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting patient-level data preprocessing script.")

try:
    # Define paths
    input_folder = "data/physionet.org/files/challenge-2019/1.0.0/training"
    output_folder = "data/preprocessed_data/"
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Output folder created at {output_folder}.")

    # Initialize a dictionary to store patient data
    patient_data = {}

    # Process files
    for folder in ["training_setA", "training_setB"]:

        folder_path = os.path.join(input_folder, folder)
        logging.info(f"Processing folder: {folder_path}")
        
        for file in tqdm(os.listdir(folder_path)):

            if file.endswith(".psv"):
                file_path = os.path.join(folder_path, file)
                logging.info(f"Reading file: {file_path}")
                
                # Read the PSV file
                df = pd.read_csv(file_path, sep="|")  
                patient_id = file.replace(".psv", "")
                sepsis_label = int(df["SepsisLabel"].iloc[-1])  
                
                # Store data in the dictionary
                patient_data[patient_id] = {
                    "timeseries": df.drop(columns=["SepsisLabel"]).to_dict(orient="records"),
                    "sepsis_label": sepsis_label
                }

    logging.info("Successfully parsed all patient data.")

    # Split patient IDs into train, validation, and test sets
    patient_ids = list(patient_data.keys())
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=2/3, random_state=42)

    # Create splits
    splits = {
        "train": {pid: patient_data[pid] for pid in train_ids},
        "val": {pid: patient_data[pid] for pid in val_ids},
        "test": {pid: patient_data[pid] for pid in test_ids},
    }

    # Save splits to JSON
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_folder, f"{split_name}.json")
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=4)
        logging.info(f"{split_name.capitalize()} data saved to {output_file}.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    raise
