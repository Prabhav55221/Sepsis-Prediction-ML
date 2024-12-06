# Causal Structure Learning for Sepsis Prediction in ICU Patients

## Submission Details

```
Subject: EN.601.675
Submitted By:
    Prabhav Singh (psingh54)
    Aravind Kavaturu (akavutu2)
    Mukund Iyengar (miyenga2)
    Sandesh Rangreji (srangre1)
```

Link to Presentation - https://docs.google.com/presentation/d/1g3y-otPHT_9zfrosbfByGmatw2ygRKpEWuV1QGrEMes/edit?usp=sharing

The Writeup is Uploaded with the Repository. Please contact us if data takes too long to preprocess - We can upload the same on Drive if needed.

> Please follow the steps below to replicate results. Note that our environment setup is tested on Apple Silicon (M2 and M3) and on Apple Intel Based Macbook.

## Setup For Replication

#### Environment Setup

> To replicate the codebase, first clone the repository to your local. Ensure you have Conda/Miniconda or some kind of environment handling enabled in your local environment. We do provide an environment.yml file for Conda. For other package management tools, you can use the same packages as that.

**It will take time to clone the repo since we include the data as a part of the repo!**

```bash
git clone <repo-link>
cd Sepsis-Prediction-ML
conda env create -f environment.yml
```

> **Important** Note that *only if you are using an Apple Silicon based Macbook*, you **MIGHT** have to download the below library with brew for XGBoost to work. In some cases, depending on Conda Version, it might install it automatically. If not, run the below command!:

```bash
# Only use if needed!
brew install libomp
```

#### Base Data

We provide the data required for this project as part of this Github Repo. You can access the original data, if required, by running the below command. We however, store the data in the repo in the directory ```./data/physionet.org```.

```bash
# OPTIONAL - You don't need to do this since we package the decompressed data as part of the repo.
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
```

Once you have cloned the repo, please run the first script to preprocess the data and create the train, test and validation data in JSON format. Refer to our paper to understand the data storage format.

```bash
conda activate sepsis-prediction
# ENSURE THESE FOLDERS ARE CREATED IN THESE PATHS!
mkdir data/preprocessed_data
mkdir logs
python preprocess.py
```

Now you should have train, test and val JSON in the path ```data/preprocessed_data```.

#### Replication of Results

Follow the below steps to replicate:

**Step 1**

Create the visualizations used by us in our writeup. For this, run the below command as it is. If you want to observe different visualizations, feel free to change the parameters in the main function for python file.

```bash
# This will take time - 3GBs worth of data!
python visualization_analysis.py
```

Logs will be stored in the ```./logs``` folder. Visualizations will be in ```data/visualizations```.

**Step 2**

Missing Data Handling and Aggregation - As described in the writeup, we impute missing data and then aggregate to create training data. For this, run the below command - processed_data will be stored in ```data/processed_data```.

```bash
# Ignore the warnings!
python data_cleanse.py
```

**Step 3**

Train the models! As described in writeup, we use two models - **Bayesian Network with Hill Climb**, **XGBoost**. You can run both models in below way and experiement with the parameters based on your use-case. To replicate our results (best), just use the commands below:

- Train Bayesian Model & Evaluate

The Bayesian Model will also store the selected features in the results folder. Please refer to that to see the final features. Please refer to the code for other options. Below are the one's we used for the final version.

```bash
python model_training.py --train data/processed_data/processed_train.csv --val data/processed_data/processed_val.csv --test data/processed_data/processed_test.csv --model bayesian --scoring_method "bicscore"

python evaluate.py --pred_file data/predictions/test_predictions_bayesian.csv --output_path data/results/test_evaluation_bayesian.txt
python evaluate.py --pred_file data/predictions/val_predictions_bayesian.csv --output_path data/results/val_evaluation_bayesian.txt
```

- Train XGBoost & Evaluate

> **DISCLAIMER** - Please run the Bayesian model first if you want to run the XGBOOST on Bayesian Features. If you don't, you will get an error!

```bash

# If you want to train the Bayesian Version
python model_training.py --train data/processed_data/processed_train.csv --val data/processed_data/processed_val.csv --test data/processed_data/processed_test.csv --model xgboost --gridcv True --use_graph_cols True

python evaluate.py --pred_file data/predictions/val_predictions_xgboost.csv --output_path data/results/val_evaluation_xgboost_bayesian.txt
python evaluate.py --pred_file data/predictions/test_predictions_xgboost.csv --output_path data/results/test_evaluation_xgboost_bayesian.txt
```

```bash

# If you want to train on all features
python model_training.py --train data/processed_data/processed_train.csv --val data/processed_data/processed_val.csv --test data/processed_data/processed_test.csv --model xgboost --gridcv True

python evaluate.py --pred_file data/predictions/val_predictions_xgboost.csv --output_path data/results/val_evaluation_xgboost_full.txt
python evaluate.py --pred_file data/predictions/test_predictions_xgboost.csv --output_path data/results/test_evaluation_xgboost_full.txt
```

You should now be able to see the results in ``` data/results/ ```!

For any issues, please mail us!