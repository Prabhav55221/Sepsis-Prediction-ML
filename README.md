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

## Setup For Replication

#### Environment Setup

To replicate the codebase, first clone the repository to your local. Ensure you have Conda/Miniconda or some kind of environment handling enabled in your local environment. We do provide an environment.yml file for Conda:

**It will take time to clone the repo since we include the data as a part of the repo!**

```bash
git clone 
cd Sepsis-Prediction-ML
conda env create -f environment.yml
```

#### Data

We provide the data required for this project as part of this Github Repo. You can access the original data, if required, by running the below command. We however, store the data in the repo in the directory ```./data/physionet.org```.

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
```

Once you have cloned the repo, please run the first script to preprocess the data and create the train, test and validation data in JSON format. Refer to our paper to understand the data storage format.

```bash
conda activate ml_project
mkdir data/preprocessed_data
mkdir logs
python preprocess.py
```
