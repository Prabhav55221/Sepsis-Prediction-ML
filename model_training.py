import argparse
import xgboost
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.inference import VariableElimination
import logging

# Set up logging
logging.basicConfig(
    filename="./logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Initializing Training class.")

# Class for Bayesian Network
class GraphicalModel:
    def __init__(self):
        self.model = None
        self.inference = None

    def preprocess_data(self, data, num_bins=10):
        """
        Preprocesses the data dynamically by discretizing continuous variables and handling missing values.

        Parameters:
        - data: DataFrame to preprocess.
        - num_bins: Number of bins for discretization.

        Returns:
        - processed_data: DataFrame with preprocessed values.
        """
        logging.info("Preprocessing data for Bayesian Network.")
        data.drop(columns=["PatientID"], inplace=True)

        # Discretize continuous variables dynamically
        continuous_columns = data.select_dtypes(include=[np.float64, np.int64]).columns
        discretized_data = data.copy()

        for col in continuous_columns:
            if col != "sepsis_label":  # Avoid discretizing the target variable
                min_val = data[col].min()
                max_val = data[col].max()
                bin_edges = np.linspace(min_val, max_val, num_bins + 1)
                discretized_data[col] = pd.cut(data[col], bins=bin_edges, labels=False, include_lowest=True)

        logging.info("Preprocessing complete.")
        return discretized_data

    def fit(self, data, target="sepsis_label", scoring_method="bicscore"):
        """
        Fits a Bayesian Network to the data.

        Parameters:
        - data: Preprocessed DataFrame containing features and the target variable.
        - target: The target variable to predict.
        - scoring_method: The scoring method for HillClimbSearch.
        """
        logging.info("Learning structure using HillClimbSearch.")
        hc = HillClimbSearch(data)
        structure = hc.estimate(scoring_method=scoring_method)
        self.model = BayesianNetwork(structure.edges())
        logging.info(f"Learned Structure: {structure.edges()}")

        logging.info("Fitting Bayesian Network.")
        self.model.fit(data)
        self.inference = VariableElimination(self.model)
        logging.info("Model training completed.")

    def filter_data_for_model(self, data):
        """
        Filters the input data to ensure it contains only the variables present in the model.

        Parameters:
        - data: DataFrame containing features to predict.

        Returns:
        - filtered_data: DataFrame containing only the variables in the model.
        """
        model_vars = set(self.model.nodes())
        data_vars = set(data.columns)
        common_vars = model_vars.intersection(data_vars)

        if not common_vars:
            raise ValueError("No variables in the data match the model.")

        filtered_data = data[list(common_vars)]
        return filtered_data

    def predict(self, data):
        """
        Predicts the target variable using the Bayesian Network.

        Parameters:
        - data: DataFrame containing features to predict.

        Returns:
        - predictions: Predicted labels for the target variable.
        """
        if not self.inference:
            raise RuntimeError("Model is not trained yet. Please call `fit` first.")

        logging.info("Predicting using Bayesian Network.")
        data = self.filter_data_for_model(data)
        return self.model.predict(data)


# Class for XGBoost Model
class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=3):
        self.model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        logging.info("Training XGBoost model.")
        self.model.fit(X, y)
        logging.info("XGBoost model training complete.")

    def predict(self, X):
        logging.info("Predicting using XGBoost model.")
        return self.model.predict(X)


def main():
    
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--val", type=str, required=True, help="Path to the validation data CSV file.")
    parser.add_argument("--test", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--model", type=str, required=True, choices=["bayesian", "xgboost"], help="Model type.")
    parser.add_argument("--output_dir", type=str, default="data/models/", help="Directory to save models and predictions.")
    parser.add_argument("--pred_dir", type=str, default="data/predictions/", help="Directory to save predictions.")
    parser.add_argument("--scoring_method", type=str, default="bicscore", help="Scoring method for Bayesian Network.")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for discretization (Bayesian Network).")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for XGBoost.")
    parser.add_argument("--max_depth", type=int, default=3, help="Max depth for XGBoost.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # Load data
    logging.info("Loading data files.")
    train_data = pd.read_csv(args.train)
    val_data = pd.read_csv(args.val)
    test_data = pd.read_csv(args.test)

    if args.model == "bayesian":
        gm = GraphicalModel()

        # Preprocess data
        logging.info("Preprocessing data for Bayesian Network.")
        train_data = gm.preprocess_data(train_data, num_bins=args.num_bins)
        val_data = gm.preprocess_data(val_data, num_bins=args.num_bins)
        test_data = gm.preprocess_data(test_data, num_bins=args.num_bins)

        # Train model
        gm.fit(train_data, scoring_method=args.scoring_method)

        # Save model
        model_path = os.path.join(args.output_dir, "bayesian_model.pkl")
        joblib.dump(gm, model_path)
        logging.info(f"Bayesian Network model saved at {model_path}.")

        # Predict
        val_features = val_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")
        test_features = test_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")

        val_predictions = gm.predict(val_features)
        test_predictions = gm.predict(test_features)

    elif args.model == "xgboost":
        xgb_model = XGBoostModel(n_estimators=args.n_estimators, max_depth=args.max_depth)

        # Prepare data
        logging.info("Preparing data for XGBoost model.")
        X_train = train_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")
        y_train = train_data["sepsis_label"]
        X_val = val_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")
        y_val = val_data["sepsis_label"]
        X_test = test_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")

        # Train model
        xgb_model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(args.output_dir, "xgboost_model.pkl")
        joblib.dump(xgb_model, model_path)
        logging.info(f"XGBoost model saved at {model_path}.")

        # Predict
        val_predictions = xgb_model.predict(X_val)
        test_predictions = xgb_model.predict(X_test)

    # Save predictions
    logging.info("Saving predictions.")
    val_results_path = os.path.join(args.pred_dir, f"val_predictions_{args.model}.csv")
    test_results_path = os.path.join(args.pred_dir, f"test_predictions_{args.model}.csv")
    val_data["Predicted"] = val_predictions
    test_data["Predicted"] = test_predictions
    val_data.to_csv(val_results_path, index=False)
    test_data.to_csv(test_results_path, index=False)
    logging.info(f"Validation predictions saved to {val_results_path}.")
    logging.info(f"Test predictions saved to {test_results_path}.")


if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(
        filename="./logs/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Initializing Training class.")

    main()
