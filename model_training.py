import argparse
import xgboost
import os
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.inference import VariableElimination
import logging

# DEFINE STATES
states = {
 'Age': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'BUN': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'BUN_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'BUN_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Calcium': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Calcium_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Calcium_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Chloride': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Chloride_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Chloride_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'DBP': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'DBP_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'DBP_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'FiO2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'FiO2_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'FiO2_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Gender': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Glucose': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Glucose_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Glucose_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'HR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'HR_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'HR_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Hct': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Hct_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Hct_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'ICULOS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'MAP': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'MAP_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'MAP_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Magnesium': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Magnesium_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Magnesium_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'O2Sat': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'O2Sat_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'O2Sat_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Resp': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Resp_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Resp_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'SBP': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'SBP_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'SBP_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Temp': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Temp_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Temp_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Unit1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'Unit2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'WBC': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'WBC_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'WBC_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'pH': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'pH_FINAL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'pH_median': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'sepsis_label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# DEFINE FINAL TRAINING DATA
columns_to_use = []

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
    
    def save_graph(self, output_dir="./data/visualizations", filename="graphical_model.png"):
        """
        Saves the Bayesian Network structure as a PNG file.

        Parameters:
        - output_dir: Directory to save the graph.
        - filename: Name of the PNG file.
        """
        if not self.model:
            raise RuntimeError("Model has not been trained. Train the model before saving the graph.")

        logging.info("Saving graphical model structure.")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Convert to a networkx graph and draw
        nx_graph = nx.DiGraph(self.model.edges())
        plt.figure(figsize=(24, 12))
        nx.draw(nx_graph, with_labels=True, node_size=5000, node_color="lightblue", font_size=8, font_weight="bold")
        plt.savefig(output_path, format="png")
        plt.close()
        logging.info(f"Graphical model saved at {output_path}.")

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
        self.model.fit(data, state_names=states)
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

        columns_to_use = list(common_vars)

        with open('./data/results/graph_nodes.txt', 'w') as f: 
            f.write(str(columns_to_use))

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
    
# Class for XGBoost Model with Hyperparameter Tuning
class XGBoostModelGrid:
    def __init__(self, param_grid=None, cv=5, scoring="accuracy", verbose=1):
        """
        Initializes the XGBoost model with optional hyperparameter tuning.

        Parameters:
        - param_grid: Dictionary containing hyperparameter search space for GridSearchCV.
        - cv: Number of cross-validation folds.
        - scoring: Scoring metric for hyperparameter tuning.
        - verbose: Verbosity level for GridSearchCV.
        """
        self.param_grid = param_grid or {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 7],
            'learning_rate': [0.01, 0.2]
        }
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.best_params = None
        self.model = None

    def fit(self, X, y):
        """
        Fits the XGBoost model with hyperparameter tuning.

        Parameters:
        - X: Feature matrix for training.
        - y: Target labels for training.
        """
        logging.info("Initializing GridSearchCV for hyperparameter tuning.")
        base_model = XGBClassifier(eval_metric="logloss")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            verbose=self.verbose
        )
        logging.info("Starting GridSearchCV...")
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        logging.info(f"Best parameters found: {self.best_params}")
        logging.info("XGBoost model training complete with best parameters.")

    def predict(self, X):
        """
        Predicts using the trained XGBoost model.

        Parameters:
        - X: Feature matrix for prediction.

        Returns:
        - Predicted labels for the input data.
        """
        if not self.model:
            raise RuntimeError("The model is not trained. Please call `fit` first.")
        logging.info("Predicting using the best XGBoost model.")
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
    parser.add_argument("--use_graph_cols", type=bool, default=False, help="Should the model use feature selection.")
    parser.add_argument("--gridcv", type=bool, default=False, help="Do Tuning")

    args = parser.parse_args()

    os.makedirs('./data/results')

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
        gm.save_graph()

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

        if args.gridcv:
            xgb_model = XGBoostModelGrid()
        else:
            xgb_model = XGBoostModel(n_estimators=args.n_estimators, max_depth=args.max_depth)

        # Prepare data
        logging.info("Preparing data for XGBoost model.")
        X_train = train_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")
        y_train = train_data["sepsis_label"]
        X_val = val_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")
        y_val = val_data["sepsis_label"]
        X_test = test_data.drop(columns=["sepsis_label", "PatientID"], errors="ignore")

        if args.use_graph_cols:
            # Read the file and process the variable list
            with open('./data/results/graph_nodes.txt', "r") as file:
                content = file.read()

            # Remove unnecessary characters and split into a list
            variables = content.strip("[]").replace("'", "").split(", ")

            X_train = X_train[variables]
            X_test = X_test[variables]
            X_val = X_val[variables]

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
