import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import argparse
import logging

logging.basicConfig(
    filename="./logs/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class Evaluator:
    def __init__(self, pred_file, output_path):
        """
        Initializes the evaluator.

        Parameters:
        - pred_file: Path to the prediction file (CSV format).
        - output_path: Path to save the evaluation results.
        """
        self.pred_file = pred_file
        self.output_path = output_path
        self.data = None

        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        logging.info(f"Loaded prediction file: {pred_file}")

    def load_data(self):
        """
        Loads the prediction file.
        """
        self.data = pd.read_csv(self.pred_file)
        if "sepsis_label" not in self.data.columns or "Predicted" not in self.data.columns:
            raise ValueError("Prediction file must contain 'sepsis_label' and 'Predicted' columns.")
        logging.info("Prediction data loaded successfully.")

    def evaluate(self):
        """
        Evaluates the predictions and computes metrics.

        Returns:
        - metrics: Dictionary containing evaluation metrics.
        """
        y_true = self.data["sepsis_label"]
        y_pred = self.data["Predicted"]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=1),
            "recall": recall_score(y_true, y_pred, zero_division=1),
            "f1_score": f1_score(y_true, y_pred, zero_division=1),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, zero_division=1),
        }
        logging.info("Evaluation metrics computed successfully.")
        return metrics

    def save_results(self, metrics):
        """
        Saves the evaluation results to the output path.

        Parameters:
        - metrics: Dictionary containing evaluation metrics.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Save metrics to a text file
        with open(self.output_path, "w") as f:
            f.write("Evaluation Metrics\n")
            f.write("==================\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(metrics["confusion_matrix"]) + "\n\n")
            f.write("Classification Report:\n")
            f.write(metrics["classification_report"] + "\n")

        logging.info(f"Evaluation results saved to: {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction files.")
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="Path to the prediction file (CSV format with columns 'sepsis_label' and 'Predicted').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the evaluation results.",
    )

    args = parser.parse_args()

    try:
        evaluator = Evaluator(pred_file=args.pred_file, output_path=args.output_path)
        evaluator.load_data()
        metrics = evaluator.evaluate()
        evaluator.save_results(metrics)
        print(f"Evaluation completed. Results saved to {args.output_path}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")
