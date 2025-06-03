import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def evaluate_model(model_path, test_data_path):
    """
    Evaluate model performance with metrics and visualizations.

    Args:
        model_path (str): Path to the trained model (.pkl)
        test_data_path (str): Path to the test dataset (.csv)
    """
    try:
        # Load the model and test data
        model = joblib.load(model_path)
        test_data = pd.read_csv(test_data_path)

        # Required columns
        required_cols = ['Amount', 'Transaction_Type', 'Card_Type', 'Class']
        missing = set(required_cols) - set(test_data.columns)
        if missing:
            raise ValueError(f"Missing columns in test data: {missing}")

        # Validate categorical encodings
        if test_data[['Transaction_Type', 'Card_Type']].isnull().any().any():
            raise ValueError("Invalid values found in 'Transaction_Type' or 'Card_Type'")

        X_test = test_data.drop(columns=['Class'])
        y_test = test_data['Class']

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print("╔" + "═" * 50 + "╗")
        print("║               MODEL EVALUATION                 ║")
        print("╚" + "═" * 50 + "╝\n")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")

        # Create directory for saving plots
        Path("reports/figures").mkdir(parents=True, exist_ok=True)

        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("reports/figures/confusion_matrix.png")
        plt.close()

        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, marker='.')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig("reports/figures/precision_recall_curve.png")
        plt.close()

        # Precision vs Confidence (Thresholds)
        plt.figure(figsize=(6, 5))
        plt.plot(thresholds, precision[:-1], label="Precision")
        plt.title("Precision vs Confidence")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reports/figures/precision_vs_confidence.png")
        plt.close()

        # Recall vs Confidence (Thresholds)
        plt.figure(figsize=(6, 5))
        plt.plot(thresholds, recall[:-1], label="Recall", color="orange")
        plt.title("Recall vs Confidence")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Recall")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reports/figures/recall_vs_confidence.png")
        plt.close()

        print("\n✅ Visualizations saved to: reports/figures/")
        return {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba)
        }

    except Exception as e:
        print(f"❌ Error in evaluation: {str(e)}")

if __name__ == "__main__":
    evaluate_model(
        model_path="models/random_forest.pkl",
        test_data_path="data/processed/processed.csv"
    )
