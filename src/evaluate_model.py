import pandas as pd
import joblib
from sklearn.metrics import (classification_report, 
                           roc_auc_score,
                           confusion_matrix,
                           precision_recall_curve,
                           average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_model(model_path, test_data_path):
    """
    Evaluate model performance with comprehensive metrics and visualizations
    
    Args:
        model_path: Path to trained model (.pkl)
        test_data_path: Path to test dataset (.csv)
    """
    try:
        # Load model and data
        model = joblib.load(model_path)
        test_data = pd.read_csv(test_data_path)
        
        # Validate data structure
        required_cols = ['Amount', 'Class'] + [f'V{i}' for i in range(1,29)]
        if not all(col in test_data.columns for col in required_cols):
            missing = set(required_cols) - set(test_data.columns)
            raise ValueError(f"Missing columns in test data: {missing}")

        X_test = test_data.drop('Class', axis=1)
        y_test = test_data['Class']
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
        # 1. Print metrics
        print("╔" + "═"*50 + "╗")
        print("║               MODEL EVALUATION               ║")
        print("╚" + "═"*50 + "╝\n")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        
        print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
        
        # 2. Generate visualizations
        plt.figure(figsize=(15, 5))
        
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legit', 'Fraud'],
                   yticklabels=['Legit', 'Fraud'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, marker='.')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        
        # Save plots
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig("reports/figures/model_evaluation.png")
        print("\nVisualizations saved to reports/figures/")
        
        # 3. Return metrics for documentation
        return {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba)
        }
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")

if __name__ == "__main__":
    evaluate_model(
        model_path="models/random_forest.pkl",
        test_data_path="data/processed/processed.csv"
    )
