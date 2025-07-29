import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import numpy as np
import warnings

class ModelEvaluator:
    @staticmethod
    def evaluate(y_test, y_pred, y_proba=None):
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Plot Confusion Matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        # ROC Curve and AUC
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                fpr, tpr, _ = roc_curve(y_test, y_proba)

                print(f"\n🔵 ROC-AUC Score: {auc:.2f}")

                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.title("ROC Curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()
                plt.show()
            except ValueError as e:
                warnings.warn(f"ROC Curve skipped: {str(e)}")

        # Class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        plt.figure(figsize=(5, 4))
        sns.barplot(x=unique, y=counts)
        plt.title("Class Distribution in Test Set")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
