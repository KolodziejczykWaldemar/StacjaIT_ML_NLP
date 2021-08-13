from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_prec_rec(y: np.ndarray, 
                  y_pred_positive: np.ndarray, 
                  label: str) -> None:
    """
    Function outputs plot of Precision-Recall curve for classificaiton results.
    Args:
        y: (numpy.ndarray) array of true (binary) outputs
        y_pred_positive: (numpy.ndarray) array of predicted (float) outputs
        label: (string) label of the curve
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_pred_positive)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, "b:", linewidth=2, label=label)
    plt.xlabel('Czułość', fontsize=16)
    plt.ylabel('Precyzja', fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.title('Krzywa czułość-precyzja', fontsize=18)
    plt.show()

def plot_roc_curve(y: np.ndarray, 
                   y_pred_positive: np.ndarray, 
                   label: str) -> None:
    """
    Function outputs plot of ROC curve for classificaiton results.
    Args:
        y: (numpy.ndarray) array of true (binary) outputs
        y_pred_positive: (numpy.ndarray) array of predicted (float) outputs
        label: (string) label of the curve
    """
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_positive)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b:", linewidth=2, label=label)
    plt.fill_between(fpr, tpr, color='blue', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Odsetek fałszywie pozytywnych (FPR)', fontsize=16)
    plt.ylabel('Odsetek prawdziwie pozytywnych (TPR)', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.title('Krzywa ROC, AUC={0:.3f}'.format(metrics.roc_auc_score(y, y_pred_positive)), fontsize=18)
    plt.show()

