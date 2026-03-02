import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyMetrics:
  def __init__(self, test_results, threshold):
    self.test_results = test_results
    self.threshold = threshold
    self.metrics = {}

    self.y_true = np.concatenate([
      np.zeros(len(self.test_results['normal'])),   # 정상 = 0
      np.ones(len(self.test_results['abnormal']))    # 이상 = 1
    ])
    self.y_scores = np.concatenate([self.test_results['normal'], self.test_results['abnormal']])
    self.y_pred = (self.y_scores > self.threshold).astype(int)

  def calculate_base_metrics(self):
    self.metrics = {
      'ROC': roc_curve(self.y_true, self.y_scores),
      'AUC': roc_auc_score(self.y_true, self.y_scores),
      'Accuracy': accuracy_score(self.y_true, self.y_pred),
      'Precision': precision_score(self.y_true, self.y_pred, zero_division=0),
      'Recall': recall_score(self.y_true, self.y_pred, zero_division=0),
      'F1': f1_score(self.y_true, self.y_pred, zero_division=0),
    }
    return self.metrics
  
  def plot_metrics(self, save_dir):
    self.calculate_base_metrics()
    self._plot_confusion_matrix(save_dir)
    self._plot_roc_curve(save_dir)
  
  def _plot_roc_curve(self, save_dir):
    if self.metrics['AUC'] and self.metrics['ROC']:
      fpr, tpr, _ = self.metrics['ROC']
      auc = self.metrics['AUC']

      plt.figure(figsize=(8, 8))
      plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
      plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
      plt.xlabel('False Positive Rate', fontsize=12)
      plt.ylabel('True Positive Rate', fontsize=12)
      plt.title('ROC Curve', fontsize=14, fontweight='bold')
      plt.legend()
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
      plt.savefig(f'{save_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
      plt.close()

  def _plot_confusion_matrix(self, save_dir):
    cm = confusion_matrix(self.y_true, self.y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()