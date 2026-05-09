import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

class HallucinationProbe(nn.Module):
    """
    Sklearn's LogisticRegression with optional PCA.
    """
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.pca = None
        self.clf = None
        self.threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300) -> "HallucinationProbe":
        """
        Train the logistic regression model.
        """
        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Reduce dimensionality with PCA to 64 components
        n_components = min(64, X_scaled.shape[1], X_scaled.shape[0] - 1)
        if n_components < X_scaled.shape[1]:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_reduced = self.pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        # Train logistic regression
        self.clf = LogisticRegression(
            C=0.05,                 # strong L2 regularization
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        self.clf.fit(X_reduced, y)
        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: dict = None,
    ) -> "HallucinationProbe":
        """
        Tune decision threshold on validation set to maximize F1
        """
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_th = 0.5
        best_f1 = -1.0
        for t in candidates:
            pred = (probs >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(t)
        self.threshold = best_th
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_reduced = self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        return self.clf.predict_proba(X_reduced)