# src/adaboost.py
import numpy as np
from src.weak_classifier import WeakClassifier

class AdaBoost:
    def __init__(self, T=10):
        self.T = T
        self.classifiers = []

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        X: matriz de características (n_samples, n_features)
        y: etiquetas (+1 cara, -1 no cara)
        """
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples  # pesos iniciales

        for t in range(self.T):
            w /= np.sum(w)  # normalizar pesos

            # Buscar el mejor clasificador débil
            min_error = float('inf')
            best_clf = None

            for j in range(n_features):
                feature_values = X[:, j]
                thresholds = np.unique(feature_values)

                for thr in thresholds[::max(1, len(thresholds)//10)]:  # reducir cómputo
                    for polarity in [1, -1]:
                        preds = np.ones(n_samples)
                        preds[polarity * feature_values < polarity * thr] = -1
                        err = np.sum(w * (preds != y))

                        if err < min_error:
                            min_error = err
                            best_clf = WeakClassifier(j, thr, polarity, 0.0)

            # Calcular el peso (alpha)
            eps = 1e-10
            alpha = 0.5 * np.log((1 - min_error + eps) / (min_error + eps))
            best_clf.alpha = alpha
            self.classifiers.append(best_clf)

            # Actualizar pesos
            preds = best_clf.predict(X[:, best_clf.feature_idx])
            w *= np.exp(-alpha * y * preds)

            print(f"[Iter {t+1}] Error: {min_error:.4f}, Alpha: {alpha:.3f}, Feature: {best_clf.feature_idx}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Clasifica cada muestra usando el clasificador fuerte.
        """
        final_score = np.zeros(X.shape[0])
        for clf in self.classifiers:
            preds = clf.predict(X[:, clf.feature_idx])
            final_score += clf.alpha * preds
        return np.sign(final_score)
