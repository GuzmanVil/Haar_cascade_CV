# src/weak_classifier.py
import numpy as np
from dataclasses import dataclass

@dataclass
class WeakClassifier:
    feature_idx: int
    threshold: float
    polarity: int
    alpha: float  # peso asignado por AdaBoost

    def predict(self, feature_values: np.ndarray) -> np.ndarray:
        """
        feature_values: array con los valores de la feature para cada muestra.
        Retorna predicciones (+1 o -1).
        """
        predictions = np.ones_like(feature_values)
        predictions[self.polarity * feature_values < self.polarity * self.threshold] = -1
        return predictions
