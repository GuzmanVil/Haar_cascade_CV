# main.py
import os
import random
import numpy as np
import cv2

from src.data_loader import load_annotations, load_image
from src.integral_image import compute_integral_image
from src.haar_features import generate_haar_features, evaluate_feature_on_window
from src.adaboost import AdaBoost


def extract_window(img, bbox, size=(24, 24)):
    """Recorta y normaliza una ventana 24x24 centrada en una cara."""
    x, y, w, h = bbox
    face = img[int(y):int(y+h), int(x):int(x+w)]
    if face.size == 0:
        return None
    return cv2.resize(face, size)


def create_training_samples(annotations, n_pos=20, n_neg=20):
    """
    Crea ejemplos de entrenamiento: ventanas de caras (positivas)
    y ventanas aleatorias del fondo (negativas).
    """
    pos_samples, neg_samples = [], []

    img_paths = list(annotations.keys())
    random.shuffle(img_paths)

    for path in img_paths[:n_pos]:
        img = load_image(path)
        for bbox in annotations[path]:
            for (x, y, w, h) in bbox:
                face = extract_window(img, (x, y, w, h))
                if face is not None:
                    pos_samples.append(face)
                    break  # usar solo una cara por imagen
            if len(pos_samples) >= n_pos:
                break

    # Ventanas negativas (fondo aleatorio)
    for path in img_paths[:n_neg]:
        img = load_image(path)
        H, W = img.shape
        for _ in range(3):
            x = random.randint(0, W - 24)
            y = random.randint(0, H - 24)
            window = img[y:y+24, x:x+24]
            if window.shape == (24, 24):
                neg_samples.append(window)
            if len(neg_samples) >= n_neg:
                break
        if len(neg_samples) >= n_neg:
            break

    print(f"Ventanas positivas: {len(pos_samples)}, negativas: {len(neg_samples)}")
    return pos_samples, neg_samples


def compute_features_for_samples(samples, features):
    """
    Calcula los valores de todas las features Haar para cada ventana.
    Retorna una matriz X de (n_samples x n_features).
    """
    X = np.zeros((len(samples), len(features)), dtype=np.float32)
    for i, sample in enumerate(samples):
        ii = compute_integral_image(sample)
        for j, feat in enumerate(features):
            X[i, j] = evaluate_feature_on_window(ii, feat)
    return X


if __name__ == "__main__":
    # --- Cargar dataset ---
    train_mat = "dataset/train/wider_face_train.mat"
    train_images = "dataset/train/images"
    annotations = load_annotations(train_mat, train_images)
    print(f"Total de imágenes con anotaciones (train): {len(annotations)}")

    # --- Crear muestras de entrenamiento ---
    pos_samples, neg_samples = create_training_samples(annotations, n_pos=200, n_neg=200)
    samples = pos_samples + neg_samples
    labels = np.array([1]*len(pos_samples) + [-1]*len(neg_samples))

    # --- Generar features Haar ---
    features = generate_haar_features(window_size=(24, 24))
    print(f"Total de features generadas: {len(features)}")

    # --- Calcular valores de features ---
    X = compute_features_for_samples(samples, features[:500])  # 500 features para test rápido
    print(f"Matriz X: {X.shape}, etiquetas: {labels.shape}")

    # --- Entrenar AdaBoost ---
    model = AdaBoost(T=10)
    model.train(X, labels)

    # --- Evaluar ---
    preds = model.predict(X)
    acc = np.mean(preds == labels)
    print(f"\nPrecisión de entrenamiento (train): {acc*100:.2f}%")

