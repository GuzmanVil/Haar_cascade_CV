"""
Comparación de rendimiento entre:

  1) Cascada Haar+AdaBoost propia (haar_cascade.pkl)
  2) Haar cascade de OpenCV (haarcascade_frontalface_default.xml)

Métricas por PATCH (cara / no cara):
  - accuracy
  - precision
  - recall
  - F1
  - tiempo medio de inferencia

Ejemplos de uso:

  python compare_cascades_metrics.py \
      --mat data/annotations.mat \
      --images data/images_root \
      --model haar_cascade.pkl \
      --max-imgs 200 \
      --neg-per-img 2

Ajusta rutas y parámetros según tu proyecto.
"""

import argparse
import time

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Ajusta estos imports a tu estructura real
from src.load_dataset import load_annotations, generate_samples_in_memory
from src.haar_features import compute_haar_features
from src.detect_faces import predict_cascade


# ------------------------------------------------------------
# Predicción con tu cascada propia sobre UN patch  (cara / no cara)
# ------------------------------------------------------------
def predict_custom_patch(patch, cascade, scaler, win_size=24, prob_threshold=None):
    """
    patch: imagen 2D (grayscale) o 3D (BGR) de tamaño cualquiera.
    Devuelve:
        pred_label: 0 o 1
        avg_prob: probabilidad promedio de la cascada
    """
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch

    gray_resized = cv2.resize(gray, (win_size, win_size)).astype(np.float64) / 255.0
    ii = cv2.integral(gray_resized)[1:, 1:]

    feat = compute_haar_features(ii, win_size=win_size, step=4).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    passed, avg_prob = predict_cascade(feat_scaled, cascade)

    if prob_threshold is not None:
        pred_label = int(passed and avg_prob >= prob_threshold)
    else:
        pred_label = int(passed)

    return pred_label, float(avg_prob)


# ------------------------------------------------------------
# Predicción con el Haar cascade de OpenCV sobre UN patch
# ------------------------------------------------------------
def predict_opencv_patch(patch, cv_cascade, upscale=2.0):
    """
    patch: imagen 2D (grayscale) o 3D (BGR) con (cara o no).
    La decisión es:
        - 1 si detecta al menos una cara
        - 0 en otro caso
    """
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch

    if upscale != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_LINEAR,
        )

    faces = cv_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    return 1 if len(faces) > 0 else 0


# ------------------------------------------------------------
# Carga de modelos
# ------------------------------------------------------------
def load_custom_cascade(model_path="haar_cascade.pkl"):
    data = joblib.load(model_path)
    cascade = data["cascade"]
    scaler = data["scaler"]
    win_size = data.get("win_size", 24)
    if isinstance(win_size, tuple):
        win_size = win_size[0]
    return cascade, scaler, win_size


def load_opencv_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"No se pudo cargar el cascade de OpenCV desde: {cascade_path}")
    return face_cascade


# ------------------------------------------------------------
# Evaluación en patches positivos/negativos
# ------------------------------------------------------------
def evaluate_cascades_on_patches(
    annotations_mat,
    images_root,
    model_path,
    img_size=24,
    neg_per_img=1,
    max_imgs=None,
    custom_prob_threshold=None,
    cv_upscale=2.0,
):
    print("[INFO] Cargando anotaciones...")
    annotations = load_annotations(annotations_mat, images_root)

    print("[INFO] Generando patches positivos y negativos...")
    positives, negatives = generate_samples_in_memory(
        annotations,
        img_size=(img_size, img_size),
        neg_per_img=neg_per_img,
        max_imgs=max_imgs,
    )

    y_pos = np.ones(len(positives), dtype=int)
    y_neg = np.zeros(len(negatives), dtype=int)

    X = np.concatenate([positives, negatives], axis=0)
    y_true = np.concatenate([y_pos, y_neg], axis=0)

    print(f"[INFO] Total patches: {len(X)} (positivos={len(positives)}, negativos={len(negatives)})")

    # Mezclar para que positivos/negativos queden mezclados
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X = X[idx]
    y_true = y_true[idx]

    print("[INFO] Cargando cascada propia...")
    cascade, scaler, win_size = load_custom_cascade(model_path)

    print("[INFO] Cargando Haar cascade de OpenCV...")
    cv_cascade = load_opencv_cascade()

    y_pred_custom = []
    y_pred_cv = []
    t_custom_total = 0.0
    t_cv_total = 0.0

    for i, patch in enumerate(X):
        # --- Modelo propio ---
        t0 = time.perf_counter()
        pred_c, avg_prob = predict_custom_patch(
            patch,
            cascade,
            scaler,
            win_size=win_size,
            prob_threshold=custom_prob_threshold,
        )
        t_custom_total += (time.perf_counter() - t0)

        # --- OpenCV ---
        t0 = time.perf_counter()
        pred_cv = predict_opencv_patch(patch, cv_cascade, upscale=cv_upscale)
        t_cv_total += (time.perf_counter() - t0)

        y_pred_custom.append(pred_c)
        y_pred_cv.append(pred_cv)

    y_pred_custom = np.array(y_pred_custom)
    y_pred_cv = np.array(y_pred_cv)

    n = len(X)
    ms_custom = (t_custom_total / n) * 1000.0
    ms_cv = (t_cv_total / n) * 1000.0

    print("\n================= RESULTADOS (por patch) =================")
    print(f"Total muestras evaluadas: {n}")

    # --- Métricas modelo propio ---
    print("\n--- Cascada propia (Haar + AdaBoost) ---")
    print(f"Tiempo medio por patch: {ms_custom:.3f} ms")
    print(f"Accuracy : {accuracy_score(y_true, y_pred_custom):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_custom, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred_custom, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred_custom, zero_division=0):.4f}")
    print(f"CM: {confusion_matrix(y_true, y_pred_custom)}")
    print("\nReporte de clasificación (modelo propio):")
    print(classification_report(y_true, y_pred_custom, digits=4, zero_division=0))

    # --- Métricas OpenCV ---
    print("\n--- Haar cascade OpenCV ---")
    print(f"Tiempo medio por patch: {ms_cv:.3f} ms")
    print(f"Accuracy : {accuracy_score(y_true, y_pred_cv):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_cv, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred_cv, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred_cv, zero_division=0):.4f}")
    print(f"CM: {confusion_matrix(y_true, y_pred_cv)}")
    print("\nReporte de clasificación (OpenCV):")
    print(classification_report(y_true, y_pred_cv, digits=4, zero_division=0))


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar cascada propia vs Haar cascade de OpenCV en patches (cara/no cara)."
    )
    parser.add_argument("--mat", required=True, help="Ruta al archivo .mat de anotaciones")
    parser.add_argument("--images", required=True, help="Ruta base de las imágenes originales")
    parser.add_argument("--model", default="haar_cascade.pkl", help="Ruta al modelo propio (joblib)")
    parser.add_argument("--win-size", type=int, default=24, help="Tamaño de ventana (lado) usado en el entrenamiento")
    parser.add_argument("--neg-per-img", type=int, default=1, help="Número de negativos por imagen al generar patches")
    parser.add_argument("--max-imgs", type=int, default=None, help="Máximo de imágenes a procesar para generar patches")
    parser.add_argument(
        "--custom-threshold",
        type=float,
        default=None,
        help="Umbral global opcional sobre avg_prob de la cascada propia (si None, solo usa passed).",
    )
    parser.add_argument(
        "--cv-upscale",
        type=float,
        default=2.0,
        help="Factor para reescalar el patch antes de pasarlo al Haar de OpenCV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_cascades_on_patches(
        annotations_mat=args.mat,
        images_root=args.images,
        model_path=args.model,
        img_size=args.win_size,
        neg_per_img=args.neg_per_img,
        max_imgs=args.max_imgs,
        custom_prob_threshold=args.custom_threshold,
        cv_upscale=args.cv_upscale,
    )
