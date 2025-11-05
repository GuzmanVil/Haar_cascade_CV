from src.load_dataset import load_annotations, generate_samples_in_memory
from src.haar_features import extract_features
from src.train_model import train_model
import numpy as np

# === 1. Cargar dataset ===
mat_path = "dataset/train/wider_face_train.mat"
images_base_path = "dataset/train/images"
annotations = load_annotations(mat_path, images_base_path)

val_mat_path = "dataset/val/wider_face_val.mat"
val_images_base_path = "dataset/val/images"
val_annotations = load_annotations(val_mat_path, val_images_base_path)

win_size = (24, 24)

# === 2. Generar muestras ===

positives, negatives = generate_samples_in_memory(
    annotations,
    img_size=win_size,
    neg_per_img=10  # limitá según tu RAM
)

print("Positivos:", positives.shape, "Negativos:", negatives.shape)

test_pos, test_neg = generate_samples_in_memory(
    val_annotations,
    img_size=win_size,
    neg_per_img=10  # limitá según tu RAM
)

# === 3. Extraer features ===
print("[INFO] Extrayendo características...")
X_pos = extract_features(positives)
print("Shape de X_pos:", X_pos.shape)
X_neg = extract_features(negatives)
X_train = np.vstack([X_pos, X_neg])
y_train = np.array([1]*len(X_pos) + [0]*len(X_neg))

X_test_pos = extract_features(test_pos)
X_test_neg = extract_features(test_neg)
X_test = np.vstack([X_test_pos, X_test_neg])
y_test = np.array([1]*len(X_test_pos) + [0]*len(X_test_neg))


# === 4. Entrenar modelo ===
print("[INFO] Entrenando modelo...")
model, scaler = train_model(X_train, y_train, X_test, y_test, win_size=win_size, save_path="haar_adaboost.pkl")
