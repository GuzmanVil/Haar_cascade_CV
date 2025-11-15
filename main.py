from src.load_dataset import load_annotations, generate_samples_in_memory
from src.haar_features import extract_features
from src.train_model import train_cascade
import numpy as np
import os

# === 1. Cargar dataset ===
mat_path = "dataset/train/wider_face_train.mat"
images_base_path = "dataset/train/images"
annotations = load_annotations(mat_path, images_base_path)

val_mat_path = "dataset/val/wider_face_val.mat"
val_images_base_path = "dataset/val/images"
val_annotations = load_annotations(val_mat_path, val_images_base_path)

# Config base
win_size = (24, 24)
NEG_PER_IMG = 15               # ¡mantené este valor sincronizado!

# === Helpers de caché ===
os.makedirs("cache", exist_ok=True)

def ensure_samples(ann, pos_fp, neg_fp, img_size, neg_per_img):
    """Carga SAMPLES (parches imgs) si existen; si no, los genera, guarda y retorna."""
    if os.path.exists(pos_fp) and os.path.exists(neg_fp):
        print(f"[INFO] Cargando samples desde caché: {pos_fp} / {neg_fp}")
        pos = np.load(pos_fp)
        neg = np.load(neg_fp)
        return pos, neg
    print("[INFO] Generando samples en memoria...")
    pos, neg = generate_samples_in_memory(ann, img_size=img_size, neg_per_img=neg_per_img)
    # guardamos como uint8 (parches en escala de grises 0..255)
    np.save(pos_fp, pos.astype(np.uint8))
    np.save(neg_fp, neg.astype(np.uint8))
    print(f"[OK] Samples cacheados: {pos_fp} / {neg_fp}")
    return pos, neg

def ensure_feats(images, path, win):
    """Carga FEATURES si existen; si no, las calcula, guarda y retorna."""
    if os.path.exists(path):
        print(f"[INFO] Cargando características desde caché: {path}")
        return np.load(path)
    print("[INFO] Extrayendo características...")
    feats = extract_features(images, win_size=win[0]).astype(np.float32)  # tu extract_features espera int
    np.save(path, feats)
    print(f"[OK] Features cacheadas: {path}")
    return feats

# === 2. Generar (o cargar) SAMPLES ===
train_pos_imgs_fp = f"cache/train_pos_imgs_neg.npy"
train_neg_imgs_fp = f"cache/train_neg_imgs_neg.npy"
val_pos_imgs_fp   = f"cache/val_pos_imgs_neg.npy"
val_neg_imgs_fp   = f"cache/val_neg_imgs_neg.npy"

positives, negatives = ensure_samples(
    annotations, train_pos_imgs_fp, train_neg_imgs_fp, img_size=win_size, neg_per_img=NEG_PER_IMG
)
test_pos, test_neg = ensure_samples(
    val_annotations, val_pos_imgs_fp, val_neg_imgs_fp, img_size=win_size, neg_per_img=NEG_PER_IMG
)

# === 3. Extraer (o cargar) FEATURES ===
train_pos_fp = f"cache/train_pos_feats.npy"
train_neg_fp = f"cache/train_neg_feats.npy"
val_pos_fp   = f"cache/val_pos_feats.npy"
val_neg_fp   = f"cache/val_neg_feats.npy"

X_pos       = ensure_feats(positives, train_pos_fp, win_size)
X_neg       = ensure_feats(negatives, train_neg_fp, win_size)
X_test_pos  = ensure_feats(test_pos,  val_pos_fp,   win_size)
X_test_neg  = ensure_feats(test_neg,  val_neg_fp,   win_size)

# === 4. Armar datasets y mezclar ===
X_train = np.vstack([X_pos, X_neg])
y_train = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=np.int8)

X_test  = np.vstack([X_test_pos, X_test_neg])
y_test  = np.array([1]*len(X_test_pos) + [0]*len(X_test_neg), dtype=np.int8)

# Mezcla (shuffle) para no dejar clases agrupadas
perm = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm], y_train[perm]

# === 5. Entrenar modelo ===
print("[INFO] Entrenando modelo...")
cascada, scaler = train_cascade(
    X_train, y_train, X_test, y_test,
    num_stages=4,
    save_path="haar_cascade_extra_filter.pkl"
)
