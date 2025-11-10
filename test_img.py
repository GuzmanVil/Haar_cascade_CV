import cv2
import joblib
from src.detect_faces import detect_faces, non_max_suppression

# === 1. Cargar modelo entrenado ===
data = joblib.load("haar_adaboost.pkl")
model = data["model"]
scaler = data["scaler"]
window_size = data["win_size"]

# === 2. Cargar imagen a testear ===
# Cambiá esta ruta por la imagen que quieras probar
image_path = "dataset/test/images/0--Parade/0_Parade_marchingband_1_289.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

# === 3. Detectar caras ===
print("[INFO] Detectando caras...")
boxes = detect_faces(img, model, scaler, win_size=window_size[0], step=10, prob_threshold=0.7, scale_factor=1.2, max_scales=6)
boxes = non_max_suppression(boxes, overlapThresh=0.4)
print(f"[OK] {len(boxes)} detecciones encontradas")

# === 4. Dibujar resultados ===
for (x, y, w, h, p) in boxes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("Detecciones", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
