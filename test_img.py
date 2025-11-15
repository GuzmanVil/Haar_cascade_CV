import os
import random
import cv2
import joblib
from src.detect_faces import detect_faces, non_max_suppression

# === 1. Cargar modelo entrenado ===
data = joblib.load("haar_cascade_bigger_threshold.pkl")
cascade = data["cascade"]
scaler = data["scaler"]

# Manejo de win_size que puede venir como tuple (24, 24)
win_size = data.get("win_size", 24)
if isinstance(win_size, tuple):
    win_size = win_size[0]

print(f"[INFO] Tamaño de ventana: {win_size}x{win_size}")
print(f"[INFO] Etapas en cascada: {len(cascade)}")

# === 2. Cargar imagen a testear ===
image_path = "dataset/test/images/1--Handshaking/1_Handshaking_Handshaking_1_177.jpg"  # Cambiar por la ruta de tu imagen
print(f"[INFO] Procesando imagen: {image_path}")
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

# === 3. Detectar caras ===
print("[INFO] Detectando caras...")
boxes = detect_faces(
    img, cascade, scaler,
    win_size=win_size,
    step=20,                
    scale_factor=1.2,    
    max_scales=5          
)

boxes = non_max_suppression(boxes, overlapThresh=0.2)
print(f"[OK] {len(boxes)} detecciones encontradas")

# === 4. Dibujar resultados ===
for (x, y, w, h, _) in boxes: 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detecciones", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
