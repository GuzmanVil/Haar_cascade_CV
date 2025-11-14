import os
import random
import cv2
import joblib
from src.detect_faces import detect_faces, non_max_suppression

# === 1. Cargar modelo entrenado ===
data = joblib.load("haar_cascade.pkl")
cascade = data["cascade"]
scaler = data["scaler"]

# Manejo de win_size que puede venir como tuple (24, 24)
win_size = data.get("win_size", 24)
if isinstance(win_size, tuple):
    win_size = win_size[0]

print(f"[INFO] Tamaño de ventana: {win_size}x{win_size}")
print(f"[INFO] Etapas en cascada: {len(cascade)}")

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    scale = 0.10
    small = cv2.resize(frame, None, fx=scale, fy=scale)

    # === 3. Detectar caras ===
    print("[INFO] Detectando caras...")
    boxes = detect_faces(
        small, cascade, scaler,
        win_size=win_size,
        step=20,                
        scale_factor=2,    
        max_scales=4          
    )

    boxes = non_max_suppression(boxes, overlapThresh=0.2)
    print(f"[OK] {len(boxes)} detecciones encontradas")

    # === 4. Dibujar resultados ===
    for (x, y, w, h, _) in boxes: 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detecciones", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
