"""
Comparación de DETECCIÓN entre:

  1) Tu cascada Haar + AdaBoost (haar_cascade.pkl, usando detect_faces)
  2) El Haar cascade de OpenCV (haarcascade_frontalface_default.xml)

A nivel de imagen completa:
  - Se recorre un subconjunto de imágenes anotadas (train o val, NO test)
  - Para cada imagen:
      * se obtienen las cajas ground truth
      * se corre tu detect_faces + NMS  -> predicciones modelo propio
      * se corre cv2.CascadeClassifier -> predicciones OpenCV
      * se emparejan predicciones con GT vía IoU >= umbral
      * se acumulan TP, FP, FN para cada modelo

  - Se calcula precision, recall, F1 y tiempo medio por imagen.

IMPORTANTE:
  - Usa solo wider_face_train.mat o wider_face_val.mat (el test NO tiene face_bbx_list).
  - Esto va a ser LENTO para tu modelo; usa pocos ejemplos (max_images pequeño).
"""

import argparse
import os
import time

import cv2
import joblib
import numpy as np
from scipy.io import loadmat

from src.detect_faces import detect_faces, non_max_suppression


# ------------------------------------------------------------
# Carga anotaciones WIDER (train/val) directamente del .mat
#   Devuelve lista de (ruta_imagen, boxes_gt)
#   boxes_gt: np.array de shape (N, 4) con [x, y, w, h]
# ------------------------------------------------------------
def load_wider_annotations(mat_path, images_root, max_images=None):
    data = loadmat(mat_path)
    events = data["event_list"]
    files_all = data["file_list"]
    boxes_all = data["face_bbx_list"]

    items = []
    total = 0

    num_events = events.shape[0]
    for i in range(num_events):
        event_name = events[i][0][0]
        file_list = files_all[i][0]
        box_list = boxes_all[i][0]

        num_imgs = file_list.shape[0]
        for j in range(num_imgs):
            file_name = file_list[j][0][0]  # ej: "0_Parade_0_0"
            img_path = os.path.join(images_root, event_name, file_name + ".jpg")

            # box_list[j] suele ser un arreglo con shape (1, num_faces, 4)
            bbs = box_list[j][0]
            if bbs.size == 0:
                boxes = np.zeros((0, 4), dtype=float)
            else:
                bbs = np.array(bbs, dtype=float)
                boxes = bbs[:, :4]  # [x, y, w, h]

            items.append((img_path, boxes))
            total += 1

            if max_images is not None and total >= max_images:
                break

        if max_images is not None and total >= max_images:
            break

    print(f"[INFO] Cargadas {len(items)} imágenes anotadas desde {mat_path}")
    return items


# ------------------------------------------------------------
# Carga de tu cascada entrenada
# ------------------------------------------------------------
def load_custom_cascade(model_path="haar_cascade.pkl"):
    data = joblib.load(model_path)
    cascade = data["cascade"]
    scaler = data["scaler"]
    win_size = data.get("win_size", 24)
    if isinstance(win_size, tuple):
        win_size = win_size[0]
    return cascade, scaler, win_size


# ------------------------------------------------------------
# Carga Haar cascade de OpenCV
# ------------------------------------------------------------
def load_opencv_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"No se pudo cargar el cascade de OpenCV desde: {cascade_path}")
    return face_cascade


# ------------------------------------------------------------
# IoU y matching de cajas
# ------------------------------------------------------------
def compute_iou(box_a, box_b):
    """
    box_*: [x, y, w, h]
    """
    x1_a, y1_a, w_a, h_a = box_a
    x2_a = x1_a + w_a
    y2_a = y1_a + h_a

    x1_b, y1_b, w_b, h_b = box_b
    x2_b = x1_b + w_b
    y2_b = y1_b + h_b

    xx1 = max(x1_a, x1_b)
    yy1 = max(y1_a, y1_b)
    xx2 = min(x2_a, x2_b)
    yy2 = min(y2_a, y2_b)

    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a + area_b - inter + 1e-8

    return inter / union


def match_detections(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    pred_boxes: np.array (M, 4)  [x, y, w, h]
    gt_boxes:   np.array (N, 4)
    Devuelve: tp, fp, fn
    """
    if gt_boxes.size == 0:
        # No hay caras en GT; todo lo que predigas es FP
        tp = 0
        fp = pred_boxes.shape[0]
        fn = 0
        return tp, fp, fn

    matched_gt = set()
    tp = 0
    fp = 0

    for i in range(pred_boxes.shape[0]):
        pb = pred_boxes[i]
        best_iou = 0.0
        best_j = -1

        for j in range(gt_boxes.shape[0]):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = gt_boxes.shape[0] - len(matched_gt)
    return tp, fp, fn


# ------------------------------------------------------------
# Evaluación a nivel de imagen completa
# ------------------------------------------------------------
def evaluate_detection(
    annotations,
    cascade,
    scaler,
    win_size,
    cv_cascade,
    iou_thresh=0.5,
    custom_step=4,
    custom_scale_factor=1.2,
    custom_max_scales=1,
    resize_long_side=None,
):
    """
    annotations: lista de (img_path, boxes_gt)
    """
    tp_custom = fp_custom = fn_custom = 0
    tp_cv = fp_cv = fn_cv = 0

    time_custom = 0.0
    time_cv = 0.0
    num_imgs = 0

    for img_path, gt_boxes in annotations:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] No se pudo leer {img_path}, se salta.")
            continue

        orig_h, orig_w = img.shape[:2]

        # Redimensionar opcionalmente para acelerar
        scale = 1.0
        if resize_long_side is not None:
            long_side = max(orig_w, orig_h)
            if long_side > resize_long_side:
                scale = resize_long_side / float(long_side)
                img = cv2.resize(img, None, fx=scale, fy=scale)

        # Escalar GT si se redimensionó la imagen
        if scale != 1.0:
            gt_scaled = gt_boxes.copy().astype(float)
            gt_scaled[:, 0] *= scale
            gt_scaled[:, 1] *= scale
            gt_scaled[:, 2] *= scale
            gt_scaled[:, 3] *= scale
        else:
            gt_scaled = gt_boxes

        # --- Detección con tu cascada ---
        t0 = time.perf_counter()
        boxes_custom = detect_faces(
            img,
            cascade,
            scaler,
            win_size=win_size,
            step=custom_step,
            scale_factor=custom_scale_factor,
            max_scales=custom_max_scales,
        )
        boxes_custom = non_max_suppression(boxes_custom, overlapThresh=0.3)
        time_custom += (time.perf_counter() - t0)

        if len(boxes_custom) > 0:
            boxes_c = np.array([[b[0], b[1], b[2], b[3]] for b in boxes_custom], dtype=float)
        else:
            boxes_c = np.zeros((0, 4), dtype=float)

        tp_c, fp_c, fn_c = match_detections(boxes_c, gt_scaled, iou_thresh=iou_thresh)
        tp_custom += tp_c
        fp_custom += fp_c
        fn_custom += fn_c

        # --- Detección con OpenCV ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        faces_cv = cv_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        time_cv += (time.perf_counter() - t0)

        if len(faces_cv) > 0:
            boxes_cv = np.array(faces_cv, dtype=float)  # [x, y, w, h]
        else:
            boxes_cv = np.zeros((0, 4), dtype=float)

        tp_v, fp_v, fn_v = match_detections(boxes_cv, gt_scaled, iou_thresh=iou_thresh)
        tp_cv += tp_v
        fp_cv += fp_v
        fn_cv += fn_v

        num_imgs += 1

    # Métricas
    def metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    p_c, r_c, f1_c = metrics(tp_custom, fp_custom, fn_custom)
    p_v, r_v, f1_v = metrics(tp_cv, fp_cv, fn_cv)

    ms_custom = (time_custom / num_imgs) * 1000.0 if num_imgs > 0 else 0.0
    ms_cv = (time_cv / num_imgs) * 1000.0 if num_imgs > 0 else 0.0

    print("\n================= RESULTADOS DETECCIÓN (imagen completa) =================")
    print(f"Imágenes evaluadas: {num_imgs}")
    print(f"IoU threshold: {iou_thresh}")

    print("\n--- Tu cascada Haar + AdaBoost ---")
    print(f"TP={tp_custom}, FP={fp_custom}, FN={fn_custom}")
    print(f"Precision: {p_c:.4f}")
    print(f"Recall   : {r_c:.4f}")
    print(f"F1       : {f1_c:.4f}")
    print(f"Tiempo medio por imagen: {ms_custom:.2f} ms")

    print("\n--- Haar cascade OpenCV ---")
    print(f"TP={tp_cv}, FP={fp_cv}, FN={fn_cv}")
    print(f"Precision: {p_v:.4f}")
    print(f"Recall   : {r_v:.4f}")
    print(f"F1       : {f1_v:.4f}")
    print(f"Tiempo medio por imagen: {ms_cv:.2f} ms")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar detección (imagen completa) entre tu cascada y la de OpenCV."
    )
    parser.add_argument("--mat", required=True, help="Ruta al .mat (train o val, NO test).")
    parser.add_argument("--images", required=True, help="Carpeta base de imágenes (WIDER_train/images o WIDER_val/images).")
    parser.add_argument("--model", default="haar_cascade.pkl", help="Ruta a tu modelo entrenado (joblib).")
    parser.add_argument("--max-images", type=int, default=20, help="Máximo de imágenes a evaluar (para que no tarde horas).")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU mínimo para considerar TP.")
    parser.add_argument("--custom-step", type=int, default=4, help="Step de sliding window para tu cascada.")
    parser.add_argument("--custom-scale-factor", type=float, default=1.2, help="Factor de escala entre tamaños de ventana.")
    parser.add_argument("--custom-max-scales", type=int, default=1, help="Número máximo de escalas para tu cascada.")
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=640,
        help="Redimensiona la imagen para que el lado largo sea como mucho este valor (acelera tu modelo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    annotations = load_wider_annotations(args.mat, args.images, max_images=args.max_images)
    cascade, scaler, win_size = load_custom_cascade(args.model)
    cv_cascade = load_opencv_cascade()

    evaluate_detection(
        annotations,
        cascade,
        scaler,
        win_size,
        cv_cascade,
        iou_thresh=args.iou_thresh,
        custom_step=args.custom_step,
        custom_scale_factor=args.custom_scale_factor,
        custom_max_scales=args.custom_max_scales,
        resize_long_side=args.resize_long_side,
    )
