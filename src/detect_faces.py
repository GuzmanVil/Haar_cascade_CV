import cv2
import numpy as np

from src.haar_features import compute_haar_features

def detect_faces(img, model, scaler, win_size=24, step=10, scale_factor=1.25, prob_threshold=0.8, max_scales=10):
    boxes = []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = win_size
    scale_count = 0  # contador de escalas

    while size < min(h, w) and scale_count < max_scales:
        for y in range(0, h - size, step):
            for x in range(0, w - size, step):
                patch = gray[y:y+size, x:x+size]
                patch_resized = cv2.resize(patch, (win_size, win_size))
                ii = cv2.integral(patch_resized)[1:, 1:]
                feat = compute_haar_features(ii, win_size=win_size).reshape(1, -1)
                feat_scaled = scaler.transform(feat)

                prob = model.predict_proba(feat_scaled)[0, 1]
                if prob > prob_threshold:
                    boxes.append((x, y, size, size, prob))

        size = int(size * scale_factor)
        scale_count += 1  # aumentar contador

    print(f"[INFO] Detecciones iniciales: {len(boxes)} (en {scale_count} escalas)")
    return boxes


def non_max_suppression(boxes, overlapThresh=0.5):
    """
    Elimina cajas solapadas conservando la de mayor probabilidad.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]

        order = order[np.where(overlap <= overlapThresh)[0] + 1]

    final_boxes = boxes[keep].astype(int).tolist()
    print(f"[OK] {len(final_boxes)} detecciones finales tras NMS.")
    return final_boxes
