import numpy as np
import cv2
from src.integral_image import integral_image

def rect_sum(ii, x, y, w, h):
    H, W = ii.shape
    x2 = min(x + w - 1, W - 1)
    y2 = min(y + h - 1, H - 1)

    A = ii[y - 1, x - 1] if x > 0 and y > 0 else 0
    B = ii[y - 1, x2] if y > 0 else 0
    C = ii[y2, x - 1] if x > 0 else 0
    D = ii[y2, x2]
    return D - B - C + A


def compute_haar_features(ii, win_size=24, step=4):
    """Genera las características Haar para una imagen integral (ii)."""
    if isinstance(win_size, tuple):
        win_size = win_size[0]

    feats = []
    H, W = win_size, win_size

    for w in range(12, win_size + 1, 8):
        for h in range(12, win_size + 1, 8):
            for x in range(0, max(1, W - w + 1), step):
                for y in range(0, max(1, H - h + 1), step):
                    mid_w = w // 2
                    if mid_w > 0:
                        left = rect_sum(ii, x, y, mid_w, h)
                        right = rect_sum(ii, x + mid_w, y, mid_w, h)
                        feats.append(left - right)

                    mid_h = h // 2
                    if mid_h > 0:
                        top = rect_sum(ii, x, y, w, mid_h)
                        bottom = rect_sum(ii, x, y + mid_h, w, mid_h)
                        feats.append(top - bottom)

                    third_w = w // 3
                    if third_w > 0 and x + 3 * third_w <= W:
                        a = rect_sum(ii, x, y, third_w, h)
                        b = rect_sum(ii, x + third_w, y, third_w, h)
                        c = rect_sum(ii, x + 2 * third_w, y, third_w, h)
                        feats.append(a - b + c)

                    third_h = h // 3
                    if third_h > 0 and y + 3 * third_h <= H:
                        a = rect_sum(ii, x, y, w, third_h)
                        b = rect_sum(ii, x, y + third_h, w, third_h)
                        c = rect_sum(ii, x, y + 2 * third_h, w, third_h)
                        feats.append(a - b + c)

                    half_w = w // 2
                    half_h = h // 2
                    if half_w > 0 and half_h > 0:
                        A = rect_sum(ii, x, y, half_w, half_h)
                        B = rect_sum(ii, x + half_w, y, half_w, half_h)
                        C = rect_sum(ii, x, y + half_h, half_w, half_h)
                        D = rect_sum(ii, x + half_w, y + half_h, half_w, half_h)
                        feats.append(A - B - C + D)

    return np.array(feats, dtype=np.float64)


def extract_features(images, win_size=24):
    """Extrae las características Haar de una lista de imágenes."""
    all_feats = []
    for img in images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img.astype(np.float64) / 255.0
        ii = integral_image(img)
        feats = compute_haar_features(ii, win_size=win_size, step=4)
        all_feats.append(feats)

    feats = np.vstack(all_feats)
    return feats
