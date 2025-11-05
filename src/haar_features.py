import numpy as np
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
    feats = []
    H, W = win_size, win_size

    for w in range(8, win_size+1, 4):
        for h in range(8, win_size+1, 4):
            for x in range(0, max(1, W - w + 1), step):
                for y in range(0, max(1, H - h + 1), step):
                    # Tipo 1: dos rectángulos horizontales
                    mid_w = w // 2
                    if mid_w > 0:
                        left = rect_sum(ii, x, y, mid_w, h)
                        right = rect_sum(ii, x+mid_w, y, mid_w, h)
                        feats.append(left - right)

                    # Tipo 2: dos rectángulos verticales
                    mid_h = h // 2
                    if mid_h > 0:
                        top = rect_sum(ii, x, y, w, mid_h)
                        bottom = rect_sum(ii, x, y+mid_h, w, mid_h)
                        feats.append(top - bottom)

                    # Tipo 3: tres rectángulos horizontales
                    third_w = w // 3
                    if third_w > 0 and x + 3*third_w <= W:
                        a = rect_sum(ii, x, y, third_w, h)
                        b = rect_sum(ii, x+third_w, y, third_w, h)
                        c = rect_sum(ii, x+2*third_w, y, third_w, h)
                        feats.append(a - b + c)

                    # Tipo 4: cuatro rectángulos (checkerboard)
                    half_w = w // 2
                    half_h = h // 2
                    if half_w > 0 and half_h > 0:
                        A = rect_sum(ii, x, y, half_w, half_h)
                        B = rect_sum(ii, x+half_w, y, half_w, half_h)
                        C = rect_sum(ii, x, y+half_h, half_w, half_h)
                        D = rect_sum(ii, x+half_w, y+half_h, half_w, half_h)
                        feats.append((A - B - C + D))

    return np.array(feats, dtype=np.float64)


def extract_features(images, win_size=24):
    all_feats = []
    for img in images:
        ii = integral_image(img.astype(np.float64))
        all_feats.append(compute_haar_features(ii, win_size=win_size))
    feats = np.vstack(all_feats)
    print(f"[DEBUG] Imagen de entrada: {images[0].shape}, Features: {feats.shape[1]}")
    return feats


