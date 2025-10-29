# src/integral_image.py
import numpy as np

def compute_integral_image(img, pad=True):
    """
    Calcula la imagen integral. Si pad=True devuelve tamaño (H+1, W+1)
    con una fila y columna de ceros al inicio (más cómodo para rect_sum).
    """
    img = img.astype(np.float64)
    ii = img.cumsum(axis=0).cumsum(axis=1)
    if not pad:
        return ii
    H, W = img.shape
    out = np.zeros((H + 1, W + 1), dtype=np.float64)
    out[1:, 1:] = ii
    return out
