# src/haar_features.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# -----------------------
# Utilidades con integral
# -----------------------

def rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Suma de intensidades en rectángulo usando imagen integral PADDEADA.
    ii: integral (H+1, W+1)
    (x, y): esquina sup-izq del rectángulo en coords de imagen (0-based)
    (w, h): ancho y alto del rectángulo
    """
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    # Nota: como ii está paddeada, desplazamos +1 en ambos ejes.
    return ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]

# -----------------------
# Definición de features
# -----------------------

@dataclass(frozen=True)
class HaarFeature:
    """
    ftype:
      - 'two_h'   -> dos rectángulos horizontales (lado a lado)
      - 'two_v'   -> dos rectángulos verticales (uno sobre otro)
      - 'three_h' -> tres rectángulos horizontales
      - 'four'    -> cuatro rectángulos en cuadrícula 2x2
    x, y:   esquina sup-izq dentro de la ventana base (24x24)
    w, h:   tamaño total que ocupa la feature dentro de la ventana
            (respetando múltiplos según tipo)
    """
    ftype: str
    x: int
    y: int
    w: int
    h: int


def generate_haar_features(window_size: Tuple[int, int]=(24, 24)) -> List[HaarFeature]:
    """
    Genera TODAS las features posibles dentro de una ventana 24x24 (por defecto).
    Regla de múltiplos:
      - two_h: w divisible por 2
      - two_v: h divisible por 2
      - three_h: w divisible por 3
      - four: w y h divisibles por 2
    """
    W, H = window_size
    feats: List[HaarFeature] = []

    # two_h: [blanco | negro] (horizontal)
    for h in range(1, H + 1):
        for w in range(2, W + 1, 2):  # múltiplo de 2
            for y in range(0, H - h + 1):
                for x in range(0, W - w + 1):
                    feats.append(HaarFeature('two_h', x, y, w, h))

    # two_v: [blanco/negro en vertical]
    for h in range(2, H + 1, 2):      # múltiplo de 2
        for w in range(1, W + 1):
            for y in range(0, H - h + 1):
                for x in range(0, W - w + 1):
                    feats.append(HaarFeature('two_v', x, y, w, h))

    # three_h: [blanco | negro | blanco] (horizontal)
    for h in range(1, H + 1):
        for w in range(3, W + 1, 3):  # múltiplo de 3
            for y in range(0, H - h + 1):
                for x in range(0, W - w + 1):
                    feats.append(HaarFeature('three_h', x, y, w, h))

    # four: cuadrícula 2x2 (blanco-negro / negro-blanco)
    for h in range(2, H + 1, 2):
        for w in range(2, W + 1, 2):
            for y in range(0, H - h + 1):
                for x in range(0, W - w + 1):
                    feats.append(HaarFeature('four', x, y, w, h))

    return feats


def evaluate_feature_on_window(ii_win: np.ndarray, feat: HaarFeature) -> float:
    """
    Evalúa el valor de la feature en una ventana 24x24 (integral PADDEADA de esa ventana).
    Convención de signos (Viola-Jones):
      - Dos rects: suma(rect_negro) - suma(rect_blanco)
      - Tres rects: centro (negro) - (izq + der) (blancos)
      - Cuatro: (sum(diag negra) - sum(diag blanca)) -> convención común
    Puedes cambiar la convención; AdaBoost aprenderá el umbral/inequidad.
    """
    x, y, w, h = feat.x, feat.y, feat.w, feat.h

    if feat.ftype == 'two_h':
        w2 = w // 2
        left = rect_sum(ii_win, x, y, w2, h)
        right = rect_sum(ii_win, x + w2, y, w2, h)
        return right - left  # ojos (oscuro) vs mejillas (claro) si se alinea

    elif feat.ftype == 'two_v':
        h2 = h // 2
        top = rect_sum(ii_win, x, y, w, h2)
        bottom = rect_sum(ii_win, x, y + h2, w, h2)
        return bottom - top

    elif feat.ftype == 'three_h':
        w3 = w // 3
        left  = rect_sum(ii_win, x, y, w3, h)
        mid   = rect_sum(ii_win, x + w3, y, w3, h)
        right = rect_sum(ii_win, x + 2*w3, y, w3, h)
        return mid - (left + right)

    elif feat.ftype == 'four':
        w2 = w // 2
        h2 = h // 2
        tl = rect_sum(ii_win, x, y, w2, h2)               # top-left
        tr = rect_sum(ii_win, x + w2, y, w2, h2)          # top-right
        bl = rect_sum(ii_win, x, y + h2, w2, h2)          # bottom-left
        br = rect_sum(ii_win, x + w2, y + h2, w2, h2)     # bottom-right
        return (tl + br) - (tr + bl)

    else:
        raise ValueError(f"Tipo de feature desconocido: {feat.ftype}")
