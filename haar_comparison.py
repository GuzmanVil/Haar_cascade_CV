# compare_cascades.py
#
# Compare:
#   - Custom Haar + AdaBoost cascade (your model, loaded from haar_cascade.pkl)
#   - OpenCV's built-in Haar cascade (haarcascade_frontalface_default.xml)
#
# Usage:
#   python compare_cascades.py              # uses webcam (default camera 0)
#   python compare_cascades.py path/to.jpg  # runs on a single image
#

import sys
import time
import cv2
import joblib
import numpy as np

from src.detect_faces import detect_faces, non_max_suppression


def load_custom_cascade(model_path: str = "haar_cascade.pkl"):
    """Load your trained cascade and scaler."""
    data = joblib.load(model_path)
    cascade = data["cascade"]
    scaler = data["scaler"]
    win_size = data.get("win_size", 24)

    # In your code win_size may be stored as (24, 24)
    if isinstance(win_size, tuple):
        win_size = win_size[0]

    return cascade, scaler, win_size


def load_opencv_cascade():
    """Load OpenCV's built-in frontal face Haar cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise IOError(f"Could not load OpenCV cascade from: {cascade_path}")

    return face_cascade


def run_custom_detector(frame, cascade, scaler, win_size):
    """Run your custom Haar cascade on a single frame and time it."""
    t0 = time.time()
    boxes = detect_faces(
        frame,
        cascade,
        scaler,
        win_size=win_size,
        step=4,           # adjust if you want more speed / more precision
        scale_factor=1.2,
        max_scales=3
    )
    boxes = non_max_suppression(boxes, overlapThresh=0.2)
    t1 = time.time()
    elapsed_ms = (t1 - t0) * 1000.0
    return boxes, elapsed_ms


def run_opencv_detector(frame, cv_cascade):
    """Run OpenCV's Haar cascade on a single frame and time it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t0 = time.time()
    faces = cv_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    t1 = time.time()
    elapsed_ms = (t1 - t0) * 1000.0
    return faces, elapsed_ms


def draw_comparison(frame, boxes_custom, time_custom, faces_cv, time_cv):
    """
    Draw results on a copy of the frame:

    - Green rectangles: custom cascade
    - Blue rectangles: OpenCV cascade
    - Overlay timings and counts
    """
    vis = frame.copy()

    # Custom model (green)
    for (x, y, w, h, _) in boxes_custom:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # OpenCV model (blue)
    for (x, y, w, h) in faces_cv:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    text_custom = f"Custom: {time_custom:.1f} ms, {len(boxes_custom)} boxes"
    text_cv = f"OpenCV: {time_cv:.1f} ms, {len(faces_cv)} boxes"

    cv2.putText(
        vis, text_custom, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    cv2.putText(
        vis, text_cv, (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
    )

    return vis


def run_on_image(image_path):
    """Compare both cascades on a single image."""
    cascade, scaler, win_size = load_custom_cascade()
    cv_cascade = load_opencv_cascade()

    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Could not read image: {image_path}")

    boxes_custom, t_custom = run_custom_detector(img, cascade, scaler, win_size)
    faces_cv, t_cv = run_opencv_detector(img, cv_cascade)

    vis = draw_comparison(img, boxes_custom, t_custom, faces_cv, t_cv)

    print(f"[IMAGE] Custom: {t_custom:.2f} ms, OpenCV: {t_cv:.2f} ms")
    cv2.imshow("Comparison (image)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run on a single image
        run_on_image(sys.argv[1])
    else:
        # Run live on webcam
        run_on_webcam()
