import os
import cv2
import numpy as np
import random
from scipy.io import loadmat

def load_annotations(mat_path, images_base_path):
    data = loadmat(mat_path)
    events = data["event_list"]
    files_all = data["file_list"]
    boxes_all = data["face_bbx_list"]

    annotations = {}
    total_imgs = 0

    for i in range(events.shape[0]):
        event_name = events[i][0][0]
        event_dir = os.path.join(images_base_path, event_name)
        files = files_all[i][0]
        boxes = boxes_all[i][0]
        for j in range(len(files)):
            file_name = files[j][0][0] + ".jpg"
            image_path = os.path.join(event_dir, file_name).replace("\\", "/")
            bboxes = np.array(boxes[j][0], dtype=np.float32)  # Nx4
            annotations[image_path] = bboxes
            total_imgs += 1
    print(f"[INFO] Eventos: {events.shape[0]} | Imágenes anotadas: {total_imgs}")
    return annotations


def generate_samples_in_memory(annotations, img_size=(24, 24), neg_per_img=1, max_imgs=None):
    positives, negatives = [], []
    count = 0

    for img_path, bboxes in annotations.items():
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        # POSITIVOS
        for (x, y, bw, bh) in bboxes:
            x, y, bw, bh = map(int, [x, y, bw, bh])
            if bw <= 5 or bh <= 5 or x < 0 or y < 0 or x + bw > w or y + bh > h:
                continue
            crop = img[y:y+bh, x:x+bw]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(
                cv2.resize(crop, img_size, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY
            )
            positives.append(gray)

        # NEGATIVOS
        for _ in range(neg_per_img):
            tries = 0
            while tries < 20:
                nx = random.randint(0, max(0, w - img_size[0]))
                ny = random.randint(0, max(0, h - img_size[1]))
                overlap = any(
                    (nx < x + bw and nx + img_size[0] > x and ny < y + bh and ny + img_size[1] > y)
                    for (x, y, bw, bh) in bboxes
                )
                if not overlap:
                    patch = img[ny:ny+img_size[1], nx:nx+img_size[0]]
                    if patch.size:
                        negatives.append(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
                        break
                tries += 1

        count += 1
        if max_imgs and count >= max_imgs:
            break

    print(f"[OK] Generadas {len(positives)} positivas y {len(negatives)} negativas ({count} imágenes procesadas)")
    return np.array(positives), np.array(negatives)
