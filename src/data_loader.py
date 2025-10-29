import os
from scipy.io import loadmat
import cv2
import random


def load_annotations(mat_path, images_base_path):

    data = loadmat(mat_path)
    event_list = [e[0][0] for e in data["event_list"]]
    file_list = [f for f in data["file_list"][0]]
    bbox_list = [b for b in data["face_bbx_list"][0]]

    annotations = {}

    for event, files, boxes in zip(event_list, file_list, bbox_list):
        event_dir = os.path.join(images_base_path, event)
        for file_entry, bbox_entry in zip(files, boxes):
            # Extraer correctamente el nombre de archivo y las cajas
            file_name = file_entry[0][0]+".jpg"  # <-- acceso doble
            image_path = os.path.join(event_dir, file_name).replace("\\","/")
            bboxes = bbox_entry.tolist()
            annotations[image_path] = bboxes

    return annotations



def load_image(image_path):
    """
    Carga una imagen desde disco y la convierte a escala de grises.

    Parámetros:
    -----------
    image_path : str
        Ruta absoluta o relativa de la imagen.

    Retorna:
    --------
    img_gray : np.ndarray
        Imagen en escala de grises (uint8).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


if __name__ == "__main__":
    # Ejemplo de uso:
    val_mat = "dataset/val/wider_face_val.mat"
    val_images = "dataset/val/images"

    annotations = load_annotations(val_mat, val_images)
    print(f"Total de imágenes con anotaciones: {len(annotations)}")

    # Mostrar la primera imagen y sus bounding boxes
    img_path = random.choice(list(annotations.keys()))
    bboxes = annotations[img_path][0]
    print(f"Ejemplo: {img_path}")
    print(f"Cajas: {bboxes}")

    img = load_image(img_path)

    # Dibujar las cajas para verificar
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    cv2.imshow("Ejemplo con Anotaciones", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
