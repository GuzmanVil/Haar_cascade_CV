# Reconocimiento Facial con Haar Cascade y AdaBoost

Este proyecto implementa un sistema de **reconocimiento de caras** usando el algoritmo **Haar Cascade** entrenado con **AdaBoost**.  
Permite detectar caras en imágenes estáticas mediante un clasificador previamente entrenado. 

---

## Descripción

El modelo se basa en la técnica de **detección en cascada**, que combina:
- **Características Haar-like** para describir patrones visuales.
- **AdaBoost** para seleccionar las características más relevantes y combinarlas en clasificadores débiles.
- **Clasificación en cascada** para acelerar el proceso de detección.

El sistema puede aplicarse a imágenes individuales o integrarse en aplicaciones de visión por computadora.

---

## Integrantes

- Pereira, Santiago
- Tambler, Federico
- Vilche, Guzman

---

## Prerequisitos
Descargar los zips con las imagenes de train, val y test de:
```bash
https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data
```
Sobre cada carpeta dentro de **/dataset** poner la carpeta correspondiente **images/** dentro de cada zip. Resultando en esta estructura
```bash
/dataset/
    ├─ test/
    │   ├─ images/
    │   ├─ wider_face-test.mat
    ├─ train/
    │   ├─ images/
    │   ├─ wider_face_train.mat
    ├─ val/
    │   ├─ images/
    │   ├─ wider_face_val.mat

```

## Instalación / Ejecucción

### 1. Clona este repositorio:
```bash
git clone https://github.com/GuzmanVil/Haar_cascade_CV.git
cd Haar_cascade_CV
```
### 2. Crear Entorno Virtual (Opcional):
```bash
python -m venv ./venv
```
#### Windows
```bash
./venv/Scripts/activate
```
#### Linux/MacOS
```bash
source ./venv/bin/activate
```
### 3. Instalar requerimientos:
```bash
pip install -r requirements.txt
