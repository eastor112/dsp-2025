import cv2
import numpy as np
import matplotlib.pyplot as plt


def obtener_contorno(imagen_path):
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def firma_distancia(contorno):
    centroid = contorno.mean(axis=0)
    return np.linalg.norm(contorno - centroid, axis=1)


def firma_curvatura(contorno):
    dx = np.gradient(contorno[:, 0])
    dy = np.gradient(contorno[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5


def firma_tangente(contorno):
    dx = np.gradient(contorno[:, 0])
    dy = np.gradient(contorno[:, 1])
    return np.arctan2(dy, dx)


def firma_compleja(contorno):
    return contorno[:, 0] + 1j * contorno[:, 1]


contorno = obtener_contorno("lab_images/im1.png")

f_dist = firma_distancia(contorno)
f_curv = firma_curvatura(contorno)
f_tan = firma_tangente(contorno)
f_comp = firma_compleja(contorno).real


plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(f_dist)
plt.title("Firma: Distancia")

plt.subplot(4, 1, 2)
plt.plot(f_curv)
plt.title("Firma: Curvatura")

plt.subplot(4, 1, 3)
plt.plot(f_tan)
plt.title("Firma: Tangente")

plt.subplot(4, 1, 4)
plt.plot(f_comp)
plt.title("Firma: Compleja (parte real)")

plt.tight_layout()
plt.show()
