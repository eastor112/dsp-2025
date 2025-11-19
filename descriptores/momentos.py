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


def calcular_momento(firma, n):
    """
    Calcula el n-ésimo momento de una firma respecto a su media

    μₙ(r) = Σ (rᵢ - m)ⁿ · g(rᵢ)

    donde:
    - rᵢ son los valores de la firma (distancias)
    - m es la media de la firma
    - g(rᵢ) = 1 para todos los puntos al ser una imagen binaria
    """
    m = np.mean(firma)  # centro de masa (media de la firma)
    momento = np.sum((firma - m)**n)
    return momento


def calcular_momento_normalizado(firma, n):
    """
    Calcula el momento normalizado (dividido por el número de puntos)
    """
    m = np.mean(firma)
    momento = np.mean((firma - m)**n)  # equivalente a sum/N
    return momento



contorno = obtener_contorno("lab_images/im1.png")
f_dist = firma_distancia(contorno)


for n in range(5):
    momento = calcular_momento(f_dist, n)
    momento_norm = calcular_momento_normalizado(f_dist, n)
    print(f"\nMomento de orden {n}:")
    print(f"  μ_{n} = {momento:.4f}")
    print(f"  μ_{n} normalizado = {momento_norm:.4f}")


m0 = calcular_momento_normalizado(f_dist, 0)
m1 = calcular_momento_normalizado(f_dist, 1)
m2 = calcular_momento_normalizado(f_dist, 2)  # Varianza
m3 = calcular_momento_normalizado(f_dist, 3)  # Asimetría
m4 = calcular_momento_normalizado(f_dist, 4)  # Curtosis

print(f"μ₀: Momento de orden 0 = {m0:.4f} (siempre 1 si normalizado)")
print(f"μ₁: Momento de orden 1 = {m1:.6f} (debe ser ≈ 0)")
print(f"μ₂: Varianza = {m2:.4f}")
print(f"    Desviación estándar = {np.sqrt(m2):.4f}")
print(f"μ₃: Relacionado con asimetría = {m3:.4f}")
print(f"μ₄: Relacionado con curtosis = {m4:.4f}")
