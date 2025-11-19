import cv2
import numpy as np
import matplotlib.pyplot as plt


def obtener_contorno(imagen_path):
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def calcular_descriptores_fourier(contorno):
    contorno = contorno.reshape(-1, 2)
    s = contorno[:, 0] + 1j * contorno[:, 1]
    return np.fft.fft(s)


def reconstruir_contorno(descriptores, M=None):
    N = len(descriptores)
    if M is None:
        M = N
    a = np.zeros(N, dtype=complex)
    half = M // 2
    a[:half] = descriptores[:half]
    a[-half:] = descriptores[-half:]
    s = np.fft.ifft(a)
    return np.column_stack((np.real(s), np.imag(s))).astype(np.int32)


def visualizar_reconstrucciones(contorno, a, M_values):
    fig, axes = plt.subplots(1, len(M_values) + 1,
                             figsize=(4 * (len(M_values)+1), 4))
    c = contorno.reshape(-1, 2)

    axes[0].plot(c[:, 0], c[:, 1], 'b-')
    axes[0].set_title('Original')
    axes[0].axis('equal')
    axes[0].invert_yaxis()

    for i, M in enumerate(M_values):
        r = reconstruir_contorno(a, M)
        axes[i+1].plot(r[:, 0], r[:, 1], 'r-')
        axes[i+1].set_title(f'M = {M}')
        axes[i+1].axis('equal')
        axes[i+1].invert_yaxis()

    plt.tight_layout()
    plt.show()


img = cv2.imread('lab_images/im1.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contorno = max(contours, key=cv2.contourArea)

a = calcular_descriptores_fourier(contorno)

M_values = [10, 30, 50, min(100, len(a)//2)]
visualizar_reconstrucciones(contorno, a, M_values)
