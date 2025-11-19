import cv2
import numpy as np
from scipy import stats
import pandas as pd

def analyze_texture(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten().astype(np.float64)

    mean = np.mean(pixels)
    std_dev = np.std(pixels)
    var = np.var(pixels)

    # R1: varianza normalizada por el rango
    R1 = var / (255.0 ** 2)

    # R2: definición alternativa
    R2 = 1 - 1 / (1 + var)

    third_moment = stats.skew(pixels)

    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist_norm = hist / np.sum(hist)

    uniformity = np.sum(hist_norm ** 2)

    h = hist_norm[hist_norm > 0]
    entropy = -np.sum(h * np.log2(h))

    return {
        'Mean': mean,
        'StdDev': std_dev,
        'Variance': var,
        'R1_normalized': R1,
        'R2_classical': R2,
        'Third_moment': third_moment,
        'Uniformity': uniformity,
        'Entropy': entropy
    }


def classify_texture(entropy):
    if entropy < 6:
        return "Smooth"
    elif entropy > 7.5:
        return "Coarse"
    return "Regular"


images = [
    'lab_images/text_suave.png',
    'lab_images/text_rugoso.png',
    'lab_images/text_regular.png'
]

results = []

for img_path in images:
    feats = analyze_texture(img_path)
    if feats:
        label = classify_texture(feats['Entropy'])
        print(f"{img_path} → {label}")

        results.append({
            'Texture': label,
            'Image': img_path,
            **feats
        })

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv('texture_analysis_results.csv', index=False)
