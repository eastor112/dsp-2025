import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_fft_spectrum(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None

    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    mag_log = np.log1p(mag)
    return img, mag, mag_log

def cartesian_to_polar(spectrum):
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.abs(np.arctan2(y - cy, x - cx))
    return spectrum, r, theta

def compute_radial_profile(spectrum, r_matrix, max_radius=None):
    if max_radius is None:
        max_radius = min(spectrum.shape) // 2
    radii = np.arange(max_radius)
    s_r = np.zeros_like(radii, dtype=float)
    for i, r in enumerate(radii):
        mask = (r_matrix >= r) & (r_matrix < r + 1)
        if mask.sum() > 0:
            s_r[i] = spectrum[mask].sum()
    return radii, s_r

def compute_angular_profile(spectrum, theta_matrix, r_matrix, num_angles=180):
    angles = np.linspace(0, np.pi, num_angles)
    s_theta = np.zeros_like(angles)
    max_r = min(spectrum.shape) // 2
    angle_w = np.pi / num_angles
    for i, ang in enumerate(angles):
        mask = (np.abs(theta_matrix - ang) < angle_w) & (r_matrix < max_r)
        if mask.sum() > 0:
            s_theta[i] = spectrum[mask].sum()
    return angles, s_theta

def extract_spectral_descriptors(radii, s_r, angles, s_theta):
    d = {}
    if s_r.sum() > 0:
        d['radial_max_location'] = radii[np.argmax(s_r)]
        d['radial_max_value'] = s_r.max()
        d['radial_mean'] = s_r.mean()
        d['radial_variance'] = s_r.var()
        d['radial_mean_location'] = (radii * s_r).sum() / s_r.sum()
        d['radial_max_mean_distance'] = abs(
            d['radial_max_location'] - d['radial_mean_location']
        )
    if s_theta.sum() > 0:
        idx = np.argmax(s_theta)
        d['angular_max_location'] = angles[idx]
        d['angular_max_value'] = s_theta.max()
        d['angular_mean'] = s_theta.mean()
        d['angular_variance'] = s_theta.var()
        d['principal_direction_deg'] = np.degrees(angles[idx])
    return d

def analyze_spectral_texture(image_path, visualize=True):
    img, mag, mag_log = compute_fft_spectrum(image_path)
    if img is None:
        return None

    spectrum, r_matrix, theta_matrix = cartesian_to_polar(mag)
    radii, s_r = compute_radial_profile(spectrum, r_matrix)
    angles, s_theta = compute_angular_profile(spectrum, theta_matrix, r_matrix)
    descriptors = extract_spectral_descriptors(radii, s_r, angles, s_theta)

    if visualize:
        plt.figure(figsize=(16,8))

        plt.subplot(2,3,1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(2,3,2)
        plt.imshow(mag_log, cmap='jet')
        plt.axis('off')

        plt.subplot(2,3,3)
        plt.imshow(mag_log, cmap='jet')
        center = np.array(mag_log.shape) // 2
        for r in [10,30,50,70]:
            circ = plt.Circle((center[1], center[0]), r, fill=False, color='white')
            plt.gca().add_patch(circ)
        plt.axis('off')

        plt.subplot(2,3,4)
        plt.plot(radii, s_r)
        if 'radial_max_location' in descriptors:
            plt.axvline(descriptors['radial_max_location'], color='r', linestyle='--')
        plt.title('S(r)')

        plt.subplot(2,3,5)
        plt.plot(np.degrees(angles), s_theta)
        if 'principal_direction_deg' in descriptors:
            plt.axvline(descriptors['principal_direction_deg'], color='r', linestyle='--')
        plt.title('S(θ)')

        plt.subplot(2,3,6, projection='polar')
        plt.plot(angles, s_theta)
        plt.title('Perfil angular')

        plt.tight_layout()
        plt.show()

    return descriptors

images = [
    'lab_images/text_suave.png',
    'lab_images/text_rugoso.png',
    'lab_images/text_regular.png'
]

all_results = {img: analyze_spectral_texture(img) for img in images}
