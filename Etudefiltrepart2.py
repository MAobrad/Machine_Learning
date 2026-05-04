import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Redimensionner en 423x318 (largeur x hauteur)
image_resized = cv2.resize(image, (423, 318))

# Définition des filtres
K1 = (1/9) * np.ones((3,3))

K2 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

K3 = np.array([
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1]
])

K4 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

K5 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

K6 = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
])

kernels = [K1, K2, K3, K4, K5, K6]
titles = ["K1 (blur)", "K2 (sharpen)", "K3", "K4", "K5 (sobel)", "K6"]

# Appliquer les filtres
results = []
for k in kernels:
    filtered = cv2.filter2D(image_resized, -1, k)
    results.append(filtered)

# Affichage
plt.figure(figsize=(12,6))

plt.subplot(2,4,1)
plt.imshow(image_resized, cmap='gray')
plt.title("Original (423x318)")
plt.axis('off')

for i in range(6):
    plt.subplot(2,4,i+2)
    plt.imshow(results[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
