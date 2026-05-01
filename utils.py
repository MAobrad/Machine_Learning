"""
utils.py — Fonctions mathématiques partagées (NumPy pur).
Softmax, cross-entropy, convolution, max-pooling, initialisation.
"""

import numpy as np


# ============================================================
# ENCODAGE ET MÉTRIQUES
# ============================================================

def one_hot(y, num_classes=10):
    """Encode les étiquettes en vecteurs one-hot."""
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y


def softmax(Z):
    """
    Softmax avec stabilité numérique.

    # Pourquoi soustraire le max ?
    # softmax(z)_k = exp(z_k) / sum(exp(z_j))
    # Si z_k est grand → exp(z_k) → inf (overflow).
    # On soustrait max(z) : exp(z_k - max) / sum(exp(z_j - max))
    # = exp(z_k)*exp(-max) / (sum(exp(z_j))*exp(-max))
    # Le résultat est identique mathématiquement mais numériquement stable.
    """
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def cross_entropy(Y, P):
    """
    Cross-entropy multi-classe.
    L = -(1/n) * sum_i sum_k y_i^(k) * ln(P_k(x_i))
    epsilon évite log(0) = -inf.
    """
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(P), axis=1))


def accuracy(y_true, y_pred):
    """Taux de bonnes prédictions."""
    return np.mean(y_true == y_pred)


def matrice_confusion(y_true, y_pred, n_classes=10):
    """Calcule la matrice de confusion manuellement (sans sklearn)."""
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


# ============================================================
# INITIALISATIONS
# ============================================================

def xavier_init(n_in, n_out):
    """
    Initialisation Xavier/Glorot.
    W ~ Uniform(-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out)))
    Recommandée avec Sigmoid/Tanh.
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


def he_init(n_in, n_out):
    """
    Initialisation He.
    W ~ N(0, sqrt(2/n_in))
    Recommandée avec ReLU : compense la moitié des neurones annulés.
    """
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


# ============================================================
# CONVOLUTION 2D MANUELLE
# ============================================================

def convolve2d(image, kernel, bias=0.0):
    """
    Convolution 2D avec zero-padding (same convolution).

    # À quoi sert le zero-padding ?
    # Sans padding, chaque convolution réduit la taille de l'image.
    # Avec 'same' padding (1 pixel de zéros tout autour pour un kernel 3×3),
    # la sortie conserve les dimensions (H, W) de l'entrée.
    # Cela permet d'empiler plusieurs couches sans réduire prématurément
    # la résolution spatiale.

    Args:
        image  : np.array (H, W) — image en niveaux de gris
        kernel : np.array (3, 3) — filtre à appliquer
        bias   : float — biais additif
    Returns:
        np.array (H, W) — feature map filtrée
    """
    H, W = image.shape
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    output = np.zeros((H, W))
    for u in range(H):
        for v in range(W):
            patch = padded[u:u+3, v:v+3]
            output[u, v] = np.sum(kernel * patch) + bias
    return output


def convolve2d_color(image_rgb, kernels_rgb, bias=0.0):
    """
    Convolution sur image couleur (3 canaux) → 1 feature map.

    m'_{u,v} = sum_c sum_{u',v'} K^(c)_{u',v'} * m^(c)_{u+u'-2, v+v'-2} + l

    Args:
        image_rgb   : np.array (H, W, 3) normalisé [0,1]
        kernels_rgb : np.array (3, 3, 3) — un filtre 3×3 par canal RGB
        bias        : float
    Returns:
        np.array (H, W) — feature map 2D
    """
    H, W = image_rgb.shape[:2]
    output = np.zeros((H, W))
    for c in range(3):
        padded = np.pad(image_rgb[:, :, c], 1, mode='constant', constant_values=0)
        for u in range(H):
            for v in range(W):
                patch = padded[u:u+3, v:v+3]
                output[u, v] += np.sum(kernels_rgb[:, :, c] * patch)
    return output + bias


# ============================================================
# MAX-POOLING 2×2
# ============================================================

def max_pooling2x2(feature_map):
    """
    Max-Pooling 2×2 avec stride=2. Réduit H et W par 2.

    # Pourquoi le Max-Pooling rend le modèle robuste aux translations ?
    # En prenant le max sur une région 2×2, une feature détectée à une
    # position légèrement différente dans cette région produit la même
    # sortie. → invariance locale aux petites translations.
    # Avantage supplémentaire : réduit le nombre de paramètres.

    Args:
        feature_map : np.array (H, W)
    Returns:
        np.array (H//2, W//2)
    """
    H, W = feature_map.shape
    out = np.zeros((H // 2, W // 2))
    for u in range(H // 2):
        for v in range(W // 2):
            out[u, v] = np.max(feature_map[2*u:2*u+2, 2*v:2*v+2])
    return out
