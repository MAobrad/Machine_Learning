"""
utils.py - Fonctions mathematiques qu'on utilise partout dans le projet.
Tout est implemente en NumPy pur : softmax, cross-entropy, convolution, max-pooling.
"""

import numpy as np


# ============================================================
# ENCODAGE ET METRIQUES
# ============================================================

def one_hot(y, num_classes=10):
    # Transforme une liste de labels (ex: [3, 7, 1]) en matrice binaire
    # Chaque ligne a un 1 a la position du bon chiffre, 0 partout ailleurs
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y


def softmax(Z):
    # Convertit des scores bruts (logits) en probabilites qui somment a 1
    # On soustrait le max avant d'exponentier pour eviter les overflow numeriques :
    # exp d'un grand nombre donne inf, ce qui casse tout le calcul
    # Mathematiquement identique car exp(z-max)/sum(exp(z-max)) = exp(z)/sum(exp(z))
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def cross_entropy(Y, P):
    # Mesure l'ecart entre ce que le modele predit et les vraies etiquettes
    # Formule : L = -(1/n) * sum(y * log(p))
    # L'epsilon a 1e-15 evite log(0) = -inf qui ferait planter le calcul
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(P), axis=1))


def accuracy(y_true, y_pred):
    # Proportion de bonnes predictions (entre 0 et 1)
    return np.mean(y_true == y_pred)


def matrice_confusion(y_true, y_pred, n_classes=10):
    # Construit la matrice de confusion sans sklearn
    # mat[i][j] = nombre d'exemples de la classe i predit comme classe j
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


# ============================================================
# INITIALISATIONS DES POIDS
# ============================================================

def xavier_init(n_in, n_out):
    # Initialisation Xavier : poids tires dans [-limit, +limit]
    # avec limit = sqrt(6 / (n_in + n_out))
    # Recommandee avec Sigmoid/Tanh pour garder les variances stables entre couches
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


def he_init(n_in, n_out):
    # Initialisation He : ecart-type = sqrt(2 / n_in)
    # Recommandee avec ReLU car ReLU annule ~50% des neurones,
    # donc on double la variance pour compenser ce "gaspillage"
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


# ============================================================
# CONVOLUTION 2D MANUELLE
# ============================================================

def convolve2d(image, kernel, bias=0.0):
    # Convolution 2D avec zero-padding "same" : la sortie a la meme taille que l'entree
    # Sans padding, chaque convolution reduirait la taille de l'image,
    # ce qui serait problematique pour empiler plusieurs couches
    # Pour un kernel 3x3, on ajoute 1 pixel de zeros tout autour
    H, W = image.shape
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    output = np.zeros((H, W))
    for u in range(H):
        for v in range(W):
            patch = padded[u:u+3, v:v+3]
            output[u, v] = np.sum(kernel * patch) + bias
    return output


def convolve2d_color(image_rgb, kernels_rgb, bias=0.0):
    # Convolution sur une image couleur (3 canaux RGB) vers une seule feature map 2D
    # On applique un filtre 3x3 sur chaque canal separement, puis on somme les resultats
    # Formule : sortie[u,v] = sum_c sum_{u',v'} K^(c)[u',v'] * image^(c)[u+u', v+v'] + biais
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
# MAX-POOLING 2x2
# ============================================================

def max_pooling2x2(feature_map):
    # Max-Pooling 2x2 avec stride=2 : on divise la resolution par 2
    # Pour chaque bloc 2x2 pixels, on ne garde que la valeur la plus haute
    # Ca rend le reseau robuste aux petites translations :
    # si une feature est detectee legerement decalee dans le bloc, le max reste le meme
    # Bonus : ca reduit aussi le nombre de parametres dans les couches suivantes
    H, W = feature_map.shape
    out = np.zeros((H // 2, W // 2))
    for u in range(H // 2):
        for v in range(W // 2):
            out[u, v] = np.max(feature_map[2*u:2*u+2, 2*v:2*v+2])
    return out


# ============================================================
# ACTIVATIONS
# ============================================================

def relu(Z):
    return np.maximum(0, Z)


def relu_deriv(Z):
    return (Z > 0).astype(np.float32)
