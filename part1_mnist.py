"""
part1_mnist.py — Partie 1 : Classification MNIST
Implémentation NumPy pure : forward, backward, descente de gradient.

Audit réalisé sur le code original (main.py / models.py) du coéquipier.
Corrections et compléments intégrés ici.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Reproductibilité — garantit des résultats identiques à chaque exécution
np.random.seed(42)
os.makedirs('rapport', exist_ok=True)

from utils import (
    one_hot, softmax, cross_entropy, accuracy,
    xavier_init, he_init, matrice_confusion
)


# ============================================================
# RAPPORT D'AUDIT DU CODE ORIGINAL
# ============================================================

def audit_partie1():
    """Affiche le rapport d'audit du code du coéquipier."""
    print("\n" + "=" * 62)
    print("  RAPPORT D'AUDIT — PARTIE 1 (code du coéquipier)")
    print("=" * 62)

    print("\n✅  CORRECT ET FONCTIONNEL :")
    print("    - Chargement MNIST via tensorflow.keras.datasets")
    print("    - Aplatissement (n,28,28) → (n,784) correct")
    print("    - Normalisation /255 → [0,1] ✓")
    print("    - Split 60 000 train / 10 000 test standard ✓")
    print("    - Softmax avec stabilité numérique (soustraction max) ✓")
    print("    - Cross-entropy avec epsilon anti-log(0) ✓")
    print("    - Mini-batch SGD avec shuffle ✓")
    print("    - 3 architectures : Linéaire, MLP-1, MLP-2 ✓")
    print("    - ReLU et sa dérivée manuelles ✓")
    print("    - Rétropropagation complète pour chaque modèle ✓")
    print("    - Taux d'erreur affiché sur train ET test ✓")
    print("    - Grid search architectures × learning rates ✓")
    print("    - Visualisation des images mal classées ✓")

    print("\n❌  ABSENT (ajouté dans cette version) :")
    print("    - np.random.seed(42) pour reproductibilité")
    print("    - Matrice de confusion")
    print("    - Visualisation PCA 2D des représentations")
    print("    - Visualisation t-SNE")
    print("    - Export figures → rapport/")
    print("    - Section if __name__ == '__main__':")
    print("    - Initialisation Xavier/He (code utilisait sqrt(2/n)≈He, OK)")

    print("\n⚠️   PRÉSENT MAIS INCOMPLET / REDONDANT :")
    print("    - one_hot défini deux fois (local + import) → redondant")
    print("    - y_train_onehot calculé au niveau module sans être utilisé")
    print("    - Couches cachées par défaut à 64 neurones < recommandé (128/256)")
    print("    - Epochs par défaut = 50 (limite basse, 50-100 recommandé)")

    print("\n📐  OBSERVATIONS MATHÉMATIQUES :")
    print("    - Convention A de forme (784,10) au lieu de (10,784) du sujet")
    print("      → mathematiquement équivalent : X@A vs (A@x^T)^T")
    print("    - dL/dA = X^T @ (P-Y)/n correct pour A (784,10) ✓")
    print("    - dL/db = sum((P-Y)/n, axis=0) correct ✓")
    print("    - ReLU' = (Z>0).astype(float) correct ✓")
    print("    - Gradient de sortie delta = (P-Y)/n (formule analytique softmax+CE) ✓")

    print("\n" + "=" * 62)
    print("  CONCLUSION : Base solide. Éléments manquants complétés ci-dessous.")
    print("=" * 62 + "\n")


# ============================================================
# 1. CHARGEMENT MNIST
# ============================================================

def charger_mnist():
    """Charge et prépare les données MNIST."""
    print("Chargement de MNIST...")
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except ImportError:
        try:
            from torchvision.datasets import MNIST
            train_ds = MNIST('.', train=True, download=True)
            test_ds  = MNIST('.', train=False, download=True)
            x_train  = np.array(train_ds.data)
            y_train  = np.array(train_ds.targets)
            x_test   = np.array(test_ds.data)
            y_test   = np.array(test_ds.targets)
        except ImportError:
            raise ImportError("Installer tensorflow ou torchvision.")

    # Aplatissement : (n, 28, 28) → (n, 784)
    x_train = x_train.reshape(x_train.shape[0], 784).astype(np.float32)
    x_test  = x_test.reshape(x_test.shape[0], 784).astype(np.float32)

    # Normalisation [0, 1]
    x_train /= 255.0
    x_test  /= 255.0

    print(f"  x_train : {x_train.shape}  y_train : {y_train.shape}")
    print(f"  x_test  : {x_test.shape}   y_test  : {y_test.shape}")
    return x_train, y_train, x_test, y_test


# ============================================================
# 2. MODÈLES NumPy PURS
# ============================================================

class ModeleLineaire:
    """
    Modèle linéaire multi-classe.
    o = X @ A + b  →  P = softmax(o)

    Convention : A de forme (input_dim, output_dim) pour format batch.
    Initialisation Xavier recommandée pour Sigmoid/couches linéaires.
    """

    def __init__(self, input_dim=784, output_dim=10):
        self.A = xavier_init(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X      = X
        self.logits = X @ self.A + self.b
        self.P      = softmax(self.logits)
        return self.P

    def backward(self, Y):
        """
        Gradient analytique cross-entropy + softmax :
        dL/do_k = (P_k - y_k) / n  [formule directe, pas de chainrule séparée]
        dL/dA   = X^T @ dZ
        dL/db   = sum(dZ, axis=0)
        """
        n        = self.X.shape[0]
        dZ       = (self.P - Y) / n
        self.dA  = self.X.T @ dZ
        self.db  = np.sum(dZ, axis=0, keepdims=True)

    def update(self, lr):
        self.A -= lr * self.dA
        self.b -= lr * self.db

    def predict(self, X):
        return np.argmax(softmax(X @ self.A + self.b), axis=1)


class ModeleUneCoucheCachee:
    """
    MLP à 1 couche cachée.
    Z1 = X @ W1 + b1  →  H = ReLU(Z1)
    Z2 = H @ W2 + b2  →  P = softmax(Z2)

    # ReLU vs Sigmoid dans les couches cachées :
    # Sigmoid : sortie ∈ (0,1), dérivée ≤ 0.25 → gradient vanishing
    #           (les gradients s'écrasent en remontant les couches)
    # ReLU    : dérivée = 1 si x > 0, 0 sinon → pas de saturation pour x > 0
    #           beaucoup plus rapide à converger en pratique
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        # He initialization recommandée avec ReLU
        self.W1 = he_init(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = he_init(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        """ReLU'(x) = 1 si x > 0, sinon 0"""
        return (Z > 0).astype(np.float32)

    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.H  = self._relu(self.Z1)
        self.Z2 = self.H @ self.W2 + self.b2
        self.P  = softmax(self.Z2)
        return self.P

    def backward(self, Y):
        """
        Rétropropagation (chapitre 5 du cours) :
        delta_out = (P - Y) / n           ← gradient couche sortie
        dW2 = H^T @ delta_out
        delta_H = delta_out @ W2^T * ReLU'(Z1)   ← propagation + activation
        dW1 = X^T @ delta_H
        """
        n        = self.X.shape[0]
        dZ2      = (self.P - Y) / n
        self.dW2 = self.H.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        dH       = dZ2 @ self.W2.T
        dZ1      = dH * self._relu_deriv(self.Z1)
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def predict(self, X):
        H = self._relu(X @ self.W1 + self.b1)
        return np.argmax(softmax(H @ self.W2 + self.b2), axis=1)

    def get_hidden(self, X):
        """Représentations de la couche cachée (pour PCA / t-SNE)."""
        return self._relu(X @ self.W1 + self.b1)


class ModeleDeuxCouchesCachees:
    """
    MLP à 2 couches cachées.
    Z1 → H1 = ReLU(Z1) → Z2 → H2 = ReLU(Z2) → Z3 → P = softmax(Z3)

    # Gradient vanishing :
    # En rétropropagation, le gradient est multiplié à chaque couche par
    # la dérivée de l'activation. Avec Sigmoid (dérivée ≤ 0.25), il décroît
    # exponentiellement → les premières couches n'apprennent plus.
    # Avec ReLU (dérivée = 1 pour x > 0), ce problème est atténué.
    # Le CNN y remédie aussi via batch normalization et connexions résiduelles.
    """

    def __init__(self, input_dim=784, hidden1=128, hidden2=64, output_dim=10):
        self.W1 = he_init(input_dim, hidden1)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = he_init(hidden1, hidden2)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = he_init(hidden2, output_dim)
        self.b3 = np.zeros((1, output_dim))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0).astype(np.float32)

    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.H1 = self._relu(self.Z1)
        self.Z2 = self.H1 @ self.W2 + self.b2
        self.H2 = self._relu(self.Z2)
        self.Z3 = self.H2 @ self.W3 + self.b3
        self.P  = softmax(self.Z3)
        return self.P

    def backward(self, Y):
        n        = self.X.shape[0]
        dZ3      = (self.P - Y) / n
        self.dW3 = self.H2.T @ dZ3
        self.db3 = np.sum(dZ3, axis=0, keepdims=True)

        dH2      = dZ3 @ self.W3.T
        dZ2      = dH2 * self._relu_deriv(self.Z2)
        self.dW2 = self.H1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        dH1      = dZ2 @ self.W2.T
        dZ1      = dH1 * self._relu_deriv(self.Z1)
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self, lr):
        self.W1 -= lr * self.dW1;  self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2;  self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3;  self.b3 -= lr * self.db3

    def predict(self, X):
        H1 = self._relu(X @ self.W1 + self.b1)
        H2 = self._relu(H1 @ self.W2 + self.b2)
        return np.argmax(softmax(H2 @ self.W3 + self.b3), axis=1)

    def get_hidden(self, X):
        """Représentations de la dernière couche cachée."""
        H1 = self._relu(X @ self.W1 + self.b1)
        return self._relu(H1 @ self.W2 + self.b2)


# ============================================================
# 3. BOUCLE D'ENTRAÎNEMENT
# ============================================================

def entrainer(modele, x_train, y_train, x_test, y_test,
              lr=0.1, epochs=50, batch_size=256, verbose=True):
    """
    Mini-batch SGD.

    # Que se passe-t-il si lr trop grand ?
    # Les mises à jour dépassent le minimum → la loss oscille ou diverge.
    # Trop petit ? Convergence très lente, risque de bloquer dans un min local.
    # Règle empirique : commencer à 0.1, diviser par 10 si instabilité.
    """
    y_train_oh = one_hot(y_train, 10)
    n          = x_train.shape[0]
    hist       = {'loss': [], 'err_train': [], 'err_test': []}

    for ep in range(epochs):
        idx  = np.random.permutation(n)
        x_sh = x_train[idx]
        y_sh = y_train_oh[idx]

        for s in range(0, n, batch_size):
            Xb = x_sh[s:s+batch_size]
            Yb = y_sh[s:s+batch_size]
            modele.forward(Xb)
            modele.backward(Yb)
            modele.update(lr)

        P_tr   = modele.forward(x_train)
        loss   = cross_entropy(y_train_oh, P_tr)
        err_tr = 1 - accuracy(y_train, modele.predict(x_train))
        err_te = 1 - accuracy(y_test,  modele.predict(x_test))

        hist['loss'].append(loss)
        hist['err_train'].append(err_tr)
        hist['err_test'].append(err_te)

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{epochs} | Loss: {loss:.4f} | "
                  f"Err train: {err_tr:.4f} | Err test: {err_te:.4f}")

    return hist


# ============================================================
# 4. VISUALISATIONS
# ============================================================

def afficher_courbes(hist, titre, fichier=None):
    """Courbes de loss et taux d'erreur train/test."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist['loss'], color='steelblue')
    ax1.set_title('Fonction de coût (cross-entropy)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')

    ax2.plot(hist['err_train'], label='Train', color='steelblue')
    ax2.plot(hist['err_test'],  label='Test',  color='tomato')
    ax2.set_title("Taux d'erreur")
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Erreur')
    ax2.legend()

    fig.suptitle(titre, fontsize=13)
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : rapport/{fichier}")
    plt.show()


def afficher_matrice_confusion(y_true, y_pred, titre, fichier=None):
    """Matrice de confusion 10×10 avec heatmap."""
    mat = matrice_confusion(y_true, y_pred, n_classes=10)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Prédit', fontsize=11)
    ax.set_ylabel('Réel',   fontsize=11)
    ax.set_title(f'Matrice de confusion — {titre}', fontsize=12)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, mat[i, j], ha='center', va='center', fontsize=7,
                    color='white' if mat[i, j] > mat.max() / 2 else 'black')
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : rapport/{fichier}")
    plt.show()


def afficher_erreurs(modele, x_test, y_test, titre, fichier=None):
    """Affiche les 10 premières images mal classées."""
    y_pred  = modele.predict(x_test)
    erreurs = np.where(y_pred != y_test)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, idx in enumerate(erreurs[:10]):
        ax = axes[i // 5, i % 5]
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Réel:{y_test[idx]}  Prédit:{y_pred[idx]}', fontsize=9)
        ax.axis('off')
    fig.suptitle(titre, fontsize=12)
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : rapport/{fichier}")
    plt.show()


def afficher_pca(modele, x_test, y_test, titre, fichier=None):
    """Projection PCA 2D des représentations apprises."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  sklearn non disponible pour PCA.")
        return

    n_subset = 2000
    idx = np.random.choice(len(x_test), n_subset, replace=False)
    repres = modele.get_hidden(x_test[idx]) if hasattr(modele, 'get_hidden') else x_test[idx]

    pca  = PCA(n_components=2)
    proj = pca.fit_transform(repres)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y_test[idx],
                    cmap='tab10', alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax, label='Chiffre')
    ax.set_title(f'PCA 2D — {titre}')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : rapport/{fichier}")
    plt.show()
    print(f"  Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")


def afficher_tsne(modele, x_test, y_test, titre, fichier=None):
    """Projection t-SNE 2D des représentations apprises."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn non disponible pour t-SNE.")
        return

    n_subset = 1000
    idx    = np.random.choice(len(x_test), n_subset, replace=False)
    repres = modele.get_hidden(x_test[idx]) if hasattr(modele, 'get_hidden') else x_test[idx]

    print("  Calcul t-SNE (peut prendre 1-2 min)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(repres)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y_test[idx],
                    cmap='tab10', alpha=0.6, s=8)
    plt.colorbar(sc, ax=ax, label='Chiffre')
    ax.set_title(f't-SNE — {titre}')
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : rapport/{fichier}")
    plt.show()


# ============================================================
# 5. GRID SEARCH
# ============================================================

def grid_search(x_train, y_train, x_test, y_test):
    """Grid search sur architectures et learning rates."""
    configs = [
        {'nom': 'Linéaire',             'modele': ModeleLineaire(),                        'lr': 0.1},
    ]
    for h in [64, 128, 256]:
        for lr in [0.01, 0.1]:
            configs.append({
                'nom': f'MLP-1 h={h} lr={lr}',
                'modele': ModeleUneCoucheCachee(hidden_dim=h), 'lr': lr
            })
    for h1, h2 in [(128, 64), (256, 128)]:
        for lr in [0.01, 0.1]:
            configs.append({
                'nom': f'MLP-2 {h1}/{h2} lr={lr}',
                'modele': ModeleDeuxCouchesCachees(hidden1=h1, hidden2=h2), 'lr': lr
            })

    resultats  = []
    meilleur   = None

    for cfg in configs:
        print(f"\n  → {cfg['nom']}")
        hist   = entrainer(cfg['modele'], x_train, y_train, x_test, y_test,
                           lr=cfg['lr'], epochs=10, verbose=False)
        err_tr = hist['err_train'][-1]
        err_te = hist['err_test'][-1]
        print(f"    Err train: {err_tr:.4f}  Err test: {err_te:.4f}")
        resultats.append({'nom': cfg['nom'], 'err_train': err_tr, 'err_test': err_te})
        if meilleur is None or err_te < meilleur['err_test']:
            meilleur = resultats[-1]

    print("\n" + "=" * 55)
    print(f"  {'Architecture':<35} {'Train':>8} {'Test':>8}")
    print("  " + "-" * 52)
    for r in resultats:
        print(f"  {r['nom']:<35} {r['err_train']:>8.4f} {r['err_test']:>8.4f}")
    print(f"\n  → Meilleure : {meilleur['nom']}")
    print(f"    Err test  : {meilleur['err_test']:.4f}")
    return meilleur, resultats


# ============================================================
# 6. MENU PARTIE 1
# ============================================================

def menu_partie1():
    """Menu interactif de la Partie 1."""
    x_train, y_train, x_test, y_test = charger_mnist()

    while True:
        print("\n" + "=" * 48)
        print("   PARTIE 1 — Classification MNIST (NumPy)")
        print("=" * 48)
        print("  1  - Audit du code original")
        print("  2  - Modèle linéaire  (train + courbes)")
        print("  3  - MLP 1 couche cachée (h=128)")
        print("  4  - MLP 2 couches cachées (128/64)")
        print("  5  - Comparer les 3 modèles")
        print("  6  - Matrice de confusion")
        print("  7  - Images mal classées")
        print("  8  - Visualisation PCA 2D")
        print("  9  - Visualisation t-SNE")
        print("  10 - Grid search complet")
        print("  0  - Retour au menu principal")
        print("-" * 48)

        choix = input("Choix : ").strip()

        if choix == '1':
            audit_partie1()

        elif choix == '2':
            modele = ModeleLineaire()
            hist   = entrainer(modele, x_train, y_train, x_test, y_test,
                               lr=0.1, epochs=50)
            afficher_courbes(hist, 'Modèle Linéaire MNIST', 'p1_lineaire_courbes.png')
            print(f"\n  Err train : {hist['err_train'][-1]:.4f}")
            print(f"  Err test  : {hist['err_test'][-1]:.4f}")

        elif choix == '3':
            modele = ModeleUneCoucheCachee(hidden_dim=128)
            hist   = entrainer(modele, x_train, y_train, x_test, y_test,
                               lr=0.1, epochs=50)
            afficher_courbes(hist, 'MLP 1 couche (h=128)', 'p1_mlp1_courbes.png')
            print(f"\n  Err train : {hist['err_train'][-1]:.4f}")
            print(f"  Err test  : {hist['err_test'][-1]:.4f}")

        elif choix == '4':
            modele = ModeleDeuxCouchesCachees(hidden1=128, hidden2=64)
            hist   = entrainer(modele, x_train, y_train, x_test, y_test,
                               lr=0.1, epochs=50)
            afficher_courbes(hist, 'MLP 2 couches (128/64)', 'p1_mlp2_courbes.png')
            print(f"\n  Err train : {hist['err_train'][-1]:.4f}")
            print(f"  Err test  : {hist['err_test'][-1]:.4f}")

        elif choix == '5':
            configs = [
                ('Linéaire',       ModeleLineaire(),                       0.1),
                ('MLP-1 (h=128)', ModeleUneCoucheCachee(hidden_dim=128),   0.1),
                ('MLP-2 (128/64)', ModeleDeuxCouchesCachees(128, 64),       0.1),
            ]
            resultats = []
            for nom, modele, lr in configs:
                print(f"\nEntraînement : {nom}")
                hist = entrainer(modele, x_train, y_train, x_test, y_test,
                                 lr=lr, epochs=50, verbose=False)
                print(f"  Err train: {hist['err_train'][-1]:.4f}  "
                      f"Err test: {hist['err_test'][-1]:.4f}")
                resultats.append((nom, hist))

            fig, ax = plt.subplots(figsize=(10, 5))
            for nom, hist in resultats:
                ax.plot(hist['err_test'], label=nom)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Erreur test')
            ax.set_title('Comparaison architectures — MNIST')
            ax.legend(); plt.tight_layout()
            plt.savefig('rapport/p1_comparaison.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("  Figure sauvegardée : rapport/p1_comparaison.png")

        elif choix == '6':
            print("Entraînement MLP-2 pour matrice de confusion...")
            modele = ModeleDeuxCouchesCachees(128, 64)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            y_pred = modele.predict(x_test)
            afficher_matrice_confusion(y_test, y_pred,
                                       'MLP-2 MNIST', 'p1_confusion.png')

        elif choix == '7':
            print("Entraînement MLP-2 pour images mal classées...")
            modele = ModeleDeuxCouchesCachees(128, 64)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            afficher_erreurs(modele, x_test, y_test,
                             'MLP-2 — Images mal classées', 'p1_erreurs.png')

        elif choix == '8':
            print("Entraînement MLP-1 pour PCA...")
            modele = ModeleUneCoucheCachee(hidden_dim=128)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            afficher_pca(modele, x_test, y_test, 'MLP-1 MNIST', 'p1_pca.png')

        elif choix == '9':
            print("Entraînement MLP-1 pour t-SNE...")
            modele = ModeleUneCoucheCachee(hidden_dim=128)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            afficher_tsne(modele, x_test, y_test, 'MLP-1 MNIST', 'p1_tsne.png')

        elif choix == '10':
            grid_search(x_train, y_train, x_test, y_test)

        elif choix == '0':
            break

        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_partie1()
