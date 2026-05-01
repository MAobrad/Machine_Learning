"""
part1_mnist.py - Partie 1 : Classification MNIST en NumPy pur
On implemente tout a la main : forward pass, retropropagation, descente de gradient.
Trois architectures testees : modele lineaire, MLP 1 couche cachee, MLP 2 couches cachees.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Fixe la graine aleatoire pour avoir des resultats reproductibles a chaque run
np.random.seed(42)
os.makedirs('rapport', exist_ok=True)

from utils import (
    one_hot, softmax, cross_entropy, accuracy,
    xavier_init, he_init, matrice_confusion
)


# ============================================================
# 1. CHARGEMENT MNIST
# ============================================================

def charger_mnist():
    """Charge et prepare les donnees MNIST (60 000 train / 10 000 test)."""
    print("Chargement de MNIST...")
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except ImportError:
        try:
            from torchvision.datasets import MNIST
            train_ds = MNIST('.', train=True, download=True)
            test_ds = MNIST('.', train=False, download=True)
            x_train = np.array(train_ds.data)
            y_train = np.array(train_ds.targets)
            x_test = np.array(test_ds.data)
            y_test = np.array(test_ds.targets)
        except ImportError:
            raise ImportError("Installer tensorflow ou torchvision.")

    # Chaque image 28x28 pixels : on l'aplatit en vecteur de 784 valeurs
    x_train = x_train.reshape(x_train.shape[0], 784).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 784).astype(np.float32)

    # Normalisation : valeurs entre 0 et 1 au lieu de 0 a 255
    x_train /= 255.0
    x_test /= 255.0

    print(f"  x_train : {x_train.shape}  y_train : {y_train.shape}")
    print(f"  x_test  : {x_test.shape}   y_test  : {y_test.shape}")
    return x_train, y_train, x_test, y_test


# ============================================================
# 2. MODELES NumPy PURS
# ============================================================

class ModeleLineaire:
    """
    Modele lineaire multi-classe : o = X @ A + b, P = softmax(o)
    Pas de couche cachee, c'est la version la plus simple.
    Convention : A est de forme (input_dim, output_dim) pour pouvoir ecrire X @ A directement.
    """

    def __init__(self, input_dim=784, output_dim=10):
        # Xavier recommande pour les couches sans activation non-lineaire
        self.A = xavier_init(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        # On calcule les scores (logits) puis on les convertit en probabilites via softmax
        self.X = X
        self.logits = X @ self.A + self.b
        self.P = softmax(self.logits)
        return self.P

    def backward(self, Y):
        # Formule analytique du gradient pour cross-entropy + softmax combines :
        # dL/do = (P - Y) / n  : resultat direct, pas besoin de chain rule separee
        # dL/dA = X^T @ dZ
        # dL/db = somme de dZ sur les exemples du batch
        n = self.X.shape[0]
        dZ = (self.P - Y) / n
        self.dA = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

    def update(self, lr):
        self.A -= lr * self.dA
        self.b -= lr * self.db

    def predict(self, X):
        return np.argmax(softmax(X @ self.A + self.b), axis=1)


class ModeleUneCoucheCachee:
    """
    MLP 1 couche cachee :
    Z1 = X @ W1 + b1, H = ReLU(Z1), Z2 = H @ W2 + b2, P = softmax(Z2)

    Pourquoi ReLU plutot que Sigmoid dans la couche cachee ?
    - Sigmoid : sortie dans (0,1), sa derivee vaut au maximum 0.25
      En retropropagation, le gradient est multiplie par <= 0.25 a chaque couche,
      donc avec plusieurs couches il devient quasi nul (gradient vanishing)
    - ReLU : derivee = 1 si x > 0, 0 sinon
      Pas de saturation pour les valeurs positives, convergence beaucoup plus rapide
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        # He initialization recommandee avec ReLU
        self.W1 = he_init(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = he_init(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        # Derivee de ReLU : 1 la ou Z > 0, 0 partout ailleurs
        return (Z > 0).astype(np.float32)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.H = self._relu(self.Z1)
        self.Z2 = self.H @ self.W2 + self.b2
        self.P = softmax(self.Z2)
        return self.P

    def backward(self, Y):
        # Retropropagation couche par couche, du dernier vers le premier :
        # 1. Gradient de la couche de sortie
        n = self.X.shape[0]
        dZ2 = (self.P - Y) / n
        self.dW2 = self.H.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        # 2. On propage vers la couche cachee en passant par la derivee de ReLU
        dH = dZ2 @ self.W2.T
        dZ1 = dH * self._relu_deriv(self.Z1)
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
        # Retourne les representations de la couche cachee (utilise pour PCA / t-SNE)
        return self._relu(X @ self.W1 + self.b1)


class ModeleDeuxCouchesCachees:
    """
    MLP 2 couches cachees :
    Z1, H1 = ReLU(Z1), Z2, H2 = ReLU(Z2), Z3, P = softmax(Z3)

    Note sur le gradient vanishing :
    En retropropagation, le gradient est multiplie a chaque couche par la derivee de l'activation.
    Avec Sigmoid (derivee <= 0.25), il decroit exponentiellement sur plusieurs couches,
    les premieres couches n'apprennent pratiquement plus rien.
    Avec ReLU, ce probleme est largement attenue puisque la derivee vaut 1 pour x > 0.
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
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.H1 = self._relu(self.Z1)
        self.Z2 = self.H1 @ self.W2 + self.b2
        self.H2 = self._relu(self.Z2)
        self.Z3 = self.H2 @ self.W3 + self.b3
        self.P = softmax(self.Z3)
        return self.P

    def backward(self, Y):
        # Meme principe que MLP-1 mais avec une couche de plus a retropropager
        n = self.X.shape[0]
        dZ3 = (self.P - Y) / n
        self.dW3 = self.H2.T @ dZ3
        self.db3 = np.sum(dZ3, axis=0, keepdims=True)

        dH2 = dZ3 @ self.W3.T
        dZ2 = dH2 * self._relu_deriv(self.Z2)
        self.dW2 = self.H1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        dH1 = dZ2 @ self.W2.T
        dZ1 = dH1 * self._relu_deriv(self.Z1)
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

    def predict(self, X):
        H1 = self._relu(X @ self.W1 + self.b1)
        H2 = self._relu(H1 @ self.W2 + self.b2)
        return np.argmax(softmax(H2 @ self.W3 + self.b3), axis=1)

    def get_hidden(self, X):
        # Representations de la derniere couche cachee (pour PCA / t-SNE)
        H1 = self._relu(X @ self.W1 + self.b1)
        return self._relu(H1 @ self.W2 + self.b2)


# ============================================================
# 3. BOUCLE D'ENTRAINEMENT
# ============================================================

def entrainer(modele, x_train, y_train, x_test, y_test,
              lr=0.1, epochs=50, batch_size=256, verbose=True):
    """
    Mini-batch SGD : on melange les donnees a chaque epoch, puis on fait des mises
    a jour par petits groupes (batches) plutot que sur tout le dataset d'un coup.

    Pourquoi un learning rate ni trop grand ni trop petit ?
    - Trop grand : les mises a jour depassent le minimum, la loss oscille ou diverge
    - Trop petit : convergence tres lente, on peut rester bloque dans un minimum local
    - Bonne pratique : commencer a 0.1 et diviser par 10 si on voit une instabilite
    """
    y_train_oh = one_hot(y_train, 10)
    n = x_train.shape[0]
    hist = {'loss': [], 'err_train': [], 'err_test': []}

    for ep in range(epochs):
        # Melange aleatoire a chaque epoch pour que le modele ne memorise pas l'ordre
        idx = np.random.permutation(n)
        x_sh = x_train[idx]
        y_sh = y_train_oh[idx]

        for s in range(0, n, batch_size):
            Xb = x_sh[s:s+batch_size]
            Yb = y_sh[s:s+batch_size]
            modele.forward(Xb)
            modele.backward(Yb)
            modele.update(lr)

        P_tr = modele.forward(x_train)
        loss = cross_entropy(y_train_oh, P_tr)
        err_tr = 1 - accuracy(y_train, modele.predict(x_train))
        err_te = 1 - accuracy(y_test, modele.predict(x_test))

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
    # On trace la loss et le taux d'erreur train/test en fonction des epochs
    # Si la courbe test decroche vers le haut : overfitting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist['loss'], color='steelblue')
    ax1.set_title('Fonction de cout (cross-entropy)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(hist['err_train'], label='Train', color='steelblue')
    ax2.plot(hist['err_test'], label='Test', color='tomato')
    ax2.set_title("Taux d'erreur")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Erreur')
    ax2.legend()

    fig.suptitle(titre, fontsize=13)
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardee : rapport/{fichier}")
    plt.show()


def afficher_matrice_confusion(y_true, y_pred, titre, fichier=None):
    # Heatmap 10x10 : chaque case (i,j) = nombre d'images du chiffre i classees comme j
    # La diagonale = les bonnes predictions. Les cases hors diagonale = les erreurs
    mat = matrice_confusion(y_true, y_pred, n_classes=10)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predit', fontsize=11)
    ax.set_ylabel('Reel', fontsize=11)
    ax.set_title(f'Matrice de confusion - {titre}', fontsize=12)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, mat[i, j], ha='center', va='center', fontsize=7,
                    color='white' if mat[i, j] > mat.max() / 2 else 'black')
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardee : rapport/{fichier}")
    plt.show()


def afficher_erreurs(modele, x_test, y_test, titre, fichier=None):
    # On affiche les 10 premieres images que le modele a mal classees
    # Utile pour comprendre visuellement quels cas posent probleme
    y_pred = modele.predict(x_test)
    erreurs = np.where(y_pred != y_test)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, idx in enumerate(erreurs[:10]):
        ax = axes[i // 5, i % 5]
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Reel:{y_test[idx]}  Predit:{y_pred[idx]}', fontsize=9)
        ax.axis('off')
    fig.suptitle(titre, fontsize=12)
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardee : rapport/{fichier}")
    plt.show()


def afficher_pca(modele, x_test, y_test, titre, fichier=None):
    # PCA = reduction de dimension lineaire, on projette les representations sur 2 axes
    # Si les chiffres forment des clusters bien separes, le modele a bien appris a les distinguer
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  sklearn non disponible pour PCA.")
        return

    n_subset = 2000
    idx = np.random.choice(len(x_test), n_subset, replace=False)
    repres = modele.get_hidden(x_test[idx]) if hasattr(modele, 'get_hidden') else x_test[idx]

    pca = PCA(n_components=2)
    proj = pca.fit_transform(repres)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y_test[idx],
                    cmap='tab10', alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax, label='Chiffre')
    ax.set_title(f'PCA 2D - {titre}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardee : rapport/{fichier}")
    plt.show()
    print(f"  Variance expliquee : {pca.explained_variance_ratio_.sum():.2%}")


def afficher_tsne(modele, x_test, y_test, titre, fichier=None):
    # t-SNE = reduction de dimension non-lineaire, meilleure que PCA pour visualiser des clusters
    # Plus lente a calculer (1-2 min) mais les groupes de chiffres apparaissent plus clairement
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn non disponible pour t-SNE.")
        return

    n_subset = 1000
    idx = np.random.choice(len(x_test), n_subset, replace=False)
    repres = modele.get_hidden(x_test[idx]) if hasattr(modele, 'get_hidden') else x_test[idx]

    print("  Calcul t-SNE (peut prendre 1-2 min)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(repres)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y_test[idx],
                    cmap='tab10', alpha=0.6, s=8)
    plt.colorbar(sc, ax=ax, label='Chiffre')
    ax.set_title(f't-SNE - {titre}')
    plt.tight_layout()
    if fichier:
        plt.savefig(f'rapport/{fichier}', dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardee : rapport/{fichier}")
    plt.show()


# ============================================================
# 5. GRID SEARCH
# ============================================================

def grid_search(x_train, y_train, x_test, y_test):
    # On teste toutes les combinaisons architectures x learning rates
    # pour trouver empiriquement la meilleure configuration sur le test set
    configs = [
        {'nom': 'Lineaire', 'modele': ModeleLineaire(), 'lr': 0.1},
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

    resultats = []
    meilleur = None

    for cfg in configs:
        print(f"\n  {cfg['nom']}")
        hist = entrainer(cfg['modele'], x_train, y_train, x_test, y_test,
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
    print(f"\n  Meilleure : {meilleur['nom']}")
    print(f"  Err test  : {meilleur['err_test']:.4f}")
    return meilleur, resultats


# ============================================================
# 6. MENU PARTIE 1
# ============================================================

def menu_partie1():
    """Menu interactif de la Partie 1."""
    x_train, y_train, x_test, y_test = charger_mnist()

    while True:
        print("\n" + "=" * 48)
        print("   PARTIE 1 - Classification MNIST (NumPy)")
        print("=" * 48)
        print("  1  - Modele lineaire  (train + courbes)")
        print("  2  - MLP 1 couche cachee (h=128)")
        print("  3  - MLP 2 couches cachees (128/64)")
        print("  4  - Comparer les 3 modeles")
        print("  5  - Matrice de confusion")
        print("  6  - Images mal classees")
        print("  7  - Visualisation PCA 2D")
        print("  8  - Visualisation t-SNE")
        print("  9  - Grid search complet")
        print("  0  - Retour au menu principal")
        print("-" * 48)

        choix = input("Choix : ").strip()

        if choix == '1':
            modele = ModeleLineaire()
            hist = entrainer(modele, x_train, y_train, x_test, y_test, lr=0.1, epochs=50)
            afficher_courbes(hist, 'Modele Lineaire MNIST', 'p1_lineaire_courbes.png')
            print(f"\n  Err train : {hist['err_train'][-1]:.4f}")
            print(f"  Err test  : {hist['err_test'][-1]:.4f}")

        elif choix == '2':
            modele = ModeleUneCoucheCachee(hidden_dim=128)
            hist = entrainer(modele, x_train, y_train, x_test, y_test, lr=0.1, epochs=50)
            afficher_courbes(hist, 'MLP 1 couche (h=128)', 'p1_mlp1_courbes.png')
            print(f"\n  Err train : {hist['err_train'][-1]:.4f}")
            print(f"  Err test  : {hist['err_test'][-1]:.4f}")

        elif choix == '3':
            modele = ModeleDeuxCouchesCachees(hidden1=128, hidden2=64)
            hist = entrainer(modele, x_train, y_train, x_test, y_test, lr=0.1, epochs=50)
            afficher_courbes(hist, 'MLP 2 couches (128/64)', 'p1_mlp2_courbes.png')
            print(f"\n  Err train : {hist['err_train'][-1]:.4f}")
            print(f"  Err test  : {hist['err_test'][-1]:.4f}")

        elif choix == '4':
            configs = [
                ('Lineaire', ModeleLineaire(), 0.1),
                ('MLP-1 (h=128)', ModeleUneCoucheCachee(hidden_dim=128), 0.1),
                ('MLP-2 (128/64)', ModeleDeuxCouchesCachees(128, 64), 0.1),
            ]
            resultats = []
            for nom, modele, lr in configs:
                print(f"\nEntrainement : {nom}")
                hist = entrainer(modele, x_train, y_train, x_test, y_test,
                                 lr=lr, epochs=50, verbose=False)
                print(f"  Err train: {hist['err_train'][-1]:.4f}  "
                      f"Err test: {hist['err_test'][-1]:.4f}")
                resultats.append((nom, hist))

            fig, ax = plt.subplots(figsize=(10, 5))
            for nom, hist in resultats:
                ax.plot(hist['err_test'], label=nom)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Erreur test')
            ax.set_title('Comparaison architectures - MNIST')
            ax.legend()
            plt.tight_layout()
            plt.savefig('rapport/p1_comparaison.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("  Figure sauvegardee : rapport/p1_comparaison.png")

        elif choix == '5':
            print("Entrainement MLP-2 pour matrice de confusion...")
            modele = ModeleDeuxCouchesCachees(128, 64)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            y_pred = modele.predict(x_test)
            afficher_matrice_confusion(y_test, y_pred, 'MLP-2 MNIST', 'p1_confusion.png')

        elif choix == '6':
            print("Entrainement MLP-2 pour images mal classees...")
            modele = ModeleDeuxCouchesCachees(128, 64)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            afficher_erreurs(modele, x_test, y_test,
                             'MLP-2 - Images mal classees', 'p1_erreurs.png')

        elif choix == '7':
            print("Entrainement MLP-1 pour PCA...")
            modele = ModeleUneCoucheCachee(hidden_dim=128)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            afficher_pca(modele, x_test, y_test, 'MLP-1 MNIST', 'p1_pca.png')

        elif choix == '8':
            print("Entrainement MLP-1 pour t-SNE...")
            modele = ModeleUneCoucheCachee(hidden_dim=128)
            entrainer(modele, x_train, y_train, x_test, y_test,
                      lr=0.1, epochs=30, verbose=False)
            afficher_tsne(modele, x_test, y_test, 'MLP-1 MNIST', 'p1_tsne.png')

        elif choix == '9':
            grid_search(x_train, y_train, x_test, y_test)

        elif choix == '0':
            break

        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_partie1()
