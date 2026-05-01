"""
part2_cifar10.py - Partie 2 : CIFAR-10 + Convolutions + CNN PyTorch
Etudes preliminaires avec MLP NumPy, demonstration des filtres, CNN complet.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('rapport', exist_ok=True)

from utils import (
    one_hot, softmax, cross_entropy, accuracy,
    convolve2d, convolve2d_color, max_pooling2x2, matrice_confusion
)
from part1_mnist import (
    ModeleLineaire, ModeleUneCoucheCachee, ModeleDeuxCouchesCachees, entrainer
)


# ============================================================
# 1. CHARGEMENT CIFAR-10
# ============================================================

CLASSES_CIFAR = ['avion', 'auto', 'oiseau', 'chat', 'cerf',
                  'chien', 'grenouille', 'cheval', 'bateau', 'camion']


def charger_cifar10():
    """Charge CIFAR-10 via tensorflow.keras ou torchvision."""
    print("Chargement de CIFAR-10...")
    try:
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    except ImportError:
        try:
            from torchvision.datasets import CIFAR10
            train_ds = CIFAR10('.', train=True, download=True)
            test_ds = CIFAR10('.', train=False, download=True)
            x_train = np.array(train_ds.data)
            y_train = np.array(train_ds.targets)
            x_test = np.array(test_ds.data)
            y_test = np.array(test_ds.targets)
        except ImportError:
            raise ImportError("Installer tensorflow ou torchvision.")

    print(f"  x_train : {x_train.shape}  x_test : {x_test.shape}")
    print(f"  Classes : {', '.join(f'{i}={c}' for i, c in enumerate(CLASSES_CIFAR))}")
    return x_train, y_train, x_test, y_test


# ============================================================
# 2. PREPARATION DES DONNEES
# ============================================================

def rgb_vers_gris(images):
    # Formule standard ITU-R BT.601 pour convertir RGB en niveaux de gris
    # Les coefficients (0.299, 0.587, 0.114) refletent la sensibilite de l'oeil humain :
    # on est plus sensible au vert qu'au rouge, et tres peu sensible au bleu
    return (0.299 * images[:, :, :, 0] +
            0.587 * images[:, :, :, 1] +
            0.114 * images[:, :, :, 2])


def preparer_gris(x_train, x_test):
    """Images CIFAR (N,32,32,3) vers vecteurs niveaux de gris (N,1024) dans [0,1]."""
    g_tr = rgb_vers_gris(x_train).reshape(-1, 1024).astype(np.float32) / 255.0
    g_te = rgb_vers_gris(x_test).reshape(-1, 1024).astype(np.float32) / 255.0
    return g_tr, g_te


def preparer_couleur(x_train, x_test):
    """Images CIFAR (N,32,32,3) vers vecteurs couleur aplatis (N,3072) dans [0,1]."""
    c_tr = x_train.reshape(-1, 3072).astype(np.float32) / 255.0
    c_te = x_test.reshape(-1, 3072).astype(np.float32) / 255.0
    return c_tr, c_te


# ============================================================
# 3. ETUDES PRELIMINAIRES A ET B
# ============================================================

def etudes_preliminaires(x_train, y_train, x_test, y_test):
    """
    Etude A : images en niveaux de gris (1024 dims)
    Etude B : images couleur aplaties  (3072 dims)
    On applique les 3 architectures MLP NumPy sur CIFAR-10 pour comparer.

    Pourquoi CIFAR-10 est plus difficile que MNIST ?
    - Les images sont en couleur (32x32x3) avec une grande variabilite intra-classe :
      un chien peut etre vu de face, de profil, debout, couche...
    - Un MLP lineaire n'a pas d'invariance spatiale, il ne reconnait pas un objet decale
    - MNIST : chiffres centres sur fond blanc, tres peu de variabilite, tache plus simple
    - La couleur ajoute de l'information mais aussi plus de parametres a apprendre
    """
    print("\n" + "=" * 60)
    print("  ETUDES PRELIMINAIRES - MLP NumPy sur CIFAR-10")
    print("=" * 60)

    g_tr, g_te = preparer_gris(x_train, x_test)
    c_tr, c_te = preparer_couleur(x_train, x_test)

    configs = [
        ('Lineaire', 'gris', ModeleLineaire(input_dim=1024), g_tr, g_te, 0.1),
        ('MLP-1 (256)', 'gris', ModeleUneCoucheCachee(1024, 256), g_tr, g_te, 0.1),
        ('MLP-2 (256/128)', 'gris', ModeleDeuxCouchesCachees(1024, 256, 128), g_tr, g_te, 0.1),
        ('Lineaire', 'couleur', ModeleLineaire(input_dim=3072), c_tr, c_te, 0.1),
        ('MLP-1 (256)', 'couleur', ModeleUneCoucheCachee(3072, 256), c_tr, c_te, 0.1),
        ('MLP-2 (256/128)', 'couleur', ModeleDeuxCouchesCachees(3072, 256, 128), c_tr, c_te, 0.1),
    ]

    resultats = []
    for nom, inp, modele, xtr, xte, lr in configs:
        print(f"\n  {nom} ({inp})")
        hist = entrainer(modele, xtr, y_train, xte, y_test,
                         lr=lr, epochs=10, batch_size=256, verbose=False)
        err_tr = hist['err_train'][-1]
        err_te = hist['err_test'][-1]
        print(f"    Err train: {err_tr:.4f}  Err test: {err_te:.4f}")
        resultats.append((nom, inp, err_tr, err_te))

    print("\n" + "=" * 65)
    print(f"  {'Architecture':<22} {'Input':<9} {'Err train':>10} {'Err test':>10}")
    print("  " + "-" * 54)
    for nom, inp, err_tr, err_te in resultats:
        print(f"  {nom:<22} {inp:<9} {err_tr:>10.4f} {err_te:>10.4f}")

    print("\n  Benchmarks CIFAR-10 (etat de l'art) :")
    print("    Deep Belief Networks  (2010) : 21.1%")
    print("    Maxout Networks       (2013) :  9.38%")
    print("    ViT Vision Transformer(2021) :  0.5%")

    # Graphique comparatif gris vs couleur pour les 3 architectures
    labels = [f"{n}\n({i})" for n, i, _, _ in resultats]
    errs_te = [r[3] for r in resultats]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, errs_te, color=['steelblue']*3 + ['tomato']*3)
    ax.set_ylabel("Taux d'erreur test")
    ax.set_title("Etudes preliminaires CIFAR-10 - MLP NumPy")
    ax.axhline(y=0.211, color='gray', linestyle='--', label='DBN 2010 (21.1%)')
    ax.axhline(y=0.0938, color='green', linestyle='--', label='Maxout 2013 (9.38%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('rapport/p2_etudes_preliminaires.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : rapport/p2_etudes_preliminaires.png")

    return resultats


# ============================================================
# 4. DEMONSTRATION - 6 FILTRES DE CONVOLUTION
# ============================================================

# Les 6 filtres qu'on va appliquer sur une image CIFAR en niveaux de gris
FILTRES = {
    'K1 - Flou (moyenne)':      (1/9) * np.ones((3, 3)),
    'K2 - Nettete':             np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    'K3 - Bords verticaux':     np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float),
    'K4 - Gradient horizontal': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    'K5 - Sobel horizontal':    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    'K6 - Sobel diagonal':      np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}


def demo_filtres(x_train_raw):
    """
    Applique les 6 filtres sur une image CIFAR et affiche le resultat.

    Effet de chaque filtre :
    K1 (Flou)       : fait la moyenne des pixels voisins, lisse le bruit et les details fins
    K2 (Nettete)    : amplifie les differences locales, les contours ressortent davantage
    K3 (Bords vert.): detecte les transitions entre zones claires/sombres de haut en bas
    K4 (Gradient H.): detecte les variations de gauche a droite (bords verticaux)
    K5 (Sobel H.)   : version plus robuste de K4, pondere plus le centre de la ligne
    K6 (Sobel diag.): detecte les bords orientes en diagonale
    """
    img_rgb = x_train_raw[np.random.randint(len(x_train_raw))]
    img_gray = rgb_vers_gris(img_rgb[np.newaxis])[0] / 255.0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original (couleur)')
    axes[0, 0].axis('off')

    for i, (nom, filtre) in enumerate(FILTRES.items()):
        row, col = (i + 1) // 4, (i + 1) % 4
        filtered = convolve2d(img_gray, filtre)
        axes[row, col].imshow(np.clip(filtered, 0, 1), cmap='gray')
        axes[row, col].set_title(nom, fontsize=9)
        axes[row, col].axis('off')

    axes[1, 3].imshow(img_gray, cmap='gray')
    axes[1, 3].set_title('Original (gris)')
    axes[1, 3].axis('off')

    plt.suptitle('Effet des 6 filtres de convolution sur une image CIFAR-10', fontsize=12)
    plt.tight_layout()
    plt.savefig('rapport/p2_filtres.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : rapport/p2_filtres.png")

    print("\n  Commentaires des filtres :")
    commentaires = {
        'K1 - Flou (moyenne)':      "lisse l'image, reduit le bruit",
        'K2 - Nettete':             "renforce les contours et details fins",
        'K3 - Bords verticaux':     "detecte les bords horizontaux (variation verticale)",
        'K4 - Gradient horizontal': "accentue les transitions gauche-droite",
        'K5 - Sobel horizontal':    "filtre de bord robuste oriente horizontalement",
        'K6 - Sobel diagonal':      "detecte les bords en diagonale",
    }
    for nom in FILTRES:
        print(f"    {nom} : {commentaires[nom]}")


def demo_convolution_couleur(x_train_raw):
    """Demo convolution 3 canaux vers 1 feature map : on applique Sobel sur chaque canal RGB."""
    img = x_train_raw[0].astype(np.float32) / 255.0

    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernels_rgb = np.stack([sobel, sobel, sobel], axis=-1)   # meme filtre sur les 3 canaux
    feature_map = convolve2d_color(img, kernels_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Image RGB originale')
    axes[0].axis('off')
    axes[1].imshow(img.mean(axis=2), cmap='gray')
    axes[1].set_title('Niveaux de gris')
    axes[1].axis('off')
    im = axes[2].imshow(np.clip(feature_map, 0, None), cmap='hot')
    axes[2].set_title('Feature map (Sobel couleur)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    plt.suptitle('Convolution couleur vers feature map 2D', fontsize=12)
    plt.tight_layout()
    plt.savefig('rapport/p2_conv_couleur.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : rapport/p2_conv_couleur.png")


def demo_max_pooling(x_train_raw):
    """Demo Max-Pooling 2x2 : on montre la reduction de resolution apres un filtre Sobel."""
    img = x_train_raw[0].astype(np.float32) / 255.0
    img_gray = rgb_vers_gris(img[np.newaxis])[0]

    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    feat = convolve2d(img_gray, sobel)
    pooled = max_pooling2x2(np.clip(feat, 0, None))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title(f'Original ({img_gray.shape[0]}x{img_gray.shape[1]})')
    axes[0].axis('off')
    axes[1].imshow(np.clip(feat, 0, None), cmap='hot')
    axes[1].set_title(f'Feature map ({feat.shape[0]}x{feat.shape[1]})')
    axes[1].axis('off')
    axes[2].imshow(pooled, cmap='hot')
    axes[2].set_title(f'Apres MaxPool ({pooled.shape[0]}x{pooled.shape[1]})')
    axes[2].axis('off')
    plt.suptitle('Max-Pooling 2x2 - reduction spatiale par 2', fontsize=12)
    plt.tight_layout()
    plt.savefig('rapport/p2_maxpooling.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : rapport/p2_maxpooling.png")


# ============================================================
# 5. CNN PYTORCH (Option B)
# ============================================================

def entrainer_cnn_cifar10(x_train_raw, y_train, x_test_raw, y_test):
    """
    CNN complet CIFAR-10 avec PyTorch.
    Architecture : Conv1(3->64), BN, ReLU, Conv2(64->64), BN, ReLU, MaxPool,
                   Conv3, BN, ReLU, MaxPool, Conv4, BN, ReLU, Flatten, Dropout, FC(10)

    Qu'est-ce que l'overfitting et comment le Dropout le limite ?
    Overfitting = le modele memorise le train set au lieu de generaliser (train acc >> test acc).
    Dropout : a chaque forward pass, on desactive aleatoirement p% des neurones.
    Le reseau ne peut plus dependre d'un seul neurone, il apprend des representations
    plus robustes et distribuees, ce qui ameliore la generalisation sur le test set.
    On detecte l'overfitting si acc_train - acc_test depasse ~10%.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("  PyTorch non installe. Executer :")
        print("  venv/bin/pip install torch torchvision")
        return None, None

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")

    # PyTorch attend le format (N, C, H, W), on transpose depuis (N, H, W, C)
    x_tr = torch.tensor(x_train_raw.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    x_te = torch.tensor(x_test_raw.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    y_tr = torch.tensor(y_train, dtype=torch.long)
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=256, shuffle=False, num_workers=0)

    class CNN_CIFAR10(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)    # 32 -> 16
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)    # 16 -> 8
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(64 * 8 * 8, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool2(x)
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.drop(torch.flatten(x, 1))
            return self.fc(x)   # logits bruts

    model = CNN_CIFAR10().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametres du modele : {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # CosineAnnealingLR diminue progressivement le learning rate en suivant une courbe en cosinus
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 20
    hist = {'loss_tr': [], 'loss_te': [], 'acc_tr': [], 'acc_te': []}

    for ep in range(epochs):
        model.train()
        loss_s, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_s += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        loss_tr = loss_s / total
        acc_tr = correct / total

        model.eval()
        loss_s2, correct2, total2 = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss_s2 += criterion(out, yb).item() * xb.size(0)
                correct2 += (out.argmax(1) == yb).sum().item()
                total2 += xb.size(0)
        loss_te = loss_s2 / total2
        acc_te = correct2 / total2

        hist['loss_tr'].append(loss_tr)
        hist['loss_te'].append(loss_te)
        hist['acc_tr'].append(acc_tr)
        hist['acc_te'].append(acc_te)

        print(f"  Epoch {ep+1:2d}/{epochs} | "
              f"Loss {loss_tr:.4f}/{loss_te:.4f} | "
              f"Acc {acc_tr:.4f}/{acc_te:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist['loss_tr'], label='Train')
    ax1.plot(hist['loss_te'], label='Test')
    ax1.set_title('Loss CNN CIFAR-10')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(hist['acc_tr'], label='Train')
    ax2.plot(hist['acc_te'], label='Test')
    ax2.set_title('Accuracy CNN CIFAR-10')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('rapport/p2_cnn_courbes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : rapport/p2_cnn_courbes.png")

    overfit = hist['acc_tr'][-1] - hist['acc_te'][-1]
    print(f"\n  Acc train finale : {hist['acc_tr'][-1]:.4f}")
    print(f"  Acc test finale  : {hist['acc_te'][-1]:.4f}")
    if overfit > 0.10:
        print(f"  Overfitting detecte (gap train-test = {overfit:.4f})")
    else:
        print(f"  Pas d'overfitting majeur (gap = {overfit:.4f})")

    return model, hist


# ============================================================
# 6. MENU PARTIE 2
# ============================================================

def menu_partie2():
    """Menu interactif de la Partie 2."""
    x_train, y_train, x_test, y_test = charger_cifar10()

    while True:
        print("\n" + "=" * 52)
        print("   PARTIE 2 - CIFAR-10 + Convolutions + CNN")
        print("=" * 52)
        print("  1  - Etudes preliminaires A et B (MLP NumPy)")
        print("  2  - Demo 6 filtres de convolution (N&B)")
        print("  3  - Demo convolution couleur (3 canaux)")
        print("  4  - Demo Max-Pooling 2x2")
        print("  5  - CNN PyTorch complet (Option B)")
        print("  0  - Retour au menu principal")
        print("-" * 52)

        choix = input("Choix : ").strip()

        if choix == '1':
            etudes_preliminaires(x_train, y_train, x_test, y_test)
        elif choix == '2':
            demo_filtres(x_train)
        elif choix == '3':
            demo_convolution_couleur(x_train)
        elif choix == '4':
            demo_max_pooling(x_train)
        elif choix == '5':
            entrainer_cnn_cifar10(x_train, y_train, x_test, y_test)
        elif choix == '0':
            break
        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_partie2()
