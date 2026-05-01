"""
main.py — Point d'entrée principal du projet ML (SM604 — EFREI Paris)
CNN : De la classification MNIST à la détection de cancers du sein.

Utilisation : venv/bin/python main.py
"""

import os
os.makedirs('rapport', exist_ok=True)


def auto_evaluation():
    """Auto-evaluation du projet — description de ce que le code fait reellement."""

    print("\n" + "=" * 68)
    print("  Auto-evaluation — Projet ML SM604 EFREI Paris")
    print("=" * 68)

    print("""
  ══════════════════════════════════════════════════════════════════
  PARTIE 1 — Classification MNIST (NumPy pur)
  ══════════════════════════════════════════════════════════════════

  Ce que le code fait :

  Trois modeles implementes entierement en NumPy :
  · Modele lineaire : o = X @ A + b, puis P = softmax(o)
    Initialisation Xavier : limit = sqrt(6 / (n_in + n_out))

  · MLP 1 couche cachee (h=128) :
    Z1 = X @ W1 + b1,  H = ReLU(Z1),  Z2 = H @ W2 + b2
    Initialisation He : std = sqrt(2 / n_in), adaptee a ReLU

  · MLP 2 couches cachees (128/64) :
    Meme principe avec une couche supplementaire H2 = ReLU(Z2)

  La retropropagation est calculee analytiquement a la main :
    dZ = (P - Y) / n  (gradient fusionne softmax + cross-entropy)
    dW = X.T @ dZ,  db = sum(dZ)
    On propage vers l'arriere via la derivee de ReLU : dZ1 = dH * (Z1 > 0)

  Deux optimiseurs disponibles :
  · SGD mini-batch (batch=256, shuffle a chaque epoch)
  · Adam : maintient deux moments par parametre
    m = beta1*m + (1-beta1)*g       (moyenne des gradients)
    v = beta2*v + (1-beta2)*g^2     (variance des gradients)
    Mise a jour : param -= lr * m_hat / (sqrt(v_hat) + eps)

  Outils d'analyse produits :
  · Courbes loss + erreur train/test pour chaque modele
  · Comparaison des 3 architectures sur un seul graphe
  · Matrice de confusion 10x10 (chiffres confondus)
  · 10 images mal classees avec label reel vs predit
  · PCA 2D et t-SNE des representations de la couche cachee
  · Grid search : 3 architectures x 2 lr x 2 optimiseurs

  ══════════════════════════════════════════════════════════════════
  PARTIE 2 — CIFAR-10 + Convolutions manuelles + CNN PyTorch
  ══════════════════════════════════════════════════════════════════

  Ce que le code fait :

  Etudes preliminaires — MLP NumPy sur CIFAR-10 :
  · Conversion RGB→gris via ITU-R BT.601 : 0.299R + 0.587G + 0.114B
  · 6 configs testees (3 architectures x 2 entrees gris/couleur)
  · Resultat attendu : ~35-45% erreur (CIFAR bien plus dur que MNIST)

  Convolutions implantees en NumPy, boucle pixel par pixel :
  · convolve2d()      : zero-padding same, kernel 3x3, 1 canal
  · convolve2d_color(): somme sur les 3 canaux RGB → 1 feature map
  · max_pooling2x2()  : stride=2, garde le max de chaque bloc 2x2
  · 6 filtres demontres : flou, nettete, bords verticaux/horizontaux,
    Sobel H, Sobel diagonal

  CNN PyTorch — architecture :
  · Bloc 1 : Conv(3→64) + Conv(64→64) + BN + ReLU + MaxPool → 16x16
  · Bloc 2 : Conv(64→128) + Conv(128→128) + BN + ReLU + MaxPool → 8x8
  · Bloc 3 : Conv(128→256) + BN + ReLU + MaxPool → 4x4
  · Tete   : Dropout(0.4) + FC(4096→512) + Dropout(0.3) + FC(512→10)
  · Augmentation : flip horizontal 50% + crop aleatoire (padding=4)
  · Normalisation : mean=[0.491,0.482,0.446] std=[0.247,0.243,0.261]
  · Scheduler CosineAnnealingLR sur 25 epochs

  ══════════════════════════════════════════════════════════════════
  PARTIE 3 — Detection cancer du sein (CBIS-DDSM)
  ══════════════════════════════════════════════════════════════════

  Ce que le code fait :

  Donnees — pipeline complet :
  · Lit le CSV CBIS-DDSM, convertit pathology → label 0/1
    (BENIGN=0, BENIGN_WITHOUT_CALLBACK=0, MALIGNANT=1)
  · Ouvre les images PNG via PIL et DICOM via pydicom
  · Si aucune image trouvee : bascule automatiquement sur donnees
    synthetiques pour pouvoir tester le pipeline

  CNN PyTorch binaire :
  · BCEWithLogitsLoss avec pos_weight = nb_benins / nb_malins
    → penalise davantage les erreurs sur les cas malins
  · Early stopping : arrete l'entrainement si la loss test ne
    s'ameliore pas pendant 6 epochs, restaure le meilleur etat
  · ReduceLROnPlateau : divise le lr par 2 apres 3 epochs stagnantes

  Metriques medicales produites :
  · Matrice de confusion : TP / TN / FP / FN
  · Sensibilite = TP / (TP + FN)  → minimiser les cancers ratesr
  · Specificite = TN / (TN + FP)  → minimiser les fausses alarmes
  · F1-score, Precision
  · Courbe ROC + AUC
  · Seuil optimal par critere de Youden J = max(sensibilite + specificite - 1)
  · Sweep de seuil 0.2→0.8 en tableau (sans generer de plots)

  ══════════════════════════════════════════════════════════════════
  FICHIERS DU PROJET
  ══════════════════════════════════════════════════════════════════

  utils.py          fonctions partagees NumPy (softmax, cross-entropy,
                    conv, pooling, relu, initialisations)
  part1_mnist.py    modeles + entrainement + visualisations MNIST
  part2_cifar10.py  etudes + filtres + CNN CIFAR-10
  part3_cbis_ddsm.py  pipeline mammographie + metriques medicales
  main.py           menu principal + auto-evaluation
  rapport/          figures PNG exportees automatiquement
""")

    print("=" * 68)


def menu_principal():
    while True:
        print("\n" + "=" * 55)
        print("   PROJET ML — CNN  (SM604 EFREI Paris)")
        print("=" * 55)
        print("  1  - Partie 1 : Classification MNIST     (NumPy pur)")
        print("  2  - Partie 2 : CIFAR-10 + Convolutions + CNN PyTorch")
        print("  3  - Partie 3 : Détection cancer du sein (CBIS-DDSM)")
        print("  a  - Auto-evaluation du projet")
        print("  0  - Quitter")
        print("-" * 55)

        choix = input("Choix : ").strip()

        if choix == '1':
            from part1_mnist import menu_partie1
            menu_partie1()

        elif choix == '2':
            from part2_cifar10 import menu_partie2
            menu_partie2()

        elif choix == '3':
            from part3_cbis_ddsm import menu_partie3
            menu_partie3()

        elif choix == 'a':
            auto_evaluation()

        elif choix == '0':
            print("Fin du programme.")
            break

        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_principal()
