# Projet ML — SM604 EFREI Paris
CNN : De la classification MNIST à la détection de cancers du sein.

## Lancer le projet

```bash
source venv/bin/activate
python main.py
```

L'interpréteur Python à utiliser est **Python 3.11** (Homebrew), installé dans le virtualenv `venv/`.
Ne jamais utiliser `/opt/homebrew/bin/python3` (Python 3.14, incompatible avec TensorFlow).

## Structure des fichiers

```
main.py              ← menu principal (entrée unique)
utils.py             ← fonctions NumPy partagées (softmax, cross-entropy, conv, pooling)
part1_mnist.py       ← Partie 1 : MNIST, modèles NumPy purs (linéaire, MLP-1, MLP-2)
part2_cifar10.py     ← Partie 2 : CIFAR-10, filtres, CNN PyTorch
part3_cbis_ddsm.py   ← Partie 3 : mammographies CBIS-DDSM, métriques médicales
models.py            ← fichier original du coéquipier (conservé pour référence)
rapport/             ← figures exportées automatiquement (PNG)
venv/                ← virtualenv Python 3.11
```

## Règle absolue du projet

- **NumPy pur obligatoire** pour : softmax, cross-entropy, rétropropagation, convolution, max-pooling
- **PyTorch autorisé** uniquement pour le CNN CIFAR-10 (section 2.6 Option B) et la Partie 3
- **sklearn autorisé** uniquement pour PCA, t-SNE, AUC-ROC (visualisation/évaluation)

## Dépendances

```bash
venv/bin/pip install numpy matplotlib tensorflow scikit-learn seaborn pandas Pillow pydicom torch
```

TensorFlow sert uniquement à télécharger MNIST et CIFAR-10 (`keras.datasets`).

## Données

- **MNIST / CIFAR-10** : téléchargés automatiquement au premier lancement
- **CBIS-DDSM** : CSV présents (`mass_case_description_*.csv`), images DICOM non téléchargées (~163 GB sur TCIA). Le pipeline bascule automatiquement sur données synthétiques si les images sont absentes.

## Conventions mathématiques

- Matrices de poids `A` de forme `(input_dim, output_dim)` — format batch `X @ A`
- Initialisation **He** pour couches ReLU, **Xavier** pour couches linéaires
- Mini-batch SGD ou **Adam** avec shuffle à chaque epoch
- `np.random.seed(42)` en tête de chaque fichier
- `relu()` et `relu_deriv()` centralisés dans `utils.py`

## Optimiseurs disponibles (Part 1)

- `optimizer='sgd'` (défaut) — SGD classique, `lr=0.1`
- `optimizer='adam'` — Adam avec moments m/v adaptatifs, `lr=0.001`
- Menu option `b` : comparaison visuelle SGD vs Adam sur MLP-2

## CNN CIFAR-10 (Part 2) — améliorations

- Architecture : 5 couches conv (64→64→128→128→256), 2 FC (512→10)
- **Data augmentation** : flip horizontal aléatoire + random crop 32×32 avec padding 4
- Normalisation CIFAR-10 : mean=[0.491,0.482,0.446] std=[0.247,0.243,0.261]
- Scheduler : CosineAnnealingLR sur 25 epochs

## CNN Mammographie (Part 3) — améliorations

- **Early stopping** : arrêt si loss test stagne `patience=6` epochs, meilleur modèle restauré
- Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)
- Fix : threshold sweep (option 3) fonctionne sans générer de plots

## Figures produites (dans rapport/)

| Fichier | Contenu |
|---|---|
| `p1_lineaire_courbes.png` | Loss + erreur train/test du modèle linéaire MNIST |
| `p1_mlp1_courbes.png` | Idem MLP 1 couche cachée |
| `p1_mlp2_courbes.png` | Idem MLP 2 couches cachées |
| `p1_comparaison.png` | Comparaison des 3 architectures |
| `p1_sgd_vs_adam.png` | Comparaison SGD vs Adam (MLP-2) |
| `p1_confusion.png` | Matrice de confusion 10×10 |
| `p1_erreurs.png` | 10 images mal classées |
| `p1_pca.png` | PCA 2D des représentations |
| `p1_tsne.png` | t-SNE des représentations |
| `p2_exemples.png` | Grille d'exemples CIFAR-10 (8 images × 10 classes) |
| `p2_etudes_preliminaires.png` | Barplot comparatif gris vs couleur |
| `p2_filtres.png` | 6 filtres de convolution sur image CIFAR |
| `p2_conv_couleur.png` | Convolution 3 canaux → feature map |
| `p2_maxpooling.png` | Démo Max-Pooling 2×2 |
| `p2_cnn_courbes.png` | Loss + accuracy CNN CIFAR-10 (avec augmentation) |
| `p3_courbes.png` | Loss + accuracy CNN mammographie |
| `p3_confusion.png` | Matrice de confusion binaire |
| `p3_roc.png` | Courbe ROC + seuil optimal |
