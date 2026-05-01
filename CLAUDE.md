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
- Mini-batch SGD avec shuffle à chaque epoch
- `np.random.seed(42)` en tête de chaque fichier

## Figures produites (dans rapport/)

| Fichier | Contenu |
|---|---|
| `p1_lineaire_courbes.png` | Loss + erreur train/test du modèle linéaire MNIST |
| `p1_mlp1_courbes.png` | Idem MLP 1 couche cachée |
| `p1_mlp2_courbes.png` | Idem MLP 2 couches cachées |
| `p1_comparaison.png` | Comparaison des 3 architectures |
| `p1_confusion.png` | Matrice de confusion 10×10 |
| `p1_erreurs.png` | 10 images mal classées |
| `p1_pca.png` | PCA 2D des représentations |
| `p1_tsne.png` | t-SNE des représentations |
| `p2_etudes_preliminaires.png` | Barplot comparatif gris vs couleur |
| `p2_filtres.png` | 6 filtres de convolution sur image CIFAR |
| `p2_conv_couleur.png` | Convolution 3 canaux → feature map |
| `p2_maxpooling.png` | Démo Max-Pooling 2×2 |
| `p2_cnn_courbes.png` | Loss + accuracy CNN CIFAR-10 |
| `p3_courbes.png` | Loss + accuracy CNN mammographie |
| `p3_confusion.png` | Matrice de confusion binaire |
| `p3_roc.png` | Courbe ROC + seuil optimal |
