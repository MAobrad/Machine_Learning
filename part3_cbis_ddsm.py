"""
part3_cbis_ddsm.py — Partie 3 : Détection de cancer du sein (CBIS-DDSM)
Classification binaire : BENIGN (0) vs MALIGNANT (1)
PyTorch + métriques médicales spécialisées.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('rapport', exist_ok=True)


# ============================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT
# ============================================================

def charger_cbis_ddsm(csv_train, csv_test='', img_dir='.', target_size=(128, 128)):
    """
    Charge CBIS-DDSM depuis CSV + images PNG/DICOM.

    Colonnes CSV attendues :
      - 'pathology'       : BENIGN / BENIGN_WITHOUT_CALLBACK / MALIGNANT
      - 'image file path' : chemin relatif vers l'image

    Labels : BENIGN* → 0, MALIGNANT → 1
    """
    try:
        import pandas as pd
        from PIL import Image
    except ImportError:
        raise ImportError("Installer : pip install pandas Pillow")

    def lire_df(path):
        df = pd.read_csv(path)
        df['label'] = (df['pathology'] == 'MALIGNANT').astype(int)
        return df

    df_train = lire_df(csv_train)
    df_test  = lire_df(csv_test) if csv_test and os.path.exists(csv_test) else None

    print(f"  Train CSV : {len(df_train)} cas")
    benins  = (df_train['label'] == 0).sum()
    malins  = (df_train['label'] == 1).sum()
    print(f"  Bénins : {benins}  Malins : {malins}  "
          f"({malins/(benins+malins)*100:.1f}% malins)")

    # Déterminer la colonne de chemin
    col_img = next((c for c in df_train.columns
                    if 'image' in c.lower() and 'path' in c.lower()), None)
    if col_img is None:
        raise ValueError("Colonne de chemin image non trouvée dans le CSV.")

    def ouvrir_image(path, target_size):
        """Supporte PNG/JPG (PIL) et DICOM (.dcm via pydicom)."""
        from PIL import Image
        if path.lower().endswith('.dcm'):
            try:
                import pydicom
                ds  = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                img = Image.fromarray((arr * 255).astype(np.uint8)).convert('L')
            except ImportError:
                raise ImportError("Installer pydicom : venv/bin/pip install pydicom")
        else:
            img = Image.open(path).convert('L')
        return np.array(img.resize(target_size, Image.LANCZOS), dtype=np.float32) / 255.0

    def charger_images(df):
        X, y = [], []
        ok, manquants = 0, 0
        for _, row in df.iterrows():
            path = os.path.join(img_dir, str(row[col_img]).strip())
            if not os.path.exists(path):
                manquants += 1
                continue
            try:
                arr = ouvrir_image(path, target_size)
                X.append(arr)
                y.append(row['label'])
                ok += 1
            except Exception as e:
                manquants += 1
        print(f"  Images chargées : {ok}  (manquantes/erreurs : {manquants})")
        if ok == 0:
            return None, None
        return np.array(X)[:, :, :, np.newaxis], np.array(y, dtype=np.int64)

    print("  Chargement images train...")
    X_train, y_train = charger_images(df_train)

    if X_train is None:
        print("\n  ⚠️  Aucune image trouvée — les fichiers DICOM ne sont pas téléchargés.")
        print("  → Basculement automatique sur données synthétiques.\n")
        return None, None, None, None, None

    if df_test is not None:
        print("  Chargement images test...")
        X_test, y_test = charger_images(df_test)
        if X_test is None:
            X_test, y_test = None, None
    else:
        X_test, y_test = None, None

    if X_test is None:
        # Split 80/20
        n     = len(X_train)
        idx   = np.random.permutation(n)
        split = int(0.8 * n)
        X_test,  y_test  = X_train[idx[split:]], y_train[idx[split:]]
        X_train, y_train = X_train[idx[:split]],  y_train[idx[:split]]
        print(f"  Split 80/20 : {len(X_train)} train / {len(X_test)} test")

    ratio_poids = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  Ratio bénin/malin (pos_weight) : {ratio_poids:.2f}")

    return X_train, y_train, X_test, y_test, ratio_poids


def donnees_synthetiques(n=300, target_size=(128, 128)):
    """
    Génère des données synthétiques pour tester le pipeline
    quand CBIS-DDSM n'est pas disponible.
    """
    print("  Génération de données synthétiques (128×128, N&B)...")
    np.random.seed(42)
    n_train = int(n * 0.8)
    n_test  = n - n_train

    X_train = np.random.rand(n_train, *target_size, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train).astype(np.int64)
    X_test  = np.random.rand(n_test,  *target_size, 1).astype(np.float32)
    y_test  = np.random.randint(0, 2, n_test).astype(np.int64)

    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  {n_train} train / {n_test} test  |  ratio bénin/malin : {ratio:.2f}")
    return X_train, y_train, X_test, y_test, ratio


# ============================================================
# 2. CNN MAMMOGRAPHIE (PyTorch)
# ============================================================

def entrainer_cnn_mammo(X_train, y_train, X_test, y_test, ratio_poids, epochs=25):
    """
    CNN binaire pour mammographies.
    Entrée : (N, 128, 128, 1) — niveaux de gris
    Sortie : logit → BCEWithLogitsLoss

    # Pourquoi un FN est-il bien plus grave qu'un FP en médical ?
    # FN (Faux Négatif) = cancer classé bénin → patient non traité → pronostic fatal
    # FP (Faux Positif) = bénin classé malin  → biopsie inutile, stress, coût
    # → En oncologie, on accepte plus de FP pour éviter les FN.
    # → On optimise la Sensibilité (recall) au détriment de la Spécificité.

    # Comment adapter le seuil de décision ?
    # Seuil par défaut = 0.5 → équilibré.
    # Baisser à 0.3 : + de prédictions "malin" → + de sensibilité, + de FP.
    # La courbe ROC permet de choisir le seuil selon le trade-off acceptable.

    # Pondération des classes :
    # pos_weight = n_bénins / n_malins > 1 → pénalise davantage les erreurs sur malins.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("  PyTorch non installé : venv/bin/pip install torch")
        return None, None, None, None

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")

    # Tenseurs (N, C, H, W)
    x_tr = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    x_te = torch.tensor(X_test.transpose(0, 3, 1, 2),  dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(x_te, y_te), batch_size=64, shuffle=False, num_workers=0)

    class CNN_Mammography(nn.Module):
        def __init__(self):
            super().__init__()
            # Bloc 1 : 128×128 → 64×64
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            # Bloc 2 : 64×64 → 32×32
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            # Bloc 3 : 32×32 → 16×16
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
            self.bn4   = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            # Tête de classification
            self.fc1  = nn.Linear(128 * 16 * 16, 256)
            self.fc2  = nn.Linear(256, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool2(x)
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.pool3(x)
            x = torch.flatten(x, 1)
            x = self.drop(self.relu(self.fc1(x)))
            return self.fc2(x)   # logits bruts

    model     = CNN_Mammography().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres : {n_params:,}")

    pos_weight = torch.tensor([ratio_poids], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    hist = {'loss_tr': [], 'loss_te': [], 'acc_tr': [], 'acc_te': []}

    for ep in range(epochs):
        model.train()
        loss_s, corr, tot = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.unsqueeze(1).to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_s += loss.item() * xb.size(0)
            corr   += ((torch.sigmoid(out) > 0.5).float() == yb).sum().item()
            tot    += xb.size(0)
        loss_tr = loss_s / tot; acc_tr = corr / tot

        model.eval()
        loss_s2, corr2, tot2 = 0.0, 0, 0
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb  = xb.to(device)
                yb2 = yb.unsqueeze(1).to(device)
                out  = model(xb)
                prob = torch.sigmoid(out)
                loss_s2 += criterion(out, yb2).item() * xb.size(0)
                pred     = (prob > 0.5).float()
                corr2   += (pred == yb2).sum().item()
                tot2    += xb.size(0)
                all_probs.extend(prob.cpu().numpy().flatten())
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(yb.numpy())
        loss_te = loss_s2 / tot2; acc_te = corr2 / tot2

        hist['loss_tr'].append(loss_tr); hist['loss_te'].append(loss_te)
        hist['acc_tr'].append(acc_tr);   hist['acc_te'].append(acc_te)

        print(f"  Epoch {ep+1:2d}/{epochs} | "
              f"Loss {loss_tr:.4f}/{loss_te:.4f} | "
              f"Acc {acc_tr:.4f}/{acc_te:.4f}")

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Courbes d'entraînement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist['loss_tr'], label='Train'); ax1.plot(hist['loss_te'], label='Test')
    ax1.set_title('Loss — Mammographie'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(hist['acc_tr'], label='Train'); ax2.plot(hist['acc_te'], label='Test')
    ax2.set_title('Accuracy — Mammographie'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig('rapport/p3_courbes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardée : rapport/p3_courbes.png")

    return model, y_true, y_pred, y_probs


# ============================================================
# 3. ÉVALUATION MÉDICALE
# ============================================================

def evaluer_medical(y_true, y_pred, y_probs=None, seuil=0.5):
    """
    Métriques médicales complètes.

    Discussion clinique obligatoire :
    - FN (cancer non détecté) : potentiellement fatal si le patient n'est pas traité.
    - FP (fausse alarme) : biopsie inutile, stress, coût élevé, mais réversible.
    - Contexte clinique acceptable : Sensibilité ≥ 0.90, quitte à avoir Spécificité ≈ 0.70.
    - Seuil de décision : descendre sous 0.5 augmente la sensibilité au prix de plus de FP.
    """
    y_pred_bin = (y_pred >= seuil).astype(int) if y_pred.max() <= 1.0 else y_pred.astype(int)

    TP = int(((y_pred_bin == 1) & (y_true == 1)).sum())
    TN = int(((y_pred_bin == 0) & (y_true == 0)).sum())
    FP = int(((y_pred_bin == 1) & (y_true == 0)).sum())
    FN = int(((y_pred_bin == 0) & (y_true == 1)).sum())

    sensibilite = TP / max(TP + FN, 1)
    specificite = TN / max(TN + FP, 1)
    precision   = TP / max(TP + FP, 1)
    f1          = 2 * TP / max(2 * TP + FP + FN, 1)

    print("\n" + "=" * 48)
    print("  ÉVALUATION MÉDICALE")
    print("=" * 48)
    print(f"  Matrice de confusion (seuil={seuil}) :")
    print(f"                    Prédit Bénin  Prédit Malin")
    print(f"    Réel Bénin  :   TN = {TN:6d}   FP = {FP:6d}")
    print(f"    Réel Malin  :   FN = {FN:6d}   TP = {TP:6d}")
    print(f"\n  Sensibilité (Recall)  : {sensibilite:.4f}  ← minimiser les FN")
    print(f"  Spécificité           : {specificite:.4f}")
    print(f"  Précision             : {precision:.4f}")
    print(f"  F1-score              : {f1:.4f}")

    if FN > 0:
        print(f"\n  ⚠️  {FN} cancer(s) NON DÉTECTÉ(S) [FN] — DANGER MÉDICAL")
    if sensibilite < 0.85:
        print(f"  ⚠️  Sensibilité < 85% — inacceptable cliniquement")
    else:
        print(f"  ✅ Sensibilité acceptable (≥ 85%)")

    # Matrice de confusion visuelle
    mat = np.array([[TN, FP], [FN, TP]])
    labels_ax = ['Bénin', 'Malin']
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Prédit Bénin', 'Prédit Malin'])
    ax.set_yticklabels(['Réel Bénin', 'Réel Malin'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha='center', va='center', fontsize=14,
                    color='white' if mat[i, j] > mat.max() / 2 else 'black')
    ax.set_title('Matrice de confusion — Mammographie')
    plt.tight_layout()
    plt.savefig('rapport/p3_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardée : rapport/p3_confusion.png")

    # Courbe ROC + seuil optimal
    if y_probs is not None:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            auc = roc_auc_score(y_true, y_probs)
            print(f"\n  AUC-ROC : {auc:.4f}")

            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            idx_opt = np.argmax(tpr - fpr)   # critère de Youden
            seuil_opt = thresholds[idx_opt]

            print(f"  Seuil optimal (Youden J) : {seuil_opt:.3f}")
            print(f"  → Sensibilité : {tpr[idx_opt]:.4f}  "
                  f"Spécificité : {1 - fpr[idx_opt]:.4f}")
            print(f"  → Pour minimiser les FN : utiliser seuil ≤ {seuil_opt:.3f}")

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC={auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
            ax.scatter(fpr[idx_opt], tpr[idx_opt], color='red', zorder=5,
                       label=f'Seuil optimal = {seuil_opt:.2f}')
            ax.set_xlabel('1 − Spécificité (Faux Positifs)')
            ax.set_ylabel('Sensibilité (Vrais Positifs)')
            ax.set_title('Courbe ROC — Détection Cancer du Sein')
            ax.legend()
            plt.tight_layout()
            plt.savefig('rapport/p3_roc.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("  Figure sauvegardée : rapport/p3_roc.png")

        except ImportError:
            print("  (sklearn non disponible pour AUC-ROC)")

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'sensibilite': sensibilite, 'specificite': specificite, 'f1': f1}


# ============================================================
# 4. MENU PARTIE 3
# ============================================================

def menu_partie3():
    """Menu interactif de la Partie 3."""
    print("\n" + "=" * 52)
    print("   PARTIE 3 — Détection Cancer du Sein (CBIS-DDSM)")
    print("=" * 52)
    print("  Prérequis : CSV CBIS-DDSM + images correspondantes")
    print("  Sans données → pipeline de démonstration synthétique\n")

    csv_train = input("  CSV train [mass_case_description_train_set.csv] : ").strip()
    if not csv_train:
        csv_train = 'mass_case_description_train_set.csv'
    csv_test = input("  CSV test  [mass_case_description_test_set.csv]  : ").strip()
    if not csv_test:
        csv_test = 'mass_case_description_test_set.csv'
    img_dir = input("  Dossier images [.] : ").strip() or '.'

    # Chargement ou données synthétiques
    mode = "synthétique"
    if os.path.exists(csv_train):
        result = charger_cbis_ddsm(csv_train, csv_test, img_dir)
        if result[0] is not None:
            X_train, y_train, X_test, y_test, ratio = result
            mode = "réel"
        else:
            print("  → Basculement sur données synthétiques.")
            X_train, y_train, X_test, y_test, ratio = donnees_synthetiques(n=300)
    else:
        print(f"\n  Fichier non trouvé : {csv_train}")
        print("  → Utilisation de données synthétiques pour la démonstration.")
        X_train, y_train, X_test, y_test, ratio = donnees_synthetiques(n=300)

    print(f"\n  Mode : {mode}")
    print(f"  Train : {X_train.shape}  Test : {X_test.shape}")

    modele_entraine = None
    y_true_last     = None
    y_pred_last     = None
    y_probs_last    = None

    while True:
        print("\n" + "=" * 52)
        print("   PARTIE 3 — CBIS-DDSM")
        print("=" * 52)
        print("  1  - Entraîner CNN Mammographie")
        print("  2  - Évaluation médicale complète")
        print("  3  - Analyse seuil de décision")
        print("  0  - Retour au menu principal")
        print("-" * 52)

        choix = input("Choix : ").strip()

        if choix == '1':
            epochs_str = input("  Nombre d'epochs [25] : ").strip()
            epochs     = int(epochs_str) if epochs_str.isdigit() else 25
            result = entrainer_cnn_mammo(X_train, y_train, X_test, y_test,
                                          ratio, epochs=epochs)
            if result[0] is not None:
                modele_entraine, y_true_last, y_pred_last, y_probs_last = result
                evaluer_medical(y_true_last, y_pred_last, y_probs_last)

        elif choix == '2':
            if y_true_last is None:
                print("  Entraîner d'abord le modèle (option 1).")
            else:
                evaluer_medical(y_true_last, y_pred_last, y_probs_last)

        elif choix == '3':
            if y_probs_last is None:
                print("  Entraîner d'abord le modèle (option 1).")
            else:
                print("\n  Analyse de sensibilité selon le seuil :")
                print(f"  {'Seuil':>6} {'Sensibilité':>12} {'Spécificité':>12} {'F1':>8}")
                print("  " + "-" * 42)
                for seuil in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    res = evaluer_medical(y_true_last, y_probs_last, seuil=seuil)
                    print(f"  {seuil:>6.1f} {res['sensibilite']:>12.4f} "
                          f"{res['specificite']:>12.4f} {res['f1']:>8.4f}")

        elif choix == '0':
            break
        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_partie3()
