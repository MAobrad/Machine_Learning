"""
part3_cbis_ddsm.py - Partie 3 : Detection de cancer du sein (CBIS-DDSM)
Classification binaire : BENIGN (0) vs MALIGNANT (1)
On utilise PyTorch et des metriques adaptees au contexte medical.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('rapport', exist_ok=True)


# ============================================================
# 1. CHARGEMENT ET PRETRAITEMENT
# ============================================================

def charger_cbis_ddsm(csv_train, csv_test='', img_dir='.', target_size=(128, 128)):
    """
    Charge CBIS-DDSM depuis un fichier CSV + les images PNG/DICOM correspondantes.

    Colonnes CSV attendues :
      - 'pathology'       : BENIGN / BENIGN_WITHOUT_CALLBACK / MALIGNANT
      - 'image file path' : chemin relatif vers l'image

    Labels : BENIGN = 0, MALIGNANT = 1
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
    df_test = lire_df(csv_test) if csv_test and os.path.exists(csv_test) else None

    print(f"  Train CSV : {len(df_train)} cas")
    benins = (df_train['label'] == 0).sum()
    malins = (df_train['label'] == 1).sum()
    print(f"  Benins : {benins}  Malins : {malins}  "
          f"({malins/(benins+malins)*100:.1f}% malins)")

    col_img = next((c for c in df_train.columns
                    if 'image' in c.lower() and 'path' in c.lower()), None)
    if col_img is None:
        raise ValueError("Colonne de chemin image non trouvee dans le CSV.")

    def ouvrir_image(path, target_size):
        from PIL import Image
        if path.lower().endswith('.dcm'):
            try:
                import pydicom
                ds = pydicom.dcmread(path)
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
            except Exception:
                manquants += 1
        print(f"  Images chargees : {ok}  (manquantes/erreurs : {manquants})")
        if ok == 0:
            return None, None
        return np.array(X)[:, :, :, np.newaxis], np.array(y, dtype=np.int64)

    print("  Chargement images train...")
    X_train, y_train = charger_images(df_train)

    if X_train is None:
        print("\n  Aucune image trouvee - les fichiers DICOM ne sont pas telecharges.")
        print("  Basculement automatique sur donnees synthetiques.\n")
        return None, None, None, None, None

    if df_test is not None:
        print("  Chargement images test...")
        X_test, y_test = charger_images(df_test)
        if X_test is None:
            X_test, y_test = None, None
    else:
        X_test, y_test = None, None

    if X_test is None:
        n = len(X_train)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        X_test = X_train[idx[split:]]
        y_test = y_train[idx[split:]]
        X_train = X_train[idx[:split]]
        y_train = y_train[idx[:split]]
        print(f"  Split 80/20 : {len(X_train)} train / {len(X_test)} test")

    ratio_poids = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  Ratio benin/malin (pos_weight) : {ratio_poids:.2f}")

    return X_train, y_train, X_test, y_test, ratio_poids


def afficher_stats_dataset(csv_train, csv_test=''):
    """Affiche des statistiques detaillees sur le dataset CBIS-DDSM."""
    try:
        import pandas as pd
    except ImportError:
        print("  pandas non disponible.")
        return

    print("\n" + "=" * 58)
    print("  STATISTIQUES CBIS-DDSM")
    print("=" * 58)

    def stats_df(path, nom):
        if not os.path.exists(path):
            print(f"  {nom} : fichier non trouve ({path})")
            return
        df = pd.read_csv(path)
        print(f"\n  {nom} ({len(df)} cas) :")
        if 'pathology' in df.columns:
            counts = df['pathology'].value_counts()
            for k, v in counts.items():
                print(f"    {k:<35} : {v:4d} ({v/len(df)*100:.1f}%)")
        if 'breast_density' in df.columns:
            print(f"\n  Densite mammaire :")
            for k, v in df['breast_density'].value_counts().items():
                print(f"    Densite {k} : {v} cas")
        if 'assessment' in df.columns:
            print(f"\n  Scores BI-RADS (assessment) :")
            for k, v in sorted(df['assessment'].value_counts().items()):
                print(f"    BI-RADS {k} : {v} cas")

    stats_df(csv_train, "Train set")
    stats_df(csv_test, "Test set")

    print(f"\n  NOTE : Les images DICOM compressees (~163 Go) sont disponibles")
    print(f"  sur TCIA (The Cancer Imaging Archive) : tcia.cancerimagingarchive.net")
    print(f"  Collection : CBIS-DDSM (Curated Breast Imaging Subset of DDSM)")
    print("=" * 58)


def donnees_synthetiques(n=300, target_size=(128, 128)):
    """
    Genere des donnees aleatoires pour tester le pipeline
    quand les images CBIS-DDSM ne sont pas disponibles (~163 Go sur TCIA).
    """
    print("  Generation de donnees synthetiques (128x128, N&B)...")
    np.random.seed(42)
    n_train = int(n * 0.8)
    n_test = n - n_train

    X_train = np.random.rand(n_train, *target_size, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train).astype(np.int64)
    X_test = np.random.rand(n_test, *target_size, 1).astype(np.float32)
    y_test = np.random.randint(0, 2, n_test).astype(np.int64)

    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  {n_train} train / {n_test} test  |  ratio benin/malin : {ratio:.2f}")
    return X_train, y_train, X_test, y_test, ratio


# ============================================================
# 2. CNN MAMMOGRAPHIE (PyTorch)
# ============================================================

def entrainer_cnn_mammo(X_train, y_train, X_test, y_test, ratio_poids,
                        epochs=25, patience=6):
    """
    CNN binaire pour mammographies avec early stopping.
    Entree : (N, 128, 128, 1) - images en niveaux de gris
    Sortie : logit -> BCEWithLogitsLoss (combine sigmoid + cross-entropy)

    Pourquoi un FN est-il bien plus grave qu'un FP en medecine ?
    - FN (Faux Negatif) = un cancer classe benin, le patient n'est pas traite, danger vital
    - FP (Faux Positif) = un benin classe malin, biopsie inutile, stress, mais reversible
    En oncologie, on prefere accepter plus de FP pour eviter de rater des cancers.
    On optimise donc la Sensibilite (taux de vrais positifs) meme si ca genere plus de FP.

    pos_weight = nb_benins / nb_malins : on penalise davantage les erreurs sur les malins
    pour compenser le desequilibre des classes.

    Early stopping : si la loss de validation ne s'ameliore pas pendant 'patience' epochs,
    on arrete et on restaure le meilleur modele — evite l'overfitting.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("  PyTorch non installe : venv/bin/pip install torch")
        return None, None, None, None

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")

    x_tr = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    x_te = torch.tensor(X_test.transpose(0, 3, 1, 2),  dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(x_te, y_te), batch_size=64, shuffle=False, num_workers=0)

    class CNN_Mammography(nn.Module):
        def __init__(self):
            super().__init__()
            # Bloc 1 : 128x128 → 64x64
            self.conv1 = nn.Conv2d(1,  32, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            # Bloc 2 : 64x64 → 32x32
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            # Bloc 3 : 32x32 → 16x16
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
            self.bn4   = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            # Tete de classification binaire
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
            return self.fc2(x)

    model = CNN_Mammography().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametres : {n_params:,}")

    pos_weight = torch.tensor([ratio_poids], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # ReduceLROnPlateau : divise le lr par 2 si la loss test ne diminue pas pendant 3 epochs
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=3)

    hist = {'loss_tr': [], 'loss_te': [], 'acc_tr': [], 'acc_te': []}

    # Early stopping
    best_loss = float('inf')
    best_state = None
    no_improve = 0

    import time
    t0 = time.time()

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
        loss_tr = loss_s / tot
        acc_tr  = corr / tot

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
        loss_te = loss_s2 / tot2
        acc_te  = corr2 / tot2

        hist['loss_tr'].append(loss_tr)
        hist['loss_te'].append(loss_te)
        hist['acc_tr'].append(acc_tr)
        hist['acc_te'].append(acc_te)

        scheduler.step(loss_te)
        elapsed = time.time() - t0
        print(f"  Epoch {ep+1:2d}/{epochs} | "
              f"Loss {loss_tr:.4f}/{loss_te:.4f} | "
              f"Acc {acc_tr:.4f}/{acc_te:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")

        # Early stopping
        if loss_te < best_loss - 1e-4:
            best_loss  = loss_te
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping a l'epoch {ep+1} "
                      f"(pas d'amelioration depuis {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Meilleur modele restaure (loss test = {best_loss:.4f})")

    # Recalculer les predictions avec le meilleur modele
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb  = xb.to(device)
            out  = model(xb)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()
            all_probs.extend(prob.cpu().numpy().flatten())
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(yb.numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist['loss_tr'], label='Train')
    ax1.plot(hist['loss_te'], label='Test')
    ax1.set_title('Loss - Mammographie')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(hist['acc_tr'], label='Train')
    ax2.plot(hist['acc_te'], label='Test')
    ax2.set_title('Accuracy - Mammographie')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('rapport/p3_courbes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : rapport/p3_courbes.png")

    return model, y_true, y_pred, y_probs


# ============================================================
# 3. EVALUATION MEDICALE
# ============================================================

def evaluer_medical(y_true, y_pred, y_probs=None, seuil=0.5, silent=False):
    """
    Metriques medicales completes pour la detection de cancer.

    silent=True : mode silencieux pour le sweep de seuil (pas de plots, sortie courte)

    Rappel des 4 cas possibles :
    - TP (Vrai Positif)  : cancer detecte correctement
    - TN (Vrai Negatif)  : benin detecte correctement
    - FP (Faux Positif)  : benin classe malin, biopsie inutile, mais patient averti
    - FN (Faux Negatif)  : cancer classe benin, non traite, potentiellement fatal

    En pratique clinique, on accepte une sensibilite >= 90% meme si la specificite baisse.
    Le seuil optimal est determine via la courbe ROC (critere de Youden).
    """
    y_pred_bin = (np.asarray(y_pred) >= seuil).astype(int)
    y_true_arr = np.asarray(y_true).astype(int)

    TP = int(((y_pred_bin == 1) & (y_true_arr == 1)).sum())
    TN = int(((y_pred_bin == 0) & (y_true_arr == 0)).sum())
    FP = int(((y_pred_bin == 1) & (y_true_arr == 0)).sum())
    FN = int(((y_pred_bin == 0) & (y_true_arr == 1)).sum())

    sensibilite = TP / max(TP + FN, 1)
    specificite = TN / max(TN + FP, 1)
    precision   = TP / max(TP + FP, 1)
    f1          = 2 * TP / max(2 * TP + FP + FN, 1)

    if not silent:
        print("\n" + "=" * 48)
        print("  EVALUATION MEDICALE")
        print("=" * 48)
        print(f"  Matrice de confusion (seuil={seuil}) :")
        print(f"                    Predit Benin  Predit Malin")
        print(f"    Reel Benin  :   TN = {TN:6d}   FP = {FP:6d}")
        print(f"    Reel Malin  :   FN = {FN:6d}   TP = {TP:6d}")
        print(f"\n  Sensibilite (Recall) : {sensibilite:.4f}  (objectif : minimiser les FN)")
        print(f"  Specificite          : {specificite:.4f}")
        print(f"  Precision            : {precision:.4f}")
        print(f"  F1-score             : {f1:.4f}")

        if FN > 0:
            print(f"\n  ATTENTION : {FN} cancer(s) NON DETECTE(S) [FN] — DANGER MEDICAL")
        if sensibilite < 0.85:
            print(f"  Sensibilite < 85% — inacceptable cliniquement")
        else:
            print(f"  Sensibilite acceptable (>= 85%)")

        mat = np.array([[TN, FP], [FN, TP]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mat, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predit Benin', 'Predit Malin'])
        ax.set_yticklabels(['Reel Benin', 'Reel Malin'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mat[i, j], ha='center', va='center', fontsize=14,
                        color='white' if mat[i, j] > mat.max() / 2 else 'black')
        ax.set_title('Matrice de confusion - Mammographie')
        plt.tight_layout()
        plt.savefig('rapport/p3_confusion.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  Figure sauvegardee : rapport/p3_confusion.png")

    if y_probs is not None and not silent:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            auc = roc_auc_score(y_true_arr, y_probs)
            print(f"\n  AUC-ROC : {auc:.4f}")

            fpr, tpr, thresholds = roc_curve(y_true_arr, y_probs)
            idx_opt   = np.argmax(tpr - fpr)
            seuil_opt = thresholds[idx_opt]

            print(f"  Seuil optimal (Youden J) : {seuil_opt:.3f}")
            print(f"  Sensibilite : {tpr[idx_opt]:.4f}  Specificite : {1 - fpr[idx_opt]:.4f}")
            print(f"  Pour minimiser les FN : utiliser seuil <= {seuil_opt:.3f}")

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC={auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Aleatoire')
            ax.scatter(fpr[idx_opt], tpr[idx_opt], color='red', zorder=5,
                       label=f'Seuil optimal = {seuil_opt:.2f}')
            ax.set_xlabel('1 - Specificite (Faux Positifs)')
            ax.set_ylabel('Sensibilite (Vrais Positifs)')
            ax.set_title('Courbe ROC - Detection Cancer du Sein')
            ax.legend()
            plt.tight_layout()
            plt.savefig('rapport/p3_roc.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("  Figure sauvegardee : rapport/p3_roc.png")

        except ImportError:
            print("  (sklearn non disponible pour AUC-ROC)")

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'sensibilite': sensibilite, 'specificite': specificite,
            'precision': precision, 'f1': f1}


# ============================================================
# 4. MENU PARTIE 3
# ============================================================

def menu_partie3():
    """Menu interactif de la Partie 3."""
    print("\n" + "=" * 55)
    print("   PARTIE 3 - Detection Cancer du Sein (CBIS-DDSM)")
    print("=" * 55)
    print("  Prerequis : CSV CBIS-DDSM + images correspondantes")
    print("  Sans donnees : pipeline de demonstration synthetique\n")

    csv_train = input("  CSV train [mass_case_description_train_set.csv] : ").strip()
    if not csv_train:
        csv_train = 'mass_case_description_train_set.csv'
    csv_test = input("  CSV test  [mass_case_description_test_set.csv]  : ").strip()
    if not csv_test:
        csv_test = 'mass_case_description_test_set.csv'
    img_dir = input("  Dossier images [.] : ").strip() or '.'

    mode = "synthetique"
    if os.path.exists(csv_train):
        result = charger_cbis_ddsm(csv_train, csv_test, img_dir)
        if result[0] is not None:
            X_train, y_train, X_test, y_test, ratio = result
            mode = "reel"
        else:
            print("  Basculement sur donnees synthetiques.")
            X_train, y_train, X_test, y_test, ratio = donnees_synthetiques(n=300)
    else:
        print(f"\n  Fichier non trouve : {csv_train}")
        print("  Utilisation de donnees synthetiques pour la demonstration.")
        X_train, y_train, X_test, y_test, ratio = donnees_synthetiques(n=300)

    print(f"\n  Mode : {mode}")
    print(f"  Train : {X_train.shape}  Test : {X_test.shape}")

    modele_entraine = None
    y_true_last  = None
    y_pred_last  = None
    y_probs_last = None

    while True:
        print("\n" + "=" * 55)
        print("   PARTIE 3 - CBIS-DDSM")
        print("=" * 55)
        print("  1  - Entrainer CNN Mammographie (early stopping)")
        print("  2  - Evaluation medicale complete")
        print("  3  - Analyse du seuil de decision")
        print("  4  - Statistiques du dataset CSV")
        print("  0  - Retour au menu principal")
        print("-" * 55)

        choix = input("Choix : ").strip()

        if choix == '1':
            epochs_str = input("  Nombre d'epochs max [25] : ").strip()
            epochs = int(epochs_str) if epochs_str.isdigit() else 25
            result = entrainer_cnn_mammo(X_train, y_train, X_test, y_test,
                                         ratio, epochs=epochs)
            if result[0] is not None:
                modele_entraine, y_true_last, y_pred_last, y_probs_last = result
                evaluer_medical(y_true_last, y_pred_last, y_probs_last)

        elif choix == '2':
            if y_true_last is None:
                print("  Entrainer d'abord le modele (option 1).")
            else:
                evaluer_medical(y_true_last, y_pred_last, y_probs_last)

        elif choix == '3':
            if y_probs_last is None:
                print("  Entrainer d'abord le modele (option 1).")
            else:
                print("\n  Analyse de sensibilite selon le seuil de decision :")
                print(f"  {'Seuil':>6} {'Sensibilite':>12} {'Specificite':>12} {'F1':>8} {'FN':>5} {'FP':>5}")
                print("  " + "-" * 52)
                for seuil in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    # silent=True : pas de plots, juste les chiffres
                    res = evaluer_medical(y_true_last, y_probs_last,
                                          seuil=seuil, silent=True)
                    print(f"  {seuil:>6.1f} {res['sensibilite']:>12.4f} "
                          f"{res['specificite']:>12.4f} {res['f1']:>8.4f} "
                          f"{res['FN']:>5} {res['FP']:>5}")
                print("\n  Interpretation :")
                print("    Baisser le seuil : + Sensibilite (moins de FN), + FP")
                print("    Hausser le seuil : + Specificite (moins de FP), + FN")
                print("    En oncologie : privilegier la sensibilite pour eviter les FN")

        elif choix == '4':
            afficher_stats_dataset(csv_train, csv_test)

        elif choix == '0':
            break
        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_partie3()
