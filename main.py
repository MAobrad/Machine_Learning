import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ============================================================
# 1. Chargement et préparation des données MNIST
# ============================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Transformation des images 28x28 en vecteurs de taille 784
x_train = x_train.reshape(x_train.shape[0], 784).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], 784).astype(np.float32)

# Normalisation des pixels entre 0 et 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Encodage one-hot des labels
def one_hot(y, num_classes=10):
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)

print("x_train :", x_train.shape)
print("y_train_onehot :", y_train_onehot.shape)
print("x_test :", x_test.shape)
print("y_test_onehot :", y_test_onehot.shape)



from models import (
    one_hot,
    cross_entropy,
    accuracy,
    ModeleLineaire,
    ModeleUneCoucheCachee,
    ModeleDeuxCouchesCachees
)

def entrainer_modele(
    modele,
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate=0.1,
    epochs=50,
    batch_size=256
):
    y_train_onehot = one_hot(y_train, 10)
    y_test_onehot = one_hot(y_test, 10)

    n_train = x_train.shape[0]

    train_losses = []
    train_errors = []
    test_errors = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_onehot[indices]

        for start in range(0, n_train, batch_size):
            end = start + batch_size

            X_batch = x_train_shuffled[start:end]
            Y_batch = y_train_shuffled[start:end]

            # Forward propagation
            P = modele.forward(X_batch)

            # Backward propagation
            modele.backward(Y_batch)

            # Mise à jour des paramètres
            modele.update(learning_rate)

        # Évaluation après chaque époque
        P_train = modele.forward(x_train)
        loss_train = cross_entropy(y_train_onehot, P_train)

        y_train_pred = modele.predict(x_train)
        y_test_pred = modele.predict(x_test)

        train_acc = accuracy(y_train, y_train_pred)
        test_acc = accuracy(y_test, y_test_pred)

        train_error = 1 - train_acc
        test_error = 1 - test_acc

        train_losses.append(loss_train)
        train_errors.append(train_error)
        test_errors.append(test_error)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {loss_train:.4f} | "
            f"Erreur train: {train_error:.4f} | "
            f"Erreur test: {test_error:.4f}"
        )

    return train_losses, train_errors, test_errors


def afficher_courbes(losses, train_errors, test_errors, titre):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Époque")
    plt.ylabel("Cross-entropy")
    plt.title(f"Fonction de coût - {titre}")
    plt.show()

    plt.figure()
    plt.plot(train_errors, label="Erreur entraînement")
    plt.plot(test_errors, label="Erreur test")
    plt.xlabel("Époque")
    plt.ylabel("Taux d'erreur")
    plt.title(f"Comparaison des erreurs - {titre}")
    plt.legend()
    plt.show()


def afficher_erreurs(modele, x_test, y_test, titre):
    y_pred = modele.predict(x_test)
    erreurs = np.where(y_pred != y_test)[0]

    plt.figure(figsize=(10, 5))

    for i in range(min(10, len(erreurs))):
        idx = erreurs[i]

        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"Vrai: {y_test[idx]} / Prédit: {y_pred[idx]}")
        plt.axis("off")

    plt.suptitle(titre)
    plt.show()

def grid_search_architecture_complete(x_train, y_train, x_test, y_test):
    configurations = []

    # 0 couche cachée : modèle linéaire
    configurations.append({
        "nom": "Linéaire",
        "nb_couches": 0,
        "neurones": [],
        "modele": ModeleLineaire(),
        "learning_rate": 0.1
    })

    # 1 couche cachée
    hidden_dims = [32, 64, 128, 256]
    learning_rates = [0.01, 0.1]

    for h in hidden_dims:
        for lr in learning_rates:
            configurations.append({
                "nom": f"1 couche cachée - {h} neurones - lr={lr}",
                "nb_couches": 1,
                "neurones": [h],
                "modele": ModeleUneCoucheCachee(hidden_dim=h),
                "learning_rate": lr
            })

    # 2 couches cachées
    architectures_2_couches = [
        (64, 32),
        (64, 64),
        (128, 64),
        (128, 128),
        (256, 128)
    ]

    for h1, h2 in architectures_2_couches:
        for lr in learning_rates:
            configurations.append({
                "nom": f"2 couches cachées - {h1}/{h2} neurones - lr={lr}",
                "nb_couches": 2,
                "neurones": [h1, h2],
                "modele": ModeleDeuxCouchesCachees(hidden1=h1, hidden2=h2),
                "learning_rate": lr
            })

    meilleur_resultat = None
    resultats = []

    for config in configurations:
        print("\n==============================")
        print(f"Test : {config['nom']}")
        print("==============================")

        loss, train_err, test_err = entrainer_modele(
            modele=config["modele"],
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            learning_rate=config["learning_rate"],
            epochs=10,
            batch_size=256
        )

        resultat = {
            "nom": config["nom"],
            "nb_couches": config["nb_couches"],
            "neurones": config["neurones"],
            "learning_rate": config["learning_rate"],
            "erreur_train": train_err[-1],
            "erreur_test": test_err[-1]
        }

        resultats.append(resultat)

        if meilleur_resultat is None or resultat["erreur_test"] < meilleur_resultat["erreur_test"]:
            meilleur_resultat = resultat

    print("\n==============================")
    print("RÉSULTATS DE LA GRID SEARCH")
    print("==============================")

    for r in resultats:
        print(
            f"{r['nom']} | "
            f"Erreur train: {r['erreur_train']:.4f} | "
            f"Erreur test: {r['erreur_test']:.4f}"
        )

    print("\n==============================")
    print("MEILLEURE CONFIGURATION")
    print("==============================")
    print(f"Nom : {meilleur_resultat['nom']}")
    print(f"Nombre de couches cachées : {meilleur_resultat['nb_couches']}")
    print(f"Neurones par couche : {meilleur_resultat['neurones']}")
    print(f"Learning rate : {meilleur_resultat['learning_rate']}")
    print(f"Erreur train : {meilleur_resultat['erreur_train']:.4f}")
    print(f"Erreur test : {meilleur_resultat['erreur_test']:.4f}")

    return meilleur_resultat, resultats

def menu():
    print("\n==============================")
    print(" PROJET MNIST - MENU")
    print("==============================")
    print("1 - Entraîner le modèle linéaire")
    print("2 - Entraîner le modèle avec une couche cachée")
    print("3 - Comparer les deux modèles")
    print("4 - Modèle avec 2 couches cachées")
    print("5 - Chercher architecture optimale")
    print("6 - Quitter")
    choix = input("Ton choix : ")
    return choix


def lancer_modele_lineaire():
    modele_lineaire = ModeleLineaire(input_dim=784, output_dim=10)

    loss_lin, train_err_lin, test_err_lin = entrainer_modele(
        modele=modele_lineaire,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_rate=0.5,
        epochs=10,
        batch_size=256
    )

    afficher_courbes(loss_lin, train_err_lin, test_err_lin, "Modèle linéaire")
    afficher_erreurs(modele_lineaire, x_test, y_test, "Erreurs du modèle linéaire")

    return modele_lineaire, loss_lin, train_err_lin, test_err_lin


def lancer_modele_couche_cachee():
    modele_cache = ModeleUneCoucheCachee(
        input_dim=784,
        hidden_dim=64,
        output_dim=10
    )

    loss_cache, train_err_cache, test_err_cache = entrainer_modele(
        modele=modele_cache,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_rate=0.1,
        epochs=10,
        batch_size=256
    )

    afficher_courbes(loss_cache, train_err_cache, test_err_cache, "Modèle avec couche cachée")
    afficher_erreurs(modele_cache, x_test, y_test, "Erreurs du modèle avec couche cachée")

    return modele_cache, loss_cache, train_err_cache, test_err_cache


def comparer_deux_modeles():
    print("\n--- Entraînement du modèle linéaire ---")
    modele_lineaire, loss_lin, train_err_lin, test_err_lin = lancer_modele_lineaire()

    print("\n--- Entraînement du modèle avec couche cachée ---")
    modele_cache, loss_cache, train_err_cache, test_err_cache = lancer_modele_couche_cachee()

    print("\n==============================")
    print("COMPARAISON FINALE")
    print("==============================")
    print(f"Modèle linéaire - erreur train finale : {train_err_lin[-1]:.4f}")
    print(f"Modèle linéaire - erreur test finale  : {test_err_lin[-1]:.4f}")
    print(f"Modèle caché - erreur train finale    : {train_err_cache[-1]:.4f}")
    print(f"Modèle caché - erreur test finale     : {test_err_cache[-1]:.4f}")

    plt.figure()
    plt.plot(test_err_lin, label="Modèle linéaire")
    plt.plot(test_err_cache, label="Modèle avec couche cachée")
    plt.xlabel("Époque")
    plt.ylabel("Taux d'erreur test")
    plt.title("Comparaison des performances sur MNIST")
    plt.legend()
    plt.show()


while True:
    choix = menu()

    if choix == "1":
        lancer_modele_lineaire()

    elif choix == "2":
        lancer_modele_couche_cachee()

    elif choix == "3":
        comparer_deux_modeles()

    elif choix == "4":
        modele = ModeleDeuxCouchesCachees()

        loss, train_err, test_err = entrainer_modele(
            modele,
            x_train,
            y_train,
            x_test,
            y_test,
            learning_rate=0.1,
            epochs=10
        )

        afficher_courbes(loss, train_err, test_err, "2 couches cachées")

        afficher_erreurs(
            modele,
            x_test,
            y_test,
            "Erreurs du modèle avec deux couches cachées"
        )


    elif choix == "5":

        meilleur_resultat, resultats = grid_search_architecture_complete(
            x_train,
            y_train,
            x_test,
            y_test
        )

    elif choix == "6":
        print("Fin du programme.")
        break

    else:
        print("Choix invalide. Réessaie.")
