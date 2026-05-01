"""
main.py — Point d'entrée principal du projet ML (SM604 — EFREI Paris)
CNN : De la classification MNIST à la détection de cancers du sein.

Utilisation : venv/bin/python main.py
"""

import os
os.makedirs('rapport', exist_ok=True)


def menu_principal():
    while True:
        print("\n" + "=" * 55)
        print("   PROJET ML — CNN  (SM604 EFREI Paris)")
        print("=" * 55)
        print("  1  - Partie 1 : Classification MNIST     (NumPy pur)")
        print("  2  - Partie 2 : CIFAR-10 + Convolutions + CNN PyTorch")
        print("  3  - Partie 3 : Détection cancer du sein (CBIS-DDSM)")
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

        elif choix == '0':
            print("Fin du programme.")
            break

        else:
            print("Choix invalide.")


if __name__ == '__main__':
    menu_principal()
