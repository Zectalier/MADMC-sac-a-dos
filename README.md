# Projet MADMC 2023-2024

## Description
Projet "Élicitation incrémentale et recherche locale pour le problème du sac à dos multi-objectifs" pour l'UE de MADMC 2023-2024.

## Sommaire
- [Installation](#installation)
- [Usage](#usage)

## Installation
Pour installer le projet, il suffit de télécharger le code source du projet.
Il est nécessaire au préalable d'avoir les packages suivants:
- **gurobipy**
- **numpy**
- **pandas**
- **matplotlib** (pour la visualisation de la recherche locale de Pareto)

## Utilisation
Pour lancer le programme, exécutez **main.py**.

Vous pouvez utiliser les arguments optionnels suivants:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `-p`   | 1 | Spécifie quel processus exécuter. |
| `-c`   | 3 | Spécifie le nombre de critères. |
| `-o`   | 50 | Spécifie le nombre d'objets. |
| `-f`   | `WS` | Spécifie la fonction d'agrégation. |
| `-eps` | 0.001 | Spécifie la valeur epsilon à laquelle l'élicitation incrémentale doit s'arrêter. |
| `-a`   | `False` | Spécifie si toutes les fonctions et procédures doivent être exécutées. |
| `-l`   | `False` | Spécifie si les résultats doivent être sauvegardés dans un fichier. |
