# Image classification

### Résumé du Code de Manipulation des Données CIFAR-10

Le code est dédié à la manipulation des données du jeu de données CIFAR-10, effectuant des opérations de lecture, de séparation des ensembles de données, et de chargement.

#### Composants principaux :
1. **`unpickle(file)` :** Cette fonction lit un fichier binaire au format CIFAR-10 et le charge dans un dictionnaire via la bibliothèque `pickle`.

2. **`read_cifar_batch(file)` :** Charge un batch spécifique de données CIFAR-10 en extrayant les données et les étiquettes du fichier.

3. **`read_cifar(path)` :** Lecture de l'ensemble des données CIFAR-10. Cette fonction parcourt tous les batchs disponibles et concatène les données et les étiquettes en un seul jeu de données et d'étiquettes.

4. **`split_dataset(data, labels, split)` :** Divise l'ensemble de données et les étiquettes en ensembles de données d'entraînement et de test. Les données sont divisées en pourcentage spécifié avec un mélange aléatoire.

#### Fonctionnalités :
- Lecture des fichiers CIFAR-10 depuis le disque.
- Organisation des données lues en ensembles d'entraînement et de test, tout en conservant la correspondance entre données et étiquettes.
- Fonctionnalité de division des données en un ensemble d'entraînement et un ensemble de test selon un pourcentage spécifié.

Ces fonctions offrent des fonctionnalités essentielles pour charger, traiter et diviser les données CIFAR-10 en ensembles d'entraînement et de test, ce qui est fondamental pour l'apprentissage des modèles sur ce jeu de données.


### Résumé du code K-NN

Le code fournit une implémentation de l'algorithme k-NN (k-Nearest Neighbors) pour la classification.

#### Composants principaux :
1. **`distance_matrix(train, test)` :** Cette fonction calcule la matrice des distances entre les ensembles d'entraînement (`train`) et de test (`test`). Utilise la racine de la somme des carrés des différences entre les vecteurs.
   
2. **`mode(x)` :** La fonction détermine le mode d'un ensemble `x`, c'est-à-dire la valeur la plus fréquente.

3. **`knn_predict(dists, labels_train, k)` :** Basé sur la matrice de distances, cette fonction prédit les étiquettes pour les données de test en utilisant k voisins les plus proches.

4. **`evaluate_knn(data_train, labels_train, data_test, labels_test, k)` :** Évalue l'algorithme k-NN pour une valeur de `k` donnée, calculant la précision (accuracy) de la prédiction.

5. **`evaluate_knn_for_k(data_train, labels_train, data_test, labels_test, k_max)` :** Évalue l'algorithme k-NN pour une plage de valeurs de `k` (de 1 à `k_max`). Retourne une liste de précisions pour chaque `k`.

6. **`plot_accuracy_versus_k(accuracies)` :** Trace un graphique illustrant la variation de la précision en fonction de la valeur de `k` dans l'algorithme k-NN. Les résultats sont enregistrés sous forme d'image.

Ces fonctions permettent de calculer la précision de l'algorithme k-NN pour différents k, puis de visualiser graphiquement la variation de la précision en fonction de k.


# Commenting the result of knn

En results/knn.png , vous allez trouver le graphe permettant de conclure que la valeur maximale de l'accuracy correspand à 0.342 qui correspand à une valeur de voisins égale à 7. Je précisé meme que plus en augmente le k plus la courbe d'accuracy diminue.

### Backpropagation in a Neural Network


# Neural Network Training and Testing Overview

Ce code Python implémente un réseau de neurones multicouches (MLP) pour l'apprentissage supervisé. Il utilise des fonctions pour l'entraînement, le test, la prédiction et la visualisation des performances du réseau.

## Principales Fonctions :

### `learn_once_mse`
- Calcule l'erreur quadratique moyenne (MSE) et effectue une étape de descente de gradient.

### `learn_once_cross_entropy`
- Calcule la perte d'entropie croisée et effectue une étape de descente de gradient avec la fonction softmax pour la sortie.

### `predict_mlp`
- Prédit les étiquettes pour les données en utilisant le réseau entraîné.

### `train_mlp`
- Entraîne le MLP pendant un nombre spécifié d'époques en utilisant la fonction `learn_once_cross_entropy`. Affiche les pertes et les précisions d'entraînement à chaque époque.

### `test_mlp`
- Évalue les performances du MLP sur un ensemble de test.

### `run_mlp_training`
- Orchestre le processus d'entraînement et de test du MLP avec des paramètres spécifiques.

### `plot_accuracy_versus_epoch`
- Trace et sauvegarde un graphique montrant l'évolution de la précision sur les époques.

## Utilisation :
1. Initialisation aléatoire des poids et des biais du réseau.
2. Entraînement du réseau sur les données d'entraînement.
3. Évaluation des performances du réseau sur les données de test.
4. Visualisation de la précision du réseau sur les époques.

Le code utilise les bibliothèques `numpy` pour les calculs matriciels, `matplotlib` pour la visualisation et `time` pour mesurer le temps d'exécution.


# Commenting the result of mlp

Nous remarquons que la valeur de l'accuracy est faible (~ 0.10) ce qui été naturelement prévu. Nous remarquons aussi que la courbe de l'accuracy augmente jusqu'a une valeur de 0.10. Cette evalution nous permet de conclure que le modele est faible.


--- 