import numpy as np
import matplotlib.pyplot as plt
import time


def learn_once_mse(w1, b1, w2, b2, data, targets, lr):
    # Passage en avant (Forward pass)
    a0 = data  # Entrée de la première couche (Input of the first layer)
    z1 = np.matmul(a0, w1) + b1  # Entrée de la couche cachée (Input of the hidden layer)
    a1 = 1 / (1 + np.exp(-z1))  # Sortie de la couche cachée (Output of the hidden layer) - Activation sigmoïde
    z2 = np.matmul(a1, w2) + b2  # Entrée de la couche de sortie (Input of the output layer)
    a2 = 1 / (1 + np.exp(-z2))  # Sortie de la couche de sortie (Output of the output layer) - Activation sigmoïde
    predictions = a2  # Les valeurs prédites sont les sorties de la couche de sortie
    # Calcul de la perte (Mean Squared Error - MSE)
    loss = np.mean(np.square(predictions - targets))
    # Calcul des gradients
    delta2 = predictions - targets
    delta1 = np.dot(delta2, w2.T) * a1 * (1 - a1)  # Gradient pour la couche cachée
    # Mise à jour des poids et des biais en utilisant les gradients
    w2 -= lr * np.dot(a1.T, delta2) / len(data)
    b2 -= lr * np.sum(delta2, axis=0) / len(data)
    w1 -= lr * np.dot(a0.T, delta1) / len(data)
    b1 -= lr * np.sum(delta1, axis=0) / len(data)
    return w1, b1, w2, b2, loss  # Renvoie les poids, les biais et la perte


def one_hot(x):
    n_classes = 10
    return np.eye(n_classes)[x]
    # Convertit les étiquettes en représentation one-hot pour un nombre de classes donné


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    # Applique la fonction softmax aux scores de sortie du réseau pour obtenir des probabilités


def learn_once_cross_entropy(w1, b1, w2, b2, data, targets, learning_rate):
    N = data.shape[0]
    # Passage en avant (Forward pass)
    a0 = data                       # les données sont l'entrée de la première couche
    z1 = np.matmul(a0, w1) + b1     # entrée de la couche cachée
    a1 = 1 / (1 + np.exp(-z1))      # sortie de la couche cachée (fonction d'activation sigmoïde)
    z2 = np.matmul(a1, w2) + b2     # entrée de la couche de sortie
    a2 = softmax(z2)                # sortie de la couche de sortie (fonction d'activation softmax)
    predictions = a2                # les valeurs prédites sont les sorties de la couche de sortie
    # Encodage one-hot des cibles
    oh_targets = one_hot(targets)
    # Calcul de la perte Cross-Entropy
    loss = - np.sum(oh_targets * np.log(predictions + 1e-9)) / N
    # Rétro-propagation (Backward pass)
    dz2 = predictions - oh_targets
    dw2 = np.dot(a1.T, dz2) / N
    db2 = np.sum(dz2, axis=0, keepdims=True) / N  
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * a1 * (1 - a1)
    dw1 = np.dot(a0.T, dz1) / N
    db1 = np.sum(dz1, axis=0, keepdims=True) / N
    # Une étape de descente de gradient
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2
    return w1, b1, w2, b2, loss  # Renvoie les poids, les biais et la perte


def predict_mlp(w1, b1, w2, b2, data):
    # Passage en avant (Forward pass)
    a0 = data                    # les données sont l'entrée de la première couche
    z1 = np.matmul(a0, w1) + b1  # entrée de la couche cachée
    a1 = 1 / (1 + np.exp(-z1))   # sortie de la couche cachée (fonction d'activation sigmoïde)
    z2 = np.matmul(a1, w2) + b2  # entrée de la couche de sortie
    a2 = softmax(z2)             # sortie de la couche de sortie (fonction d'activation softmax)
    predictions = np.argmax(a2, axis=1)
    return predictions


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    # effectue num_epoch d'itérations d'entraînement
    losses = []
    train_accuracies = [0] * num_epoch
    for epoch in range(num_epoch):
        # Effectue une étape d'apprentissage avec la fonction de perte Cross-Entropy
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        losses.append(loss)
        # Prédit les étiquettes pour les données d'entraînement
        labels_pred = predict_mlp(w1, b1, w2, b2, data_train)
        accuracy = np.mean(labels_pred == labels_train)  # Calcule l'exactitude des prédictions
        train_accuracies[epoch] = accuracy
        print(f"Epoch loss [{epoch+1}/{num_epoch}] : {loss} --- accuracy : {accuracy}")
        # Met à jour les poids et les biais pour l'itération suivante
        # Passe les paramètres mis à jour à l'itération suivante
    return w1, b1, w2, b2, train_accuracies


def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    # teste le réseau sur l'ensemble de données de test
    labels_pred = predict_mlp(w1, b1, w2, b2, data_test)  # Prédiction des étiquettes pour les données de test
    test_accuracy = np.mean(labels_pred == labels_test)  # Calcul de la précision des prédictions
    return test_accuracy


def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, lr, num_epoch):
    """Entraîne un MLP avec les paramètres donnés."""
    print("Starting Training...")
    tic = time.time()
    d_in = data_train.shape[1]  # Type d'entrée : tableau de données d'entraînement (nombre de colonnes)
    d_out = len(set(labels_train))  # Type de sortie : nombre d'étiquettes uniques dans les données d'entraînement
    # Initialisation aléatoire des poids et des biais du réseau
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # Poids de la première couche
    b1 = np.zeros((1, d_h))  # Biais de la première couche
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # Poids de la deuxième couche
    b2 = np.zeros((1, d_out))  # Biais de la deuxième couche
    w1, b1, w2, b2, accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, lr, num_epoch)
    toc = time.time()
    print("Finished Training.")
    print('Time taken for training: ', toc-tic)
    print("Starting Testing...")
    tic = time.time()
    accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    toc = time.time()
    print("Finished Testing.")
    print('Time taken for Testing: ', toc-tic)
    return accuracies, accuracy


def plot_accuracy_versus_epoch(accuracies):
    plt.figure(figsize=(18, 10))
    plt.plot(accuracies, 'o-b')
    plt.title("Variation of the accuracy over the epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(axis='both', which='both')
    plt.savefig('./results/mlp1.png')
