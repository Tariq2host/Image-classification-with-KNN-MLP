import numpy as np
import matplotlib.pyplot as plt



def distance_matrix(train, test):
    print('Computing distance matrix between train and test sets') 
    dists = np.sqrt(-2 * np.matmul(train, test.T) + 
                    np.sum(train*train, axis=1, keepdims=True) + 
                    np.sum(test*test, axis=1, keepdims=True).T)
    print('finished calculating dists')
    return dists


def mode(x):
    vals, counts = np.unique(x, return_counts=True)
    return vals[np.argmax(counts)]


def knn_predict(dists, labels_train, k):
    # dists has shape [num_train, num_test]
    indexes_of_knn = np.argsort(dists, axis=0)[0:k, :]
    nearest_labels_pred = labels_train[indexes_of_knn]
    labels_pred = np.array([ mode(label) for label in nearest_labels_pred.T ])
    return labels_pred


def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    print(f"Evaluating the k-NN with k = {k}")
    dists = distance_matrix(data_train, data_test)
    labels_pred = knn_predict(dists, labels_train, k)
    accuracy = np.sum(labels_pred == labels_test) / len(labels_test)
    return accuracy


def evaluate_knn_for_k(data_train, labels_train, data_test, labels_test, k_max):
    print(f"Evaluating the k-NN for k in range [1, {k_max}]")
    accuracies = [0] * k_max
    dists = distance_matrix(data_train, data_test)
    for k in range(1, k_max + 1):
        labels_pred = knn_predict(dists, labels_train, k)
        accuracy = np.sum(labels_pred == labels_test) / len(labels_test)
        accuracies[k - 1] = accuracy

    return accuracies


def plot_accuracy_versus_k(accuracies):
    k = len(accuracies)
    fig = plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, k+1, 1), accuracies)
    plt.title("Variation of the accuracy as a function of k")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")
    # ax = fig.gca()
    # ax.set_xticks(np.arange(1, k+1, 1))
    plt.grid(axis='both', which='both')
    plt.savefig('./results/knn.png')
