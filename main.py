from read_cifar import read_cifar, split_dataset
from knn import evaluate_knn_for_k, plot_accuracy_versus_k
import matplotlib.pyplot as plt
from mlp import run_mlp_training, plot_accuracy_versus_epoch



if __name__=="__main__":
    # data, labels = read_cifar("data\cifar-10-batches-py")
    #split = 0.9
    #data_train, labels_train, data_test, labels_test = split_dataset(data,labels,split)
    # data_train, data_test = data_train/255.0, data_test/255.0
    # kmax = 20
    #accuracies = evaluate_knn_for_k(data_train, labels_train, data_test, labels_test,kmax)
    # accuracies = [0.351,
    #             0.31316666666666665,
    #             0.329,
    #             0.33666666666666667,
    #             0.33616666666666667,
    #             0.3413333333333333,
    #             0.343,
    #             0.3428333333333333,
    #             0.341,
    #             0.3335,
    #             0.3325,
    #             0.3328333333333333,
    #             0.33016666666666666,
    #             0.3295,
    #             0.32766666666666666,
    #             0.3285,
    #             0.327,
    #             0.32716666666666666,
    #             0.32916666666666666,
    #             0.3305]
    #plot_accuracy_versus_k(accuracies)
    ####################################
    # parameters of the MLP :
    split_factor = 0.9
    data, labels = read_cifar("data\cifar-10-batches-py")
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split=split_factor)
    # print(len(data_test), len(data_train))
    data_train, data_test = data_train/255.0, data_test/255.0 # normalize ou data
    d_h = 64
    lr = 0.1
    num_epoch=100
    accuracies, _ = run_mlp_training(data_train, labels_train, data_test,
                                                       labels_test, d_h, lr, num_epoch)
    # accuracies = [0.08788888888888889, 0.08990740740740741, 0.09135185185185185, 0.09296296296296297, 0.09514814814814815, 0.09631481481481481, 0.09724074074074074, 0.09787037037037037, 0.09820370370370371, 0.09883333333333333, 0.09844444444444445, 0.09859259259259259, 0.09857407407407408, 0.09885185185185186, 0.09872222222222223, 0.09855555555555555, 0.09872222222222223, 0.09883333333333333, 0.0989074074074074, 0.09881481481481481, 0.0987962962962963, 0.09898148148148148, 0.09916666666666667, 0.09938888888888889, 0.09961111111111111, 0.09975925925925926, 0.09975925925925926, 0.1, 0.10003703703703704, 0.09998148148148148, 0.10007407407407408, 0.10011111111111111, 0.10001851851851852, 0.10014814814814815, 0.10012962962962962, 0.09998148148148148, 0.1000925925925926, 0.1000925925925926, 0.10007407407407408, 0.10005555555555555, 0.10014814814814815, 0.10018518518518518, 0.1002037037037037, 0.10018518518518518, 0.10016666666666667, 0.10011111111111111, 0.10016666666666667, 0.10012962962962962, 0.10007407407407408, 0.10005555555555555, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09998148148148148, 0.09998148148148148, 0.09996296296296296, 0.09996296296296296, 0.09996296296296296, 0.09994444444444445, 0.09994444444444445, 0.09994444444444445, 0.0999074074074074, 0.09994444444444445, 0.09996296296296296, 0.09996296296296296, 0.09996296296296296, 0.09998148148148148, 0.09996296296296296, 0.09998148148148148, 0.1, 0.1, 0.10003703703703704, 0.10003703703703704, 0.10005555555555555, 0.10007407407407408, 0.10007407407407408, 0.10007407407407408, 0.10003703703703704, 0.10001851851851852, 0.10003703703703704, 0.10003703703703704, 0.10003703703703704, 0.10001851851851852, 0.10001851851851852, 0.10003703703703704, 0.10003703703703704, 0.10005555555555555, 0.10007407407407408, 0.10007407407407408, 0.10007407407407408, 0.10007407407407408, 0.10005555555555555, 0.10005555555555555, 0.10005555555555555, 0.10007407407407408, 0.10007407407407408, 0.10007407407407408, 0.10007407407407408]
    # print(accuracies)
    plot_accuracy_versus_epoch(accuracies)
    
    
    
    


# Result for k = 1
# Reading data from disk
# [INFO] Splitting data into train/test with split=70
# [INFO] Training set has 42000 samples and testing set has 18000 samples.
# [INFO] Time taken 0
# Evaluating the k-NN with k = 1
# Computing distance matrix between train and test sets
# finished calculating dists
# Running the prediction using k-NN with k = 1
# [INFO] computing accuracy of the predictions
# accuracy = 0.3388888888888889


# Reading data from disk
# [INFO] Splitting data into train/test with split=70
# [INFO] Training set has 42000 samples and testing set has 18000 samples.
# [INFO] Time taken 0
# Evaluating the k-NN with k = 3
# Computing distance matrix between train and test sets
# finished calculating dists
# Running the prediction using k-NN with k = 3
# [INFO] computing accuracy of the predictions
# 0.3308333333333333



