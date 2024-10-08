import numpy as np
import os
from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_batch(file):
    dict = unpickle(file)
    data = dict[b'data'].astype(np.float32)
    labels = np.array(dict[b'labels'], dtype=np.int64)
    labels = labels.reshape(labels.shape[0])
    return data, labels


def read_cifar(path):
    print('Reading data from disk')
    data_batches = ["data_batch_" + str(i) for i in range(1, 6)] + ['test_batch']
    flag = True
    for db in data_batches:
        data, labels = read_cifar_batch(os.path.join(path, db))
        if flag:
                DATA = data
                LABELS = labels
                flag = False
        else:
            DATA = np.concatenate((DATA, data), axis=0, dtype=np.float32)
            LABELS = np.concatenate((LABELS, labels), axis=-1, dtype=np.int64)
    return DATA, LABELS


def split_dataset(data, labels, split):
    print(f"Splitting data into train/test with split={split}")
    n = data.shape[0]
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:int(split*n)], indices[int(split*n):]
    data_train, data_test = data[train_idx,:].astype(np.float32), data[test_idx,:].astype(np.float32)
    labels_train, labels_test = labels[train_idx].astype(np.int64), labels[test_idx].astype(np.int64)
    # data_train, data_test, labels_train, labels_test = train_test_split(data, labels,test_size=split, shuffle= True)
    return data_train, labels_train, data_test, labels_test 