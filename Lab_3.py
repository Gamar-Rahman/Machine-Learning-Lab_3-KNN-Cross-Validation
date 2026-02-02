import numpy as np
np.random.seed(1)

# -----------------------------
# Problem 1.1
# -----------------------------

def euclidean_dist(X_test, X_train):
    dists = (
        np.sum(X_test ** 2, axis=1, keepdims=True)
        + np.sum(X_train ** 2, axis=1, keepdims=True).T
        - 2 * X_test @ X_train.T
    )
    return dists


def find_k_neighbors(dists, Y_train, k):
    num_test = dists.shape[0]
    neighbors = np.zeros((num_test, k))
    sorted_idx = dists.argsort(axis=1)

    for i in range(num_test):
        neighbors[i] = Y_train[sorted_idx[i][:k]]

    return neighbors


def knn_predict(X_test, X_train, Y_train, k):
    num_test = X_test.shape[0]
    Y_pred = np.zeros(num_test, dtype=int)

    dists = euclidean_dist(X_test, X_train)
    neighbors = find_k_neighbors(dists, Y_train, k)

    for i in range(num_test):
        values, counts = np.unique(neighbors[i], return_counts=True)
        Y_pred[i] = values[np.argmax(counts)]

    return Y_pred


def compute_error_rate(ypred, ytrue):
    return (ypred != ytrue).mean() * 100


# -----------------------------
# Problem 1.2
# -----------------------------

def split_nfold(num_examples, n):
    np.random.seed(1)

    idx = np.random.permutation(num_examples).tolist()
    fold_size = num_examples // n

    train_sets = []
    validation_sets = []

    for i in range(n):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == n - 1:
            end = num_examples

        val_set = idx[start:end]
        train_set = idx[:start] + idx[end:]

        train_sets.append(train_set)
        validation_sets.append(val_set)

    return train_sets, validation_sets


def cross_validation(classifier, X, Y, n, *args):
    np.random.seed(1)

    errors = []
    size = X.shape[0]

    train_sets, val_sets = split_nfold(size, n)

    for train_idx, val_idx in zip(train_sets, val_sets):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = Y[train_idx]
        y_val = Y[val_idx]

        ypred = classifier(X_val, X_train, y_train, *args)
        errors.append(compute_error_rate(ypred, y_val))

    return np.mean(errors)


# -----------------------------
# Problem 2
# -----------------------------

def problem2():
    import os
    import gzip

    DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

    def maybe_download(filename):
        if not os.path.exists(filename):
            from urllib.request import urlretrieve
            urlretrieve(DATA_URL + filename, filename)

    def load_images(filename):
        maybe_download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28) / np.float32(256)

    def load_labels(filename):
        maybe_download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    Xtrain = load_images('train-images-idx3-ubyte.gz')
    ytrain = load_labels('train-labels-idx1-ubyte.gz')

    size = 1000
    k = 1

    cvXtrain = Xtrain[:size]
    cvytrain = ytrain[:size]

    trial_folds = [3, 10, 50, 100, 1000]
    cverror_rates = np.zeros(len(trial_folds))

    for i, f in enumerate(trial_folds):
        cverror_rates[i] = cross_validation(
            knn_predict, cvXtrain, cvytrain, f, k
        )

    return cverror_rates


# -----------------------------
# Problem 3
# -----------------------------

def problem3():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=1
    )

    list_ks = np.arange(1, 101)
    err_ks = np.zeros(len(list_ks))

    for k in list_ks:
        err_ks[k - 1] = cross_validation(knn_predict, X_train, Y_train, 10, k)

    best_k = np.argmin(err_ks) + 1

    y_pred = knn_predict(X_test, X_train, Y_train, best_k)
    err_test = compute_error_rate(y_pred, Y_test)

    nclass = len(np.unique(Y_test))
    cm = np.zeros((nclass, nclass), dtype=int)

    for i in range(len(Y_test)):
        cm[Y_test[i], y_pred[i]] += 1

    return err_ks, best_k, err_test, cm

