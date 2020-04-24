from datetime import datetime as datetime
import scipy.sparse as sp
import numpy as np


# note: params are numpy nd arrays, Y and R are sparse matrices
def batch_gradient(params, Y, R, num_users, num_books, num_features, lam):
    print(datetime.now(), "Entered batch_gradient")

    # unroll parameters
    X = np.reshape(params[:num_books * num_features], (num_books, num_features), order='F')
    Theta = np.reshape(params[num_books * num_features:], (num_users, num_features), order='F')

    # calculate cost
    cost, error = cost_func(X, Theta, Y, R, lam, num_books)

    # calculate gradients
    X_grad = error.dot(Theta) + (lam * X)
    Theta_grad = error.transpose().dot(X) + (lam * Theta)

    print(datetime.now(), "Finished gradient function")

    return cost, np.hstack((X_grad.flatten(order='F'), Theta_grad.flatten(order='F')))


def cost_func(X, Theta, Y, R, lam, num_books):
    print(datetime.now(), "Entered cost function")

    # calculate hypothesis
    # note: requires matrix product between two sparse matrices
    rows, cols = R.nonzero()
    data = []
    for i in range(len(rows)):
        data.append(X[rows[i], :] @ Theta[cols[i], :])
    hypothesis = sp.csr_matrix((data, (rows, cols)), shape=R.shape)

    # calculates cost
    error = hypothesis - Y
    reg = (lam / 2) * (np.sum(Theta ** 2) + np.sum(X ** 2))
    cost = (1 / 2) * error.power(2).sum() + reg

    print(datetime.now(), cost)

    return cost, error


def normalize(Y, R, Y_mean = None):
    # calculate Y_mean if it is not passed in
    if Y_mean is None:
        Y_mean = Y.sum(axis=1).A1
        counts = np.diff(Y.indptr)
        Y_mean[counts.nonzero()] = Y_mean[counts.nonzero()] / counts[counts.nonzero()]

    # calculate Y_norm
    Y_csc = Y.tocsc()
    Y_csc.data = Y_csc.data - np.take(Y_mean, Y_csc.indices)
    Y_norm = Y_csc.tocsr()

    return Y_norm, Y_mean


