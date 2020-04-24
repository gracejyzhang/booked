import numpy as np
from datetime import datetime as datetime


def batch_gradient(params, Y, R, num_users, num_books, num_features, lam):
    # unroll parameters
    X = np.reshape(params[:num_books * num_features], (num_books, num_features), order='F')
    Theta = np.reshape(params[num_books * num_features:], (num_users, num_features), order='F')

    # determine cost
    cost, error = cost_func(X, Theta, Y, R, lam)

    # determine gradients
    X_grad = error.dot(Theta) + (lam * X)
    Theta_grad = error.transpose().dot(X) + (lam * Theta)

    return cost, np.hstack((X_grad.flatten(order='F'), Theta_grad.flatten(order='F')))


def cost_func(X, Theta, Y, R, lam):
    print(datetime.now(), "Entered cost function")

    # calculate cost
    error = (X @ Theta.transpose() - Y) * R
    reg = (lam / 2) * (np.sum(Theta**2) + np.sum(X**2))
    cost = (1 / 2) * np.sum(error**2) + reg

    print(datetime.now(), cost)
    return cost, error


# note: Theta is a 1d num_features array
def user_gradient(Theta, X, Y, R, lam):
    cost, error = cost_func(X, Theta, Y, R, lam)
    Theta_grad = error.transpose().dot(X) + (lam * Theta)
    return cost, Theta_grad


def normalize(Y, R):
    # initialize Y_mean and Y_norm with correct shapes
    dims = np.shape(Y)
    Y_mean = np.zeros(dims[0])
    Y_norm = np.zeros(np.shape(Y))

    # complete Y_mean and Y_norm
    for i in range(dims[0]):
        rated = np.nonzero(R[i, :])
        Y_mean[i] = Y[i, rated].mean()
        Y_norm[i, rated] = Y[i, rated] - Y_mean[i]

    return Y_norm, Y_mean

