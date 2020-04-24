import Sparse.SparseCalculations as calc
from scipy.optimize import minimize
import scipy.sparse as sp
import numpy as np

# note: Y and R are sparse matrices, X and Theta are dense nd-arrays
def train(Y, R, num_features, lam):
    # normalize Y
    (Y_norm, Y_mean) = calc.normalize(Y, R)

    # set constants num_users and num_books
    dims = Y.shape
    num_users = dims[1]
    num_books = dims[0]

    # X is book latency vectors (don't really change regularly)
    X = np.random.rand(num_books, num_features)
    # Theta is user latency vectors (add/change along with users)
    Theta = np.random.rand(num_users, num_features)

    # prepare parameters for minimize function
    args = (Y_norm, R, num_users, num_books, num_features, lam)
    params = np.hstack((X.flatten(order='F'), Theta.flatten(order='F')))

    # minimize using method='CG' or method='BFGS'
    result = minimize(fun=calc.batch_gradient, x0=params, args=args, method='CG', jac=True, options={'maxiter':150, 'disp':True})
    factors = result.x

    # reshape X and Theta
    X = np.reshape(factors[:num_books * num_features], (num_books, num_features), order='F')
    Theta = np.reshape(factors[num_books * num_features:], (num_users, num_features), order='F')

    return X, Theta, Y_norm, Y_mean
