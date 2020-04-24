import numpy as np
import Sparse.SparseProcess as process
import Sparse.SparseCalculations as calc
import scipy.sparse as sp


def test_model():
    # load datasets
    print("Loading datasets")
    Y = sp.csr_matrix(np.loadtxt("movie_dataset", dtype=int, skiprows=5))
    R = sp.csr_matrix(np.loadtxt("movie_dataset_r", dtype=bool, skiprows=5))
    movies = np.genfromtxt("movie_ids.txt", dtype=str, delimiter="]")

    # initialize my ratings
    my_ratings = sp.csr_matrix((1682, 1))
    my_ratings_r = sp.csr_matrix((1682, 1), dtype=bool)
    my_ratings[1 - 1] = 4
    my_ratings[98 - 1] = 2
    my_ratings[7 - 1] = 3
    my_ratings[12 - 1]= 5
    my_ratings[54 - 1] = 4
    my_ratings[64 - 1]= 5
    my_ratings[66 - 1]= 3
    my_ratings[69 - 1] = 5
    my_ratings[183 - 1] = 4
    my_ratings[226 - 1] = 5
    my_ratings[355 - 1]= 5
    my_ratings_r[1 - 1] = True
    my_ratings_r[98 - 1] = True
    my_ratings_r[7 - 1] = True
    my_ratings_r[12 - 1]= True
    my_ratings_r[54 - 1] = True
    my_ratings_r[64 - 1]= True
    my_ratings_r[66 - 1]= True
    my_ratings_r[69 - 1] = True
    my_ratings_r[183 - 1] = True
    my_ratings_r[226 - 1] = True
    my_ratings_r[355 - 1]= True

    Y = sp.hstack((my_ratings, Y), format='csr')
    R = sp.hstack((my_ratings_r, R), format='csr')

    # train model
    (X, Theta, Y_norm, Y_mean) = process.train(Y, R, 10, 10)

    # calculate my predictions
    predict = X @ Theta.transpose()
    my_predict = predict[:, 0] + Y_mean

    # print my predictions
    sorted = np.argsort(-1 * my_predict)
    for i in range(10):
        j = sorted[i]
        print('Predicting rating %f for movie i: %s' % (my_predict[j], movies[j]))


def test_cost():
    # load data
    Y = sp.csr_matrix(np.loadtxt("movie_dataset", dtype=int, skiprows=5))
    R = sp.csr_matrix(np.loadtxt("movie_dataset_r", dtype=bool, skiprows=5))
    X = np.loadtxt("paramsX", skiprows=5)
    Theta = np.loadtxt("paramsTheta", skiprows=5)

    # set up parameters
    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]
    params = np.hstack((X.flatten(order='F'), Theta.flatten(order='F')))

    # run cost function
    print(calc.batch_gradient(params, Y, R, num_users, num_movies, num_features, 1.5))


def test_gradient():
    # load data
    params = np.loadtxt("parameters", skiprows=5)
    Y = sp.csr_matrix(np.loadtxt("resultsY", skiprows=5))
    R = sp.csr_matrix(np.loadtxt("resultsR", dtype=bool, skiprows=5))

    # assign constants
    num_users = 5
    num_movies = 4
    num_features = 3
    lam = 0

    # run gradient descent
    cost, params = calc.batch_gradient(params, Y, R, num_users, num_movies, num_features, lam)

    # print results
    print(params)
    print(cost)


def test_normalize():
    #initialize matrix
    Y = sp.csr_matrix([[1,0,3],[4,0,6],[7,0,9]])

    # normalize matrix
    Y_norm, Y_mean = calc.normalize(Y, Y)

    # print outputs
    print(Y_norm.toarray())
    print(Y_mean)


if __name__ == '__main__':
    test_model()
