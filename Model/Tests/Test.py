import numpy as np
from Original import Calculations as calc
from scipy.optimize import minimize
from Original import Process


def test_model():
    # load datasets
    print("Data/Loading datasets")
    Y = np.loadtxt("Data/movie_dataset", dtype=int, skiprows=5)
    R = np.loadtxt("Data/movie_dataset_r", dtype=bool, skiprows=5)
    movies = np.genfromtxt("Data/movie_ids.txt", dtype=str, delimiter="]")

    # initialize my ratings
    my_ratings = np.zeros((1682, 1))
    my_ratings[1] = 4
    my_ratings[98] = 2
    my_ratings[7] = 3
    my_ratings[12]= 5
    my_ratings[54] = 4
    my_ratings[64]= 5
    my_ratings[66]= 3
    my_ratings[69] = 5
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355]= 5

    Y = np.hstack((my_ratings, Y))
    R = np.hstack((my_ratings > 0, R))

    # train model
    (X, Theta, Y_norm, Y_mean) = Process.train(Y, R, 10, 10)

    # calculate my predictions
    predict = X @ Theta.transpose()
    my_predict = predict[:, 0] + Y_mean

    # print my predictions
    sorted = np.argsort(-1 * my_predict)
    for i in range(10):
        j = sorted[i]
        print('Predicting rating %f for movie %s' % (my_predict[j], movies[j]))


def test_cost():
    # load data
    Y = np.loadtxt("Data/movie_dataset", dtype=int, skiprows=5)
    R = np.loadtxt("Data/movie_dataset_r", dtype=bool, skiprows=5)
    X = np.loadtxt("Data/paramsX", skiprows=5)
    Theta = np.loadtxt("Data/paramsTheta", skiprows=5)

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

# def test_gradient():
#     params = np.loadtxt("Data/parameters", skiprows=5)
#     Y = np.loadtxt("Data/resultsY", skiprows=5)
#     R = np.loadtxt("Data/resultsR", dtype=bool, skiprows=5)
#     num_users = 5
#     num_movies = 4
#     num_features = 3
#     lam = 1.5
#
#     result = calc.batch_gradient(params, Y, R, num_users, num_movies, num_features, lam)
#     print(result)


def test_train_singular():
    # load datasets
    Y = np.loadtxt("Data/movie_dataset", dtype=int, skiprows=5)
    R = np.loadtxt("Data/movie_dataset_r", dtype=bool, skiprows=5)
    movies = np.genfromtxt("Data/movie_ids.txt", dtype=str, delimiter="]")

    # train model
    (X, Theta, Y_norm, Y_mean) = Process.train(Y, R, 10, 10)

    # initialize my ratings
    my_ratings = np.zeros(shape=np.shape(Y_mean))
    my_theta = np.random.rand(10)

    my_ratings[1] = 4
    my_ratings[98] = 2
    my_ratings[7] = 3
    my_ratings[12]= 5
    my_ratings[54] = 4
    my_ratings[64]= 5
    my_ratings[66]= 3
    my_ratings[69] = 5
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355]= 5

    my_norm = my_ratings - Y_mean
    my_R = np.array(my_ratings, dtype=bool)
    args = (X, my_norm, my_R, 10)

    # train latent features for my ratings
    result = minimize(fun=calc.user_gradient, x0=my_theta, args=args, method='CG', jac=True, options={'disp':True})
    my_theta = result.x

    # calculate my predictions
    predict = X @ my_theta.transpose()
    my_predict = predict + Y_mean

    # print my predictions
    sorted = np.argsort(-1 * my_predict)
    for i in range(10):
        j = sorted[i]
        print('Predicting rating %f for movie %s' % (my_predict[j], movies[j]))


if __name__ == '__main__':
    test_train_singular()
