import Sparse.SparseProcess as process
import Sparse.SparseCalculations as calc
import numpy as np
import Sparse.LargeData as data


# use the cross validation set to determine the optimal num_features and lambda
def determine_parameters():
    # set constants
    num_features = [16]
    lam = [0, 0.1, 0.2, 0.5]
    len_features = len(num_features)
    len_lam = len(lam)
    errors = np.zeros((len_features, len_lam)) # num_features x lam array

    # load data
    Y, R = data.load_sparse_matrices('Data/training_input.txt')
    Y_cv, R_cv = data.load_sparse_matrices('Data/cv_input.txt')

    # add error into error array for each num_feature and lambda
    for i in range(len_features):
        for j in range(len_lam):
            num_feat = num_features[i]
            (X, Theta, Y_norm, Y_mean) = process.train(Y, R, num_feat, lam[j])
            num_books = np.shape(X)[0]

            Y_cv_norm = calc.normalize(Y_cv, R_cv, Y_mean)[0]
            cost = calc.cost_func(X, Theta, Y_cv_norm, R_cv, 0, num_books)[0]
            errors[i, j] = cost / len(R_cv.data)
            print(errors)


# save the learned parameters
def save_parameters():
    # load data
    Y, R = data.load_sparse_matrices('Data/processed_input.txt')

    # define constants
    num_features = 16
    lam = 0

    # train model
    (X, Theta, Y_norm, Y_mean) = process.train(Y, R, num_features, lam)

    # set num_books
    num_books = np.shape(X)[0]

    # calculate and print the error
    # note: when calculating error, use lambda = 0 and Y = Y - Ymean from every col
    error = calc.cost_func(X, Theta, Y_norm, R, 0, num_books)[0]
    print(error / len(R.data))

    # save parameters
    # np.savetxt('learned_X_150.txt', X)
    # np.savetxt('learned_Theta_150.txt', Theta)
    # np.savetxt('Data/Y_mean.txt', Y_mean)


if __name__ == "__main__":
    save_parameters()
