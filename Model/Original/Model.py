from Original import Calculations as calc
from Original import ReadData
from Original import Process
import numpy as np


# use the cross validation set to determine the optimal num_features and lambda
def determine_parameters():
    # defined constants
    num_features = [32]
    lam = [2]
    len_features = len(num_features)
    len_lam = len(lam)
    cv_errors = np.zeros((len_features, len_lam)) # num_features x lam array

    # load data
    data = ReadData.Data()
    data.load_data()

    # add error into error array for each num_feature and lambda
    for i in range(len_features):
        for j in range(len_lam):
            num_feat = num_features[i]
            (X, Theta, Y_norm, Y_mean) = Process.train(data.train_Y, data.train_R, num_feat, lam[j])
            num_users = np.shape(Theta)[0]
            expanded_mean = np.outer(Y_mean, np.ones(num_users))

            # when calculating error, use lambda = 0 and Y = Y - Ymean from every col
            error = calc.cost_func(X, Theta, data.cv_Y - expanded_mean, data.cv_R, 0)[0]
            num_cv = data.cv_thres - data.train_thres
            cv_errors[i, j] = (1 / num_cv) * error
            print(cv_errors)


# save the learned parameters
def save_parameters(X_filename, Theta_filename):
    # load data
    data = ReadData.Data()
    data.load_data()

    # define constants
    num_features = 32
    lam = 2

    # train model
    (X, Theta, Y_norm, Y_mean) = Process.train(data.all_Y, data.all_R, num_features, lam)

    # process data for error calculation
    num_users = np.shape(Theta)[0]
    expanded_mean = np.outer(Y_mean, np.ones(num_users))

    # when calculating error, use lambda = 0 and Y = Y - Ymean from every col
    error = calc.cost_func(X, Theta, data.all_Y - expanded_mean, data.all_R, 0)[0]
    print(error / data.num_all)

    # save parameters
    np.savetxt(X_filename, X)
    np.savetxt(Theta_filename, Theta)


if __name__ == '__main__':
    save_parameters('Data/learned_X_150.txt', 'Data/learned_Theta_150.txt')
