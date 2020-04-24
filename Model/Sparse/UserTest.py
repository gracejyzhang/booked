import numpy as np
from Original import Calculations as calc
from scipy.optimize import minimize
from datetime import datetime as datetime
import pandas as pd


def add_user(book_filename, X_filename, Y_mean_filename):
    print(datetime.now())

    # load data and define constants
    books = pd.read_csv(book_filename, dtype=str, delimiter=';', skiprows=1, usecols=[1,2], quotechar='"', encoding='latin-1').to_numpy(dtype=str)
    X = np.loadtxt(X_filename, delimiter=' ', skiprows=0)
    Y_mean = np.loadtxt(Y_mean_filename)
    num_features = 16
    lam = 0

    # initialize my ratings (Y) and Theta
    Y = np.zeros(shape=np.shape(Y_mean))
    Theta = np.random.rand(num_features)
    ratings = [['all quiet on the western front','erich',10], ['catch 22','heller',9],
               ['crime and punishment','fyodor', 4], ['the blind assassin','atwood', 4],
               ['slaughterhouse five','vonnegut', 10], ['cat\'s cradle','vonnegut', 7],
               ['a tale of two cities','dickens', 2]]

    # insert ratings into Y
    for i in range(np.shape(books)[0]):
        book = books[i, :]
        title = book[0].lower().replace('-',' ')
        author = book[1].lower()
        for rating in ratings:
            if rating[0] in title and rating[1] in author:
                Y[i] = rating[2]

    # calculate Y_norm and R
    Y_norm = Y - Y_mean
    R = np.array(Y, dtype=bool)

    # train latent features for user
    args = (X, Y_norm, R, lam)
    result = minimize(fun=calc.user_gradient, x0=Theta, args=args, method='CG', jac=True, options={'disp':True})
    Theta = result.x

    # calculate predictions
    predict = X @ Theta.transpose()
    my_predict = predict + Y_mean

    print(datetime.now())

    # define books of interest
    book_list = [['a clockwork orange', 'anthony'],['lord of the flies', 'william'],
                 ['nineteen eighty four', 'orwell'], ['the great gatsby', 'fitzgerald'],
                 ['fahrenheit 451', 'bradbury'], ['the stranger', 'camus'],
                 ['gone with the wind', 'mitchell'], ['pride and prejudice', 'austen'],
                 ['to kill a mockingbird', 'lee']]

    # print ratings for books of interest
    for i in range(np.shape(books)[0]):
        book = books[i, :]
        title = book[0].lower().replace('-',' ')
        author = book[1].lower()
        for entry in book_list:
            if entry[0] in title and entry[1] in author:
                print('Predicting rating %f for book %s' % (my_predict[i], books[i, 0]))

    # print top predicted ratings
    sorted = np.argsort(-1 * my_predict)
    for i in range(10):
        j = sorted[i]
        print('%d: Predicting rating %f for book %s by %s' % (i, my_predict[j], books[j, 0], books[j, 1]))


# def add_rating(user, book, rating):


if __name__ == "__main__":
    add_user('Data/BX-Books.csv', 'Data/learned_X_150.txt', 'Data/Y_mean.txt')
