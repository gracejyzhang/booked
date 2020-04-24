import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
from datetime import datetime as datetime
from scipy.optimize import minimize
from Original import Calculations as calc

# pages: explore, search, read, saved

cred = credentials.Certificate('booked-out-844dc0c72c42.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://booked-out.firebaseio.com'
})
parameters_ref = db.reference('parameters')
users_ref = db.reference('users')
books_ref = db.reference('books')


def add_parameter(X_filename = 'learned_X_150.txt', Y_mean_filename = 'Y_mean.txt'):
    X = np.loadtxt(X_filename, delimiter=' ', skiprows=0)
    Y_mean = np.loadtxt(Y_mean_filename)
    parameters_ref.set({datetime.now().strftime("%Y-%m-%d %H:%M:%S"): {'matrix': X.tolist(), 'y_mean': Y_mean.tolist()}})


def get_parameter():
    # todo: check whether it should be limit to first or last
    key, value = parameters_ref.order_by_key().limit_to_last(1).get().popitem()
    return np.array(value['matrix']), np.array(value['y_mean'])


def delete_parameters():
    parameters_ref.delete()


# note: ratings of 0
# todo: handle dangers of calling twice (currently erases everything)
def add_user(uid):
    try:
        # todo: check whether it should be limit to first or last
        key, value = users_ref.order_by_child('user_id').limit_to_last(1).get().popitem()
        id = value['user_id'] + 1
    except AttributeError:
        id = 0
    X, Y_mean = get_parameter()
    ratings = np.zeros(shape=len(Y_mean))
    users_ref.set({uid: {'user_id': id, 'ratings': ratings.tolist(), 'predictions': Y_mean.tolist(), 'saved': [], 'read': []}})


def get_user_data(uid):
    user = users_ref.child(uid)
    ratings = user.child('ratings')
    predictions = user.child('predictions')
    saved = user.child('saved')
    read = user.child('read')
    return np.array(ratings), np.array(predictions), saved, read


# note: list = 'read' or 'saved'
def add_books(uid, bookids, list):
    user = users_ref.child(uid)
    arr = user.child(list)
    for book in bookids:
        arr.append(book)
    user.update({list: arr})
    return arr


# note: assumes rating already added to user (just need to predict now)
# todo: when storing books in db, make book id start from 0
def add_ratings(user, book_rating_dict):
    X, Y_mean = get_parameter()
    num_features = 32
    lam = 2
    Y = get_user_data(user)[0]
    Theta = np.random.rand(num_features)
    for book, rating in book_rating_dict:
        Y[book] = rating
    Y_norm = Y - Y_mean
    R = np.array(Y, dtype=bool)

    args = (X, Y_norm, R, lam)
    result = minimize(fun=calc.user_gradient, x0=Theta, args=args, method='CG', jac=True, options={'disp':True})
    Theta = result.x
    predict = X @ Theta.transpose() + Y_mean

    # save data in db
    user_ref = users_ref.child(user)
    read_list = add_books(user, list(book_rating_dict.keys()), 'read') # can omit
    user_ref.update({'ratings': Y.tolist(), 'predictions': predict.tolist()})
    return Y, predict, read_list


def add_book(dict):
    books_ref.set(dict)
