import numpy as np
import pandas as pd


class Data:
    def load_data(self, filename = 'Data/ratings.txt'):
        # load data and define constants
        array = np.loadtxt(filename, dtype=int, delimiter=',', skiprows=1)
        (num_users, num_books, max_rating) = np.max(array, axis=0)

        # initialize empty matrices (book-id x user-id)
        self.all_Y = np.empty(shape=(num_books, num_users))
        self.all_R = np.zeros(shape=(num_books, num_users), dtype=bool)
        self.train_Y = np.empty(shape=(num_books, num_users))
        self.train_R = np.zeros(shape=(num_books, num_users), dtype=bool)
        self.cv_Y = np.empty(shape=(num_books, num_users))
        self.cv_R = np.zeros(shape=(num_books, num_users), dtype=bool)
        self.test_Y = np.empty(shape=(num_books, num_users))
        self.test_R = np.zeros(shape=(num_books, num_users), dtype=bool)

        # define thresholds
        np.random.shuffle(array)
        self.num_all = np.shape(array)[0]
        self.train_thres = round(self.num_all * 0.6)
        self.cv_thres = round(self.num_all * 0.8)

        # insert data into respective matrices (training vs. cv vs. test)
        i = 0
        for entry in array:
            book_id = entry[1]
            user_id = entry[0]
            rating = entry[2]

            self.all_Y[book_id - 1, user_id - 1] = rating
            self.all_R[book_id - 1, user_id - 1] = True

            if i < self.train_thres:
                self.train_Y[book_id - 1, user_id - 1] = rating
                self.train_R[book_id - 1, user_id - 1] = True
            elif i < self.cv_thres:
                self.cv_Y[book_id - 1, user_id - 1] = rating
                self.cv_R[book_id - 1, user_id - 1] = True
            else:
                self.test_Y[book_id - 1, user_id - 1] = rating
                self.test_R[book_id - 1, user_id - 1] = True

            i += 1


if __name__ == "__main__":
    data = Data()
    data.load_data()
