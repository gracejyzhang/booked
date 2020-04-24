import numpy as np
import pandas as pd
import scipy.sparse as sp

# load filename as a sparse matrix
def load_sparse_matrices(filename, shape_filename = 'Data/processed_input.txt'):
    # determine the shape of the sparse matrix using the complete set of input data
    total_matrix = np.loadtxt(shape_filename, dtype=int)
    total_shape = sp.csr_matrix((total_matrix[:, 2], (total_matrix[:, 1], total_matrix[:, 0]))).shape

    # load filename and generate sparse matrices Y and R
    matrix = np.loadtxt(filename, dtype=int)
    Y = sp.csr_matrix((matrix[:, 2], (matrix[:, 1], matrix[:, 0])), shape=total_shape)
    R = sp.csr_matrix(Y, shape=total_shape, dtype=bool)

    return Y, R


# split the input file data into two output files
def split_matrices(input_file, output_file_large, output_file_small, split):
    # load and process input file
    matrix = np.loadtxt(input_file, dtype=int)
    np.random.shuffle(matrix)
    rows = np.shape(matrix)[0]

    # split the data into two sets
    threshold = round(rows * split)
    large = matrix[:threshold, :]
    small = matrix[threshold:, :]

    # save split data into output files
    np.savetxt(output_file_large, large)
    np.savetxt(output_file_small, small)


class LargeData:
    # get the book id corresponding to the given isbn
    def get_book(self, isbn):
        try:
            x = np.nonzero(self.books == isbn)[0][0]
        except:
            x = -1
        return x


    def load_data(self, book_filename = 'Data/BX-Books.csv', book_rating_filename = 'Data/BX-Book-Ratings.csv', output_filename = 'Data/processed_input.txt'):
        # define converter functions
        get_rating = lambda x : (int(x))
        get_user_id = lambda x : (int(x) - 1)
        get_book_id = lambda x : self.get_book(x)

        # load book matrix and book rating matrix
        self.books = pd.read_csv(book_filename, delimiter=';', usecols=[0]).to_numpy().flatten()
        matrix = pd.read_csv(book_rating_filename, sep=';', converters={'User-ID':get_user_id, 'ISBN':get_book_id, 'Book-Rating':get_rating}, quotechar='"', encoding='latin-1').to_numpy()

        # save processed matrix where book is mapped to id instead of isbn
        np.savetxt(output_filename, matrix)


if __name__ == "__main__":
    split_matrices('training_cv_test.txt', 'cv_input.txt', 'test_input.txt', 0.5)
