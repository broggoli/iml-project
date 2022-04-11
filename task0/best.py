import os
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

#
# MAIN
#

def main():

    #
    # SETTINGS
    #

    # True:  use ridge regression by package 'sklearn'
    # False: perform manual ridge regression by normal equations 
    USE_SKLEARN = False

    seed = 42 # what else?

    lambdas = [0.1, 1., 10., 100., 200.]

    # shuffling in inter-split (i.e. not intra-split) fashion
    shuffle_splits = True
    n_splits = 10

    #
    # SETUP
    #

    ## train & test paths
    data_dir = "./data/"

    ### csv w/ layout: "y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13"
    train_file = "train.csv"
    data_path = data_dir + train_file

    ## prediction submission paths
    sub_dir = "./submission/"
    sub_file = "submission.csv"
    sub_path = sub_dir + sub_file

    ## submission csv header
    sub_header = ""
    sub_format = "%.13f"

    #
    # DATA
    #

    ## data type in csv
    data_type = np.double # np dtype object for values in X and Y

    X = np.genfromtxt(data_path,
                      dtype = data_type,
                      delimiter = ",",
                      skip_header = 1,
                      usecols = np.arange(1, 13 + 1))

    y = np.genfromtxt(data_path,
                      dtype = data_type,
                      delimiter = ",",
                      skip_header = 1,
                      usecols = np.arange(0, 0 + 1))

    rmse = []

    print(f"Ride Regression with {n_splits}-fold Cross Validation")

    for alpha in lambdas: # lambda is already taken ._.

        #
        # MODEL
        #

        rmse_alpha = 0

        print(f"  using alpha = {alpha}")
        model = Ridge(alpha = alpha)

        kf = KFold(n_splits = n_splits,
                   shuffle = shuffle_splits,
                   random_state = seed)

        #
        # TRAIN & EVAL
        #

        X, y = shuffle(X, y, random_state = int(seed * alpha))

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            print(X_train, X_test)
            if USE_SKLEARN:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            else:
                XT = X_train.transpose()
                XTy = np.dot(XT, y_train)
                XTX = XT @ X_train
                XTX_LI_inv = np.linalg.inv(XTX + alpha * np.eye(max(XTX.shape)))
                w_hat = np.dot(XTX_LI_inv, XTy)
                y_pred = np.dot(X_test, w_hat)

            rmse_alpha += mean_squared_error(y_test, y_pred, squared = False)

        rmse.append((1 / n_splits) * rmse_alpha)

        print(f"  average RMSE: {rmse[len(rmse)-1]}")
        print(f"")

    #
    # SERIALIZATION
    #

    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)

    submission = np.array(rmse)

    np.savetxt(sub_path,
               submission,
               fmt = sub_format,
               delimiter = ',',
               header = sub_header,
               comments = '')

#
# MAIN
#

if __name__ == "__main__":
    main()
