import os
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

#
# MAIN
#

def main():

    #
    # SETTINGS
    #

    DEBUG = True

    seed = 42

    lambdas = np.arange(0., 10.0, 0.5)

    phis = [ lambda X: X,
             lambda X: np.power(2, X),
             lambda X: np.exp(X),
             lambda X: np.cos(X),
             lambda X: np.array([1]) ]

    #
    # SETUP
    #

    ## train & test paths
    data_dir = "./data/"

    ### csv w/ layout: "Id,y,x1,x2,x3,x4,x5"
    train_file = "train.csv"
    data_path = data_dir + train_file

    data_type = np.double
    cols_x = np.arange(2,6+1)
    cols_y = np.arange(1,1+1)

    ## prediction submission paths
    sub_dir = "../submission/"
    sub_file = "submission.csv"
    sub_path = sub_dir + sub_file

    ## submission csv header
    sub_header = ""
    sub_format = "%.13f"

    #
    # DATA
    #

    X = np.genfromtxt(data_path,
                      dtype = data_type,
                      delimiter = ",",
                      skip_header = 1,
                      usecols = cols_x)

    y = np.genfromtxt(data_path,
                      dtype = data_type,
                      delimiter = ",",
                      skip_header = 1,
                      usecols = cols_y)

    #
    # CROSS VALIDATION
    # 

    if DEBUG:
        print("")
        print(":: CROSS VALIDATION")
        print("")

    rmse = np.empty( (len(lambdas),) ) # dummy element for vstack to work, will be removed

    for phi in phis:

        rmse_phi = np.array([])

        if DEBUG:
            record_lam = None
            record_err = 10000.

        for lambdaa in lambdas:

            rmse_lambda = 0.

            X, y = shuffle(X, y, random_state = seed) # int(seed * lambdaa))

            kf = KFold(n_splits = 10,
                       shuffle = True,
                       random_state = seed)

            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                Z = phi(X_train)

                ZTy = np.dot(Z.T, y_train)
                ZTZ = Z.T @ Z

                ZTZ_inv = np.linalg.inv(ZTZ + lambdaa * np.eye(max(ZTZ.shape)))

                w = np.dot(ZTZ_inv, ZTy)

                y_pred = np.dot(phi(X_test), w)

                rmse_lambda += mean_squared_error(y_test, y_pred, squared = False)

            rmse_phi = np.hstack([rmse_phi, 0.1 * rmse_lambda])

            if DEBUG and 0.1 * rmse_lambda < record_err:
                record_err = 0.1 * rmse_lambda
                record_lam = lambdaa

        rmse = np.vstack([rmse, rmse_phi])

        if DEBUG:
            print(f"  lowest rmse = {record_err} with lambda = {record_lam}")

    rmse = rmse[1:,:] # remove initial dummy element
    index_of_best_lambda_per_phi = np.argmin(rmse, axis = 1)

    #
    # TRAINING
    # 

    if DEBUG:
        print("")
        print(":: TRAINING")
        print("")

    weights = np.array([])

    for idx, phi in enumerate(phis):

        lambdaa = lambdas[index_of_best_lambda_per_phi[idx]]
        Z = phi(X)

        ZTy = np.dot(Z.T, y)
        ZTZ = Z.T @ Z

        ZTZ_inv = np.linalg.inv(ZTZ + lambdaa * np.eye(max(ZTZ.shape)))

        w = np.dot(ZTZ_inv, ZTy)

        weights = np.hstack([weights, w])

        print(f"  using lambda = {lambdaa}")

    #
    # SERIALIZATION
    #

    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)

    np.savetxt(sub_path,
               weights,
               fmt = sub_format,
               delimiter = ',',
               header = sub_header,
               comments = '')

#
# MAIN
#

if __name__ == "__main__":
    main()
