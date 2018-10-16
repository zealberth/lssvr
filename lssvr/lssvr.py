"""Least Squares Support Vector Regression."""
import numpy as np
import sklearn

from sklearn.base import BaseEstimator, RegressorMixin


class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, supportVectors=None,supportVectorLabels=None):
        self.supportVectors      = supportVectors
        self.supportVectorLabels = supportVectorLabels

    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None):
        if idxs == None:
            self.supportVectors      = x_train
            self.supportVectorLabels = y_train
        else
            self.supportVectors      = x_train[idxs,:]
            self.supportVectorLabels = y_train[idxs,:]

        K = self.kernel_func(kernel, x_train, x_train, sigma)

        idx = np.diag_indices_from(K)
        OMEGA = K
        OMEGA[idx] += 1/gamma

        size = (OMEGA.shape[0]+1, OMEGA.shape[1]+1)

        D = np.zeros(size)

        D[1:,1:] = OMEGA
        D[0, 1:] = np.ones(y_train.shape[0])
        D[1:,0 ] = np.ones(y_train.shape[0])

        t = np.zeros((y_train.shape[0]+1, ))
        t[1:] = y_train

        z = np.linalg.lstsq(D, t, rcond=-1)

        self.bias   = z[0][0]
        self.alphas = z[0][1:]

        return self

    def predict(self, x_test, kernel='linear', sigma=0.05):
        K = self.kernel_func(kernel, x_test, self.supportVectors)

        return np.sum(K * (np.tile(self.alphas, (K.shape[0], 1))), axis=1) + self.bias

    def kernel_func(self, kernel, u, v, sigma = 0.05):
        if kernel == 'linear':
            k = np.dot(u, v.T)
        if kernel == 'rbf':
            k = sklearn.metrics.pairwise.rbf_kernel(u, v, gamma=1/(sigma**2))
        return k