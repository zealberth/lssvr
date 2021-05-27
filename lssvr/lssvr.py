"""Least Squares Support Vector Regression."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import kernels
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.sparse import linalg


class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=2, kernel='linear', gamma=None):
        self.supportVectors      = None
        self.supportVectorLabels = None
        self.C = C
        self.gamma = gamma
        self.kernel= kernel
        self.idxs  = None
        self.K = None
        self.bias = None 
        self.alphas = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x_train, y_train):
        # self.idxs can be used to select points as support vectors,
        # so you need another algorithm or criteria to choose them
        if type(self.idxs) == type(None):
            self.idxs=np.ones(x_train.shape[0], dtype=bool)

        self.supportVectors      = x_train[self.idxs, :]
        self.supportVectorLabels = y_train[self.idxs]

        K = self.kernel_func(self.kernel, x_train, self.supportVectors, self.gamma)

        self.K = K
        OMEGA = K
        OMEGA[self.idxs, np.arange(OMEGA.shape[1])] += 1/self.C

        D = np.zeros(np.array(OMEGA.shape) + 1)

        D[1:,1:] = OMEGA
        D[0, 1:] += 1
        D[1:,0 ] += 1

        shape = np.array(self.supportVectorLabels.shape)
        shape[0] +=1

        t = np.zeros(shape)
        
        t[1:] = self.supportVectorLabels
    
        # sometimes this function breaks
        try:
            z = linalg.lsmr(D.T, t)[0]
        except:
            z = np.linalg.pinv(D).T @ t

        self.bias   = z[0]
        self.alphas = z[1:]
        self.alphas = self.alphas[self.idxs]

        return self

    def kernel_func(self, u, v):
        if self.kernel is 'linear':
            return np.dot(u, v.T)

        elif self.kernel is 'rbf':
            return rbf_kernel(u, v, gamma=self.gamma)

        elif callable(self.kernel):
            if hasattr(self.kernel, 'gamma'):
                return self.kernel(u, v, gamma=self.gamma)
            else:
                return self.kernel(u, v)
        else:
          # default to linear
          return np.dot(u, v.T)

    def score(self, X, y):
        from scipy.stats import pearsonr
        p, _ = pearsonr(y, self.predict(X))
        return p ** 2

    def norm_weights(self):
        A = self.alpha_.reshape(-1,1) @ self.alpha_.reshape(-1,1).T

        W = A @ self.K[self.idxs,:]
        return np.sqrt(np.sum(np.diag(W)))
    
