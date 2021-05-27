import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import kernels
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.sparse import linalg


class LSSVR(BaseEstimator, RegressorMixin):
    """Least Squares Support Vector Regression.
    
    Parameters
    ----------
    C : float, default=2.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    kernel : {'linear', 'rbf'}, default='linear'
        Specifies the kernel type to be used in the algorithm.
        It must be 'linear', 'rbf' or a callable.

    gamma : float, default = None
        Kernel coefficient for 'rbf'


    Attributes
    ----------
    support_: boolean np.array of shape (n_samples,), default = None
        Array for support vector selection.
    
    alpha_ : array-like
        Weight matrix

    bias_ : array-like
        Bias vector


    """
        self.C = C
        self.gamma = gamma
        self.kernel= kernel
        self.idxs  = None
        self.K = None
        self.bias = None 
        self.alphas = None

        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        support : boolean np.array of shape (n_samples,), default = None
            Array for support vector selection.

        Returns
        -------
        self : object
            An instance of the estimator.
        """

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

        """
        Predict using the estimator.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
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
    
