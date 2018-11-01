"""Least Squares Support Vector Regression."""
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from mdlp import MDLP

from sklearn.base import BaseEstimator, RegressorMixin


class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, supportVectors=None,supportVectorLabels=None):
        self.supportVectors      = supportVectors
        self.supportVectorLabels = supportVectorLabels

    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None):
        # print(idxs.shape, idxs)
        if type(idxs) == type(None):
            idxs=np.ones(x_train.shape[0], dtype=bool)

        self.supportVectors      = x_train[idxs, :]
        self.supportVectorLabels = y_train[idxs]

        K = self.kernel_func(kernel, x_train, self.supportVectors, sigma)

        OMEGA = K
        OMEGA[idxs, np.arange(OMEGA.shape[1])] += 1/gamma

        D = np.zeros(np.array(OMEGA.shape) + 1)

        D[1:,1:] = OMEGA
        D[0, 1:] = np.ones(OMEGA.shape[1])
        D[1:,0 ] = np.ones(OMEGA.shape[0])

        n = len(self.supportVectorLabels) + 1
        t = np.zeros(n)
        
        t[1:n] = self.supportVectorLabels

        z = np.linalg.pinv(D).T @ t.ravel()

        self.bias   = z[0]
        self.alphas = z[1:]
        self.alphas = self.alphas[idxs]
        self.sigma = sigma
        self.kernel = kernel

        return self

    def predict(self, x_test):
        K = self.kernel_func(self.kernel, x_test, self.supportVectors)

        return np.sum(K * (np.tile(self.alphas, (K.shape[0], 1))), axis=1) + self.bias

    def kernel_func(self, kernel, u, v, sigma = 0.05):
        if kernel == 'linear':
            k = np.dot(u, v.T)
        if kernel == 'rbf':
            k = sklearn.metrics.pairwise.rbf_kernel(u, v, gamma=sigma)
        return k

class RegENN_LSSVR(LSSVR):
    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None, alpha=2, n_neighbors=9):
        idxs = self.RegENN(x_train, y_train, alpha, n_neighbors)

        super().fit(x_train, y_train, gamma=gamma, kernel=kernel, sigma=sigma, idxs=idxs)

    def RegENN(self,X, y, alpha, n_neighbors):
        n = len(X)
        T = np.ones(n, dtype=bool)

        for i in range(n):
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X[T]) 
            _, idxs = nbrs.kneighbors(np.asmatrix(X[i])) #idxs para montar o conjunto S

            neigh = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X[T], y[T]) 
            y_hat = neigh.predict(np.asmatrix(X[i]))

            theta = alpha * np.std(y[idxs[0,1:]])

            err = np.abs(y[i] - y_hat)
            
            if err > theta:
                T[i] = False
        return T    
    
class RegCNN_LSSVR(LSSVR):
    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None, alpha=2, n_neighbors=9):
        idxs = self.RegCNN(x_train, y_train, alpha, n_neighbors)

        super().fit(x_train, y_train, gamma=gamma, kernel=kernel, sigma=sigma, idxs=idxs)

    def RegCNN(self, X, y, alpha, n_neighbors):
        n = len(X)

        T = np.ones(n, dtype=bool)
        P = np.zeros(n, dtype=bool)

        P[0] = True
        T[0] = False

        for i in range(n):
            # tratar o lance do n_neighbors ser menor que o conjunto
            if len(X[P]) < n_neighbors:
                nbrs = NearestNeighbors(n_neighbors=len(X[P]), algorithm='ball_tree').fit(X[P])
                _, idxs = nbrs.kneighbors(np.asmatrix(X[i])) #idxs para montar o conjunto S
            else:
                nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X[P])
                _, idxs = nbrs.kneighbors(np.asmatrix(X[i])) #idxs para montar o conjunto S


            T[i] = False
            neigh = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X[T], y[T]) 
            y_hat = neigh.predict(np.asmatrix(X[i]))
            T[i] = True

            theta = alpha * np.std(y[idxs.ravel()])
            
            err = np.abs(y[i] - y_hat)

            if err > theta:
                P[i] = True
                T[i] = False
        return P
    
class DiscENN_LSSVR(LSSVR):
    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None, n_neighbors=9):
        idxs = self.DiscENN(x_train, y_train, n_neighbors)

        super().fit(x_train, y_train, gamma=gamma, kernel=kernel, sigma=sigma, idxs=idxs)

    def WilsonENN(self, X, y, n_neighbors):
        n = len(X)
        S = np.ones(n, dtype=bool)

        for i in range(n):
            P = np.ones(n, dtype=bool)
            P[i] = False
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X[P])
            _, idxs = nbrs.kneighbors(np.asmatrix(X[i])) 
            if stats.mode(y[idxs][0])[0] != y[i]:
                S[i] = False
        return S

    def DiscENN(self, X, y, n_neighbors):
        mdlp = MDLP()
        conv_y = mdlp.fit_transform(y.reshape(-1,1), y)
        return self.WilsonENN(X, conv_y, n_neighbors=n_neighbors)

class MI_LSSVR(LSSVR):
    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None, alpha=0.05, n_neighbors=6):
        idxs = self.MutualInformationSelection(x_train, y_train, alpha=alpha, n_neighbors=n_neighbors)

        super().fit(x_train, y_train, gamma=gamma, kernel=kernel, sigma=sigma, idxs=idxs)

    def MutualInformationSelection(self, X, y, alpha, n_neighbors):
        n = len(X)
        mask = np.arange(n)

        mi = [mutual_info_regression(X[mask != i], y[mask != i]) for i in range(n)]

        mi = MinMaxScaler().fit_transform(mi)

        _, neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X).kneighbors(X)

        # dropout themselves
        neighbors = neighbors[:, 1:]

        cdiff = [np.sum((mi[i] - mi[neighbors[i]]) > alpha) for i in range(n)]

        idx = np.array(cdiff) < n_neighbors

        return idx

class AM_LSSVR(LSSVR):
    def fit(self, x_train, y_train, gamma=16, kernel='linear', sigma=0.05, idxs=None, cutoff=(.2, .32), k=None):
        idxs = self.KSSelection(x_train, y_train, cutoff=cutoff, k=k)
        
        super().fit(x_train, y_train, gamma=gamma, kernel=kernel, sigma=sigma, idxs=idxs)

    def KSSelection(self, X, y, cutoff, k):
        n = len(X)

        if k is None:
            k = round(5 * np.log10(n))

        knn = NearestNeighbors(n_neighbors=int(k + 1), algorithm='ball_tree').fit(X)

        distx, ind = knn.kneighbors(X)

        knn = NearestNeighbors(n_neighbors=int(k + 1), algorithm='ball_tree').fit(y.reshape(-1,1))

        disty, ind = knn.kneighbors(y.reshape(-1,1), return_distance=True)

        zipped = list(zip(distx[:, :1], disty[:, :1]))

        p = [self.__pval_ks_2smap(entry) for entry in zipped]

        order = np.argsort(p)

        h_cutoff = round(cutoff[0] * n)
        l_cutoff = round(cutoff[1] * n)

        idx = np.zeros((n, 1), dtype=bool)
        idx[order[:l_cutoff]] = True
        idx[order[-h_cutoff:]] = True

        idx = idx.ravel()

        return idx

    @staticmethod
    def __pval_ks_2smap(entry):
        from scipy.stats import ks_2samp, zscore

        a = zscore(entry[0]) if np.std(entry[0]) > 0 else entry[0]

        b = zscore(entry[1]) if np.std(entry[1]) > 0 else entry[1]

        _, pval = ks_2samp(a, b)

        return pval