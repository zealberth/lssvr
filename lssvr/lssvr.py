"""Least Squares Support Vector Regression."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import kernels
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from scipy import stats, optimize
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.sparse import linalg
from mdlp import MDLP


class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=None, kernel=None, gamma=None):
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

        n = len(self.supportVectorLabels) + 1
        t = np.zeros(n)
        
        t[1:n] = self.supportVectorLabels

        try:
            z = linalg.lsmr(D.T, t)[0]
        except:
            z = np.linalg.pinv(D).T @ t.ravel()

        self.bias   = z[0]
        self.alphas = z[1:]
        self.alphas = self.alphas[self.idxs]

        return self

    def predict(self, x_test):
        K = self.kernel_func(self.kernel, x_test, self.supportVectors, self.gamma)

        return (K @ self.alphas) + self.bias
        # return np.sum(K * (np.tile(self.alphas, (K.shape[0], 1))), axis=1) + self.bias

    def kernel_func(self, kernel, u, v, gamma):
        if kernel == 'linear':
            k = np.dot(u, v.T)
        if kernel == 'rbf':
            k = rbf_kernel(u, v, gamma=gamma)
            # temp = kernels.RBF(length_scale=(1/gamma))
            # k = temp(u, v)
        return k

    def score(self, X, y, sample_weight=None):
        from scipy.stats import pearsonr
        p, _ = pearsonr(y, self.predict(X))
        return p ** 2
        #return RegressorMixin.score(self, X, y, sample_weight)

    def norm_weights(self):
        n = len(self.supportVectors)

        A = self.alphas.reshape(-1,1) @ self.alphas.reshape(-1,1).T
        # import pdb; pdb.set_trace()
        W = A @ self.K[self.idxs,:]
        return np.sqrt(np.sum(np.diag(W)))

class RegENN_LSSVR(LSSVR):
    def __init__(self, C=None, kernel=None, gamma=None, alpha=None, n_neighbors=None):
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        LSSVR.__init__(self, C, kernel, gamma)

    def fit(self, x_train, y_train):
        if self.n_neighbors is None or not np.isscalar(self.n_neighbors):
            self.n_neighbors = round(np.log10(len(x_train)) * 5).astype('int')

        self.idxs = self.RegENN(x_train, y_train, self.alpha, self.n_neighbors)

        super(RegENN_LSSVR, self).fit(x_train, y_train)
        return self

    def RegENN(self,X, y, alpha, n_neighbors):
        n = len(X)
        T = np.ones(n, dtype=bool)

        for i in range(n):
            if len(X[T]) < n_neighbors:
                nbrs = NearestNeighbors(n_neighbors=len(X[T]), algorithm='ball_tree').fit(X[T])
                _, idxs = nbrs.kneighbors(np.asmatrix(X[i])) #idxs para montar o conjunto S
                neigh = KNeighborsRegressor(n_neighbors=len(X[T])).fit(X[T], y[T]) 
                y_hat = neigh.predict(np.asmatrix(X[i]))
            else:
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
    def __init__(self, C=None, kernel=None, gamma=None, alpha=None, n_neighbors=None):
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        LSSVR.__init__(self, C, kernel, gamma)

    def fit(self, x_train, y_train):
        if self.n_neighbors is None or not np.isscalar(self.n_neighbors):
            self.n_neighbors = round(np.log10(len(x_train)) * 5).astype('int')

        self.idxs = self.RegCNN(x_train, y_train, self.alpha, self.n_neighbors)

        super(RegCNN_LSSVR, self).fit(x_train, y_train)

        return self

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


            # tratando erro de caso o fold ser muito pequeno
            if len(X[T]) < n_neighbors:
                T[i] = False
                neigh = KNeighborsRegressor(n_neighbors=int(round(len(X[T])*.5))).fit(X[T], y[T]) 
                y_hat = neigh.predict(np.asmatrix(X[i]))
                T[i] = True
            else:
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
    def __init__(self, C=None, kernel=None, gamma=None, n_neighbors=None):
        self.n_neighbors = n_neighbors
        LSSVR.__init__(self, C, kernel, gamma)

    def fit(self, x_train, y_train):
        if self.n_neighbors is None or not np.isscalar(self.n_neighbors):
            self.n_neighbors = round(np.log10(len(x_train)) * 5).astype('int')

        # print (self.n_neighbors)
        self.idxs = self.DiscENN(x_train, y_train, self.n_neighbors)

        super(DiscENN_LSSVR, self).fit(x_train, y_train)

        return self

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
        if np.min(y)<0:
            y = y - np.min(y)
        conv_y = mdlp.fit_transform(y.reshape(-1,1), y)

        return self.WilsonENN(X, conv_y, n_neighbors=n_neighbors)

class MI_LSSVR(LSSVR):
    def __init__(self, C=None, kernel=None, gamma=None, alpha=None, n_neighbors=None):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        LSSVR.__init__(self, C, kernel, gamma)

    def fit(self, x_train, y_train):
        if self.n_neighbors is None or not np.isscalar(self.n_neighbors):
            self.n_neighbors = round(np.log10(len(x_train)) * 5).astype('int')

        self.idxs = self.MutualInformationSelection(x_train, y_train, alpha=self.alpha, n_neighbors=self.n_neighbors)

        super(MI_LSSVR, self).fit(x_train, y_train)

        return self

    def MutualInformationSelection(self, X, y, alpha, n_neighbors):
        n = len(X)
        mask = np.arange(n)

        mi = [mutual_info_regression(X[mask != i], y[mask != i]) for i in range(n)]

        mi = MinMaxScaler().fit_transform(mi)

        _, neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X).kneighbors(X)

        # dropout themselves
        neighbors = neighbors[:, 1:]

        mask_as_set = set(mask)

        not_neighbors = [list(mask_as_set - set(neighbors[i])) for i in range(n)]

        nn_mi = [mutual_info_regression(X[not_neighbors[i]], y[not_neighbors[i]]) for i in range(n)]

        selected = np.ones(len(X), dtype=bool)

        for i in range(n):
            cdiff = np.asarray([(mi[i] - mi[k]) for k in neighbors[i]])
            selected[i] = np.sum(cdiff > alpha) > n_neighbors

        return selected

class AM_LSSVR(LSSVR):
    def __init__(self, C=None, kernel=None, gamma=None, cut_high=None, cut_low = None, n_neighbors=None):
        self.n_neighbors = n_neighbors
        self.cut_high = cut_high
        self.cut_low = cut_low
        LSSVR.__init__(self, C, kernel, gamma)

    def fit(self, x_train, y_train):
        if self.n_neighbors is None or not np.isscalar(self.n_neighbors):
            self.n_neighbors = round(np.log10(len(x_train)) * 5).astype('int')

        self.idxs = self.KSSelection(x_train, y_train, cut_high=self.cut_high, cut_low=self.cut_low, n_neighbors=self.n_neighbors)
        
        super(AM_LSSVR, self).fit(x_train, y_train)

        return self

    def KSSelection(self, X, y, cut_high, cut_low, n_neighbors):
        n = len(X)

        knn = NearestNeighbors(n_neighbors=int(n_neighbors + 1), algorithm='ball_tree').fit(X)

        distx, ind = knn.kneighbors(X)

        knn = NearestNeighbors(n_neighbors=int(n_neighbors + 1), algorithm='ball_tree').fit(y.reshape(-1,1))

        disty, ind = knn.kneighbors(y.reshape(-1,1), return_distance=True)

        zipped = list(zip(distx[:, :1], disty[:, :1]))

        p = [self.__pval_ks_2smap(entry) for entry in zipped]

        order = np.argsort(p)

        h_cutoff = int(round(cut_high * n))
        l_cutoff = int(round(cut_low * n))

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

class NL_LSSVR(LSSVR):
    def __init__(self, C=None, kernel=None, gamma=None):
        LSSVR.__init__(self, C, kernel, gamma)

    def fit(self, x_train, y_train):
        self.idxs = self.NLSelection(x_train, y_train)
        
        super(NL_LSSVR, self).fit(x_train, y_train)

        return self

    def NLSelection(self, X, y):
        n = len(X)

        # trick to sort rows
        order = self.__class__.__order_of(X)

        yl = y[order].ravel()

        s = np.round(np.sqrt(np.std(yl)))
        if s < 1:
            s = 1
        h_peaks, l = find_peaks(yl, distance=s)
        l_peaks, l = find_peaks(-yl, distance=s)

        idx = np.zeros(n, dtype=bool)

        idx[h_peaks] = True
        idx[l_peaks] = True

        return idx

    @staticmethod
    def __order_of(X):

        x_origin = np.min(X, axis=0)
        keys = cdist(np.asmatrix(x_origin), X)

        return np.argsort(keys)
