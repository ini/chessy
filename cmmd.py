import numpy as np

from scipy.stats import cauchy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel



class RFF(BaseEstimator):
    """ 
    Random fourier features for Gaussian/Laplacian Kernels 
    Code adapted from https://github.com/hichamjanati/srf
    """

    def __init__(self, input_data_dim, rff_dim=50, gamma=1, metric='rbf'):
        self.gamma = gamma
        self.metric = metric
        self.rff_dim, self.input_data_dim = rff_dim, input_data_dim
        self.fit(input_data_dim)
        

    def fit(self, input_data_dim):
        """ Initializes random fourier features """
        if self.metric == 'rbf':
            self.w = np.sqrt(2 * self.gamma) * np.random.normal(size=(self.rff_dim, self.input_data_dim))
        elif self.metric == 'laplace':
            self.w = cauchy.rvs(scale=self.gamma, size=(self.rff_dim, self.input_data_dim))
        
        self.u = 2 * np.pi * np.random.rand(self.rff_dim) 
        self.fitted = True
        return self
    

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components) """
        if not self.fitted:
            raise NotFittedError('RFF must be fitted beform computing the feature map Z')

        # Compute feature map Z(x):
        Z = np.sqrt(2 / self.rff_dim) * np.cos((X.dot(self.w.T) + self.u[np.newaxis, :]))
        return Z
    

    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError('RFF must be fitted beform computing the kernel matrix')

        Z = self.transform(X)
        K = Z.dot(Z.T)
        return K



def conditional_operator(X, Y, X_rff=None, Y_rff=None, alpha=0.1):
    """ Computes the embedding operator for the conditional distribution Y|X """
    if X_rff is not None:
        X = X_rff.transform(X)
    if Y_rff is not None:
        Y = Y_rff.transform(Y)

    clf = Ridge(alpha=alpha)
    clf.fit(X, Y)
    return clf.coef_


def conditional_operator_slow(X, Y, X_rff, Y_rff, lamb=0.1):
    """ Computes the embedding operator for the conditional distribution Y|X """
    Phi = Y_rff.transform(Y).T
    Upsilon = X_rff.transform(X).T
    K_X = X_rff.compute_kernel(X)
    return Phi.dot(np.linalg.inv(K_X + lamb * np.eye(K_X.shape[0]))).dot(Upsilon.T)

