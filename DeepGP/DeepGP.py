from abc import ABCMeta, abstractmethod
import math
import numpy as np

from scipy.spatial.distance import pdist, cdist as sccdist, squareform
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import kv, gamma

#helper

def cdist(XA, XB, metric='euclidean',*args, **kwargs):
    try:
        return sccdist(XA,XB, metric=metric, *args, **kwargs)
    except ValueError:
        return sccdist(np.matrix(XA).T, np.matrix(XB).T, metric=metric, *args, **kwargs)


length_scale = 1
##kernels
def matern_cov(dists, nu):
    if nu == 0.5:
        K = np.exp(-dists)
    elif nu == 1.5:
        K = dists * math.sqrt(3)
        K = (1. + K) * np.exp(-K)
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
    else:  # general case; expensive to evaluate
        K = dists
        K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
        tmp = (math.sqrt(2 * nu) * K)
        K.fill((2 ** (1. - nu)) / gamma(nu))
        K *= tmp ** nu
        K *= kv(nu, tmp)
    return K


def w_pCN(y, T, n_iter, beta, phi, xi = None, burnin_ratio = 5):
    accepted = []
    if not xi:
        xi = np.random.standard_normal(y.shape)
    for n in range(n_iter):
        acc = False
        xi_hat = np.sqrt(1-beta**2) * xi + beta * np.random.standard_normal(xi.shape)
        u = T@xi
        u_hat = T@xi_hat
        log_prob = min(0, phi(u, y) - phi(u_hat,y)) # min to combat overflow
        if  np.exp(log_prob) >= np.random.rand():
            xi = xi_hat
            u = u_hat
            acc = True
        if n>n_iter/burnin_ratio:
            try:
                samples = np.vstack((samples, u))
            except NameError:
                samples = u
            accepted.append(acc)
    return samples, accepted, xi


#distance metric
phi = lambda x, y: np.sum((x - y) ** 2)

def test_iter(num_basis_functions: list):
    acceptance_probability = []
    for i in num_basis_functions:    
        #generate data
        sigma = .0 #std of noise
        n_training_samples = i

        X = np.linspace(0,1,n_training_samples)*2*np.pi
        y = np.round(np.sin(X)) + sigma * np.random.standard_normal(X.shape)


        # X_eval = np.random.rand(n_eval_samples, X_train.shape[1])

        # X = np.vstack((X_train, X_eval))
        # y = np.vstack((y_train, np.random.standard_normal()))
        C = cdist(X, X)
        C = matern_cov(C, 1.5)
        #standard deviation
        T = np.linalg.cholesky(C).T.conj()
        #Run w-pCN
        n_iter = 1000
        xi = np.random.standard_normal(X.shape)
        beta = .2
        #iteration
        path, accepted, xi  = w_pCN(y, T, n_iter, beta, phi)
       
        acc_prob = sum(accepted)/len(accepted)
        for sample in path:
            plt.plot(X, sample, 'rx')
        plt.plot(X, np.mean(path, axis=0), 'b')
        plt.plot(X,y, 'g-')
        plt.text(0,0,acc_prob)
        acceptance_probability.append(acc_prob)
        plt.savefig('regression'+str(i))

    plt.plot(num_basis_functions,acceptance_probability)
    plt.show()


if __name__=='__main__':
    test_iter([1,5,10,100,500,1000])

        # fig, ax = plt.subplots()
        # fig.text(10,10, ''.join(['Accaptance Probability', str(acc_prob)]))
        # ax.plot(X,y)
        # ax.plot(X,np.mean(path, axis=0))
        # plt.show()