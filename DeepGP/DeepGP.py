from abc import ABCMeta, abstractmethod
import math
import numpy as np

from scipy.spatial.distance import pdist, cdist as sccdist, squareform
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
from scipy.integrate import romb, simps
from scipy.linalg import sqrtm

np.random.seed(0)
#helper

def cdist(XA, XB, metric='euclidean',*args, **kwargs):
    try:
        return sccdist(XA,XB, metric=metric, *args, **kwargs)
    except ValueError:
        return sccdist(np.matrix(XA).T, np.matrix(XB).T, metric=metric, *args, **kwargs)


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


def w_pCN(y, T, n_iter, beta, phi, dx, xi = None, burnin_ratio = 2, debug=False):
    #initialize xi and u
    if not xi: xi = np.random.standard_normal(y.shape)
    u = T@xi
    samples = []
    accepted = []
    #debug code
    if debug:
        av_acc = 0
        log_probs = []
    #the loop
    for i in range(n_iter):
        acc = False
        #propose update
        xi_hat =  np.sqrt(1-beta**2) * xi + beta * np.random.standard_normal(xi.shape)
        u_hat = T@xi_hat
        #evaluate update
        log_prob = phi(u-y, dx)-phi(u_hat-y, dx)
        if np.random.rand() <= np.exp(log_prob):
            xi = xi_hat
            u = u_hat
            acc = True
        #store samples
        if i > n_iter/burnin_ratio:
            samples.append(u)
            accepted.append(acc)

        #debug code
        if debug:
            av_acc = av_acc +(acc-av_acc)/(i+1) # discounted acceptance rate
            log_probs.append(log_prob)
    #debug code
    if debug:
        return samples, acc, av_acc, xi, log_probs
   
    return samples, acc


#distance metric
phi = lambda x, dx: romb(np.abs(x)**2, dx) * 10


if __name__=='__main__':
    fig = plt.figure()
    fig.tight_layout()
    for i in range(0,10):
        x = np.linspace(0,1, 2**i+1)
        fx =np.round(np.sin(2*np.pi*x)) + 0.05 * np.random.standard_normal(x.shape)
        C = sqrtm(
            matern_cov(
                cdist(x,x)/.2, 
                2.5
                ))
        samples , acc, acc_prob, _, log_probs = w_pCN(fx, C, 10000, .5, phi, x[1]-x[0], burnin_ratio=2, debug=True)
        mean = np.mean(samples, axis=0)
        
        ax = fig.add_subplot(4,4,i+1)
        ax.plot(x,fx,'b--')
        ax.set_title(
            '''Dim: %s; #Samples: %s; Acc Prob: %s'''%(2**i+1, len(samples), round(acc_prob,4), )
        )
        #ax.plot(x, samples, 'x')        
        ax.plot(x, mean, 'r')
        print(ax.title)
    plt.show()
