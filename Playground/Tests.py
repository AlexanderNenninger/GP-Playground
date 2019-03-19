import numpy as np
import math
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist as sccdist
from scipy.integrate import romb
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt

phi = lambda x, dx : romb(x**2, dx)

def cdist(XA, XB, metric='euclidean',*args, **kwargs):
    try:
        return sccdist(XA,XB, metric=metric, *args, **kwargs)
    except ValueError:
        return sccdist(np.matrix(XA).T, np.matrix(XB).T, metric=metric, *args, **kwargs)

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

if __name__=='__main__':
    for i in range(11):
        dim = 2**i + 1    
        x = np.linspace(0, 2*np.pi, dim)
        dx = x[1]-x[0]
        fx = np.sin(x)
        dists = cdist(x,x)/.5
        C = sqrtm(matern_cov(dists, 2.5))
        for i in range(1, 1000):
            g1x = C@np.random.standard_normal(fx.shape)
            g2x = C@np.random.standard_normal(fx.shape)
            log_prob = phi(g1x- fx, dx) - phi(g2x-fx, dx)
            try:
                samples = np.concatenate((samples, np.array([log_prob])))
            except NameError:
                samples = np.array([log_prob])
        print('Dim:{2} Mean:{0}, standard deviation: {1}\n'.format(
            np.mean(samples), np.std(samples), dim))
