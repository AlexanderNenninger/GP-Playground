import os
import sys
sys.path.append(os.getcwd())


import math
import numpy as np
from functools import partial

from scipy.spatial.distance import pdist, cdist as sccdist, squareform
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
from scipy.integrate import romb, simps
from scipy.linalg import sqrtm
from skimage.transform import radon, rescale, iradon

#from custom modules
from utils import plotting, arrays
#script settings
np.random.seed(0)
#helper

def cdist(XA, XB, metric='euclidean',*args, **kwargs):
    try:
        return sccdist(XA,XB, metric=metric, *args, **kwargs)
    except ValueError:
        return sccdist(np.matrix(XA).T, np.matrix(XB).T, metric=metric, *args, **kwargs)

def OP(T):
    '''Makes a function out of any numpy array'''
    return lambda x: T@x


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



def KLT(a):
    """
    Returns Karhunen Loeve Transform of the input and the transformation matrix and eigenval
    source: https://sukhbinder.wordpress.com/2014/09/11/karhunen-loeve-transform-in-python/  
    """
    val, vec = np.linalg.eig(np.cov(a))
    klt = np.dot(vec,a)
    return klt,vec,val



def w_pCN(measurement, ObservationOp, PriorOp, sample_shape, n_iter, beta, phi, xi = None, burnin_ratio = 2, debug=False):
    #initialize xi and u
    if not xi: xi = np.random.standard_normal(sample_shape)
    proposal = PriorOp(xi)
    u = ObservationOp(proposal)
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
        proposal_hat = PriorOp(xi_hat)
        u_hat = ObservationOp(proposal_hat)
        #evaluate update
        log_prob = min(phi(u, measurement)-phi(u_hat, measurement), 0)#anti overflow
        if np.random.rand() <= np.exp(log_prob):
            xi = xi_hat
            proposal = proposal_hat
            u = u_hat
            acc = True
        #store samples
        if i >= n_iter/burnin_ratio:
            samples.append(proposal)
            accepted.append(acc)

        #debug code
        if debug:
            av_acc = av_acc +(acc-av_acc)/(i+1) # discounted acceptance rate
            log_probs.append(log_prob)
    #debug code
    if debug:
        return samples, acc, av_acc, xi, log_probs
   
    return samples, accepted


if __name__=='__main__':
    #size of sample area
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    imgSize = 20j
    #generate points to sample from, eg. image data
    xx, yy = np.mgrid[xmin:xmax:imgSize, ymin:ymax:imgSize]
    X = np.vstack([xx.ravel(), yy.ravel()]).T

    #make prior
    T = sqrtm(
        matern_cov(
            dists=cdist(X,X) * 10,
            nu = 1.5
        )
    ) * 2

    def PriorTransform(fx, T):
        'T needs to an Automorphism, maps R^n to smoother functions. Assumes f was generated on a set of points X where T is some function of cdist(X)'
        shape = fx.shape
        return(T@np.ravel(fx)).reshape(shape).real
    
    #generate image data
    f = lambda xx, yy: np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)
    fX = f(xx, yy)

    #transform back to meshgird format
    Xp, Yp, Zp = plotting.plot_contour(X[:,0], X[:,1], np.ravel(fX))
    
    #show contour plot
    plt.contourf(Yp, Xp, Zp)
    plt.show()
    
    #take measurements
    theta = np.linspace(0, 180, 5, endpoint=True)
    sinogram = radon(fX, theta, circle=False)
    #setup integration steps
    delta = 1/imgSize * 1/theta.max()-theta.min()
    
    #setup measurement operator
    ObservationOp = partial(radon, theta=theta, circle=False)
    #setup prior operator
    PriorOp = partial(PriorTransform, T=T)
    
    #error function
    _phi = lambda x,y,dx: np.sum((x-y)**2) * dx
    phi = partial(_phi, dx=delta)

    #run w_pcn
    samples, accepted = w_pCN(sinogram, ObservationOp, PriorOp, fX.shape, 50000, .5, phi, burnin_ratio=5)

    av = np.mean(samples, axis=0)
    plt.imshow(av)
    print(
        (
            'Acceptance Probability: %s \n'%(np.mean(accepted),),
            'Number of Samples: %s'%(len(samples))
        )
    )
    plt.show()
   
    pass