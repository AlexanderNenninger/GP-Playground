import math
import numpy as np

from scipy.spatial.distance import pdist, cdist as sccdist, squareform
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
from scipy.integrate import romb, simps
from scipy.linalg import sqrtm
from skimage.transform import radon, rescale, iradon

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



def KLT(a):
    """
    Returns Karhunen Loeve Transform of the input and the transformation matrix and eigenval
    Ex:
    import numpy as np
    a  = np.array([[1,2,4],[2,3,10]])
    
    kk,m = KLT(a)
    print kk
    print m
    
    # to check, the following should return the original a
    print np.dot(kk.T,m).T
    source: https://sukhbinder.wordpress.com/2014/09/11/karhunen-loeve-transform-in-python/  
    """
    val, vec = np.linalg.eig(np.cov(a))
    klt = np.dot(vec,a)
    return klt,vec,val



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
   
    return samples, accepted


#potential
phi = lambda x, dx: (1 - romb(np.abs(x), dx) ** 2 ) ** 2



# def dist(D1, D2, f1, f2, cov, dx1, dx2):
#     '''
#     'Distance' between two functions defined on different sets. 
#     D1 is the domain of f1.
#     D2 is the domain of f2.
#     f1 and f2 are function values on their respective domains.
#     cov is a callable weighting funtion.
#     This is not a metric, as Manuel pointed out.
#     '''
#     if not callable(cov):
#         raise TypeError('cov need to be a function of two arguments')
#     d = 0
#     for i, x1 in enumerate(D1):
#         for j, x2 in enumerate(D2):
#             d += cov(x1,x2) * np.linalg.norm(f1[i] - f2[j])
#     return d# * dx1 * dx2


if __name__=='__main__':

    x0 = np.linspace(0,1,100)
    y0 = np.linspace(0,1, 100)

    xx, yy = np.meshgrid(x0, y0)

    f = lambda xx, yy: np.sin(np.pi * xx) * np.sin(np.pi * yy)
    
    fX = np.dstack(f(xx, yy)).reshape(-1, 1)
    X = np.dstack([xx, yy]).reshape(-1, 2)
    
    T = sqrtm(matern_cov(cdist(X,X), 2.5))


    
    
    samples, accepted = w_pCN(f(xx,yy), )
    
    ###transform
    plt.imshow(f(xx,yy))
    plt.show()

    theta = np.linspace(0,180, max(image.shape))
    sinogram = radon(image, theta=theta, circle=False)





    reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
    plt.imshow(reconstruction_fbp)
    plt.show()

    err = image - reconstruction_fbp

    plt.imshow(err)
    plt.show()

   
    pass