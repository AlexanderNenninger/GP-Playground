'''
Zhis is the main code file of my bachelor's thesis. 
Ideas:
    1. Only update the jump parameter beta during the first 10,000 interations, then fix or reduce update rate
    2. Adapt the length scale parameters during the iteration, maybe for the first 10,000, then fix or reduce update rate
    3. Implement more layers, but I need a powerful server for that

    Next Steps:
    1. Modularize the code more, so functionality is easily added
    2. Implemet idea Nr.2
    3. After Thesis: Write Performance critical components in C++/C
'''



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
from scipy.linalg import sqrtm

from skimage.transform import radon, rescale, iradon
from skimage.io import imread

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

def center_scale(arr: np.array)->np.array:
    'Centers and rescales numpy array to [-1,1]'
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min)/(arr_max-arr_min))*2 - 1, arr_min, arr_max

def icenter_scale(arr: np.array, arr_min, arr_max)->np.array:
    return (arr+1)/2*(arr_max-arr_min) + arr_min


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


def w_pCN(measurement, ObservationOp, PriorOp, sample_shape, n_iter, beta, phi, xi = None, burnin_ratio = 2, debug=False):
    

    #define internal functions. cleans up the code
    def _update_beta(beta, beta_0, log_prob, betaPdf, betaRV):
        #update beta - draw from exponential distribution, beta sets the location of the peak of the pdf
        beta_hat = betaRV(beta)
        #calculate the P(beta_hat|beta)/P(beta|beta_hat)*pi_0(b)
        beta_update_prob = np.exp(log_prob)
        beta_update_prob *= betaPdf(beta_hat, beta)
        beta_update_prob /= betaPdf(beta, beta_hat)
        beta_update_prob *= betaPdf(beta_hat, beta_0)
        beta_update_prob /= betaPdf(beta, beta_0)
        if np.random.rand() <= beta_update_prob:
            return beta_hat, True
        return beta, False
    
    #initialize values
    #xi
    if not xi: xi = np.random.standard_normal(sample_shape)
    
    #beta
    update_beta_until = 10000
    beta_0 = .5
    sigma = 20
    betaPdf = lambda x, theta: stats.truncnorm.pdf((x-theta)*sigma, a=-theta*sigma, b=sigma*(1-theta))*sigma
    betaRV = lambda theta: stats.truncnorm.rvs(-sigma*theta, sigma*(1-theta)) / sigma + theta
    
    #u
    proposal = PriorOp(xi)
    u = ObservationOp(proposal)
    
    #initialize arrays in which samples are stored
    samples = []
    accepted = []
    
    #debug code
    if debug:
        av_acc = 0
        log_probs = []
        betas = []
    
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
            u_acc = True

        if i < update_beta_until:
            beta, beta_acc = _update_beta(beta, beta_0, log_prob, betaPdf, betaRV)
            betas.append(beta)
        elif i == update_beta_until:
            beta = np.mean(betas[int(len(betas)*burnin_ratio):])
            print(i, ' Stopped updating Beta')

        #debug messages
        if i%1000 == 0 or i==1:
            print('%s Beta: %s, LogProb: %s'%(i, beta, log_prob))
        
        #store samples
        if i >= n_iter*burnin_ratio:
            samples.append(proposal)
            accepted.append((u_acc, beta_acc))

        #debug code
        if debug:
            av_acc = av_acc +(acc-av_acc)/(i+1) # discounted acceptance rate
            log_probs.append(log_prob)
    #debug code
    if debug:
        return samples, accepted, av_acc, xi, log_probs, betas
   
    return samples, accepted


if __name__=='__main__':
    #load image
    image = imread('data/phantom.png', as_gray=True)
    image = rescale(image, scale=.05, mode='reflect', multichannel=False)


    
    #size of sample area
    sample_area = [(0, 1)]*image.ndim
    
    #pixel locations
    xx, yy = tuple(
        np.linspace(
            sample_area[i][0],sample_area[i][1],image.shape[i]
        ) for i in range(image.ndim)
    )

    #coordinate list from coodinate vectors
    xx, yy = np.meshgrid(xx, yy)
    X = np.vstack([xx.ravel(), yy.ravel()]).T

    #make prior
    T = sqrtm(
        matern_cov(
            dists=cdist(X,X) * 15,
            nu = 1.5
        )
    ) * 2

    def PriorTransform(fx, T):
        'T needs to an Automorphism, maps R^n to smoother functions. Assumes f was generated on a set of points X where T is some function of cdist(X)'
        shape = fx.shape
        return(T@np.ravel(fx)).reshape(shape).real
    
    #generate image data
    # f = lambda xx, yy: np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)
    # fX = f(xx, yy)
    fX, _, _ = center_scale(image)
    #transform back to meshgird format
    Xp, Yp, Zp = plotting.plot_contour(X[:,0], X[:,1], np.ravel(fX))
    
    #make contour plot - check if coordinates are correct
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].contourf(Xp, Yp, Zp)
       
    #take measurements
    theta = np.linspace(0, 180, 10, endpoint=True)
    sinogram = radon(fX, theta, circle=False)
    #corrupt measurements with gaussian noise
    sinogram += .1 * (sinogram.max() - sinogram.min()) * np.random.standard_normal(sinogram.shape)
    #check quality of algebraic inversion
    inv = iradon(sinogram, theta, circle=False)
    inv = rescale(inv, 20, multichannel=False)
    ax[0,1].imshow(inv)
   
    #setup integration steps - the spacing of the measurement points is proportional to 1/image.shape[0] in skimage.radon
    delta = 1/image.shape[0] * 1/theta.max()-theta.min()
    
    #setup measurement operator
    ObservationOp = partial(radon, theta=theta, circle=False)
    #setup prior operator
    PriorOp = partial(PriorTransform, T=T)
    
    #error function
    _phi = lambda x,y,dx: np.sum((x-y)**2) * dx
    phi = partial(_phi, dx=delta)

    #run w_pcn
    samples, accepted, av_acc, xi, log_probs, betas = w_pCN(sinogram, ObservationOp, PriorOp, fX.shape, 50000, .9, phi, burnin_ratio=.2, debug=True)

    av = np.mean(samples, axis=0)
    av = rescale(av, 20, multichannel=False)
    ax[1,0].imshow(av)
    ax[1,1].plot(betas)
    print(
        ' Acceptance Probability: %s \n'%(np.mean(accepted),),
        'Number of Samples: %s'%(len(samples))
    )
    plt.show()
   
    pass