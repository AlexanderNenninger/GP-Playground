'''
This is the main code file of my bachelor's thesis. 

Problems:
    1. The jump parameter beta always converges to the area around beta_0, the peak of the prior. Something is wrong there!

Ideas:
    1. Only update the jump parameter beta during the first 10,000 iterations, then fix or reduce update rate - Done
    2. Adapt the length scale parameters during the iteration, maybe for the first 10,000 iterations, then fix or reduce update rate
    3. Implement more layers, but I need a powerful server for that and need to switch to an ONB representation of the function space
    4. Performance Metrics - Need to decide with Tim which ones he wants to see

Next Steps:
    1. Modularize the code more, so functionality is easily added - done
    2. Implement idea Nr.2
    3. Implement Idea Nr.4
    
After Thesis (maybe): 
    1. Implement different Priors - Besov is an option!; Might need to switch to basis representations of the space then.
    2. Write performance critical components in C++/C
    3. Optimize regularization parameter
    4. Use a coarser tiling for deeper layers
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

def PriorTransform(fx, T, amplitude = 1):
    ''''
    T needs to be a symmetric, positive semi-definite matrix, which maps White Noise on R^n to smoother functions.
    Assumes f was generated on a set of points X where T generates a RKHS
    '''
    shape = fx.shape
    return(T@np.ravel(fx)).reshape(shape).real * amplitude


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


def w_pCN(measurement, ObservationOp, PriorOp, sample_shape, X, n_iter, beta, phi, T, xi = None, xi2=None, burnin_ratio = 2, debug=False):
    

    #define internal functions. cleans up the code
    def _update_beta(beta, acc_rate, betaRV):
        'Update beta with a metropolis step. Could be optimized!'
        #update beta
        beta = betaRV(max(beta + .1*(acc_rate - .23), .9999))
        return beta
    
    #this is what actually reconstructs images
    def _update_main(xi, proposal, u, beta, phi, PriorOp, ObservationOp):
        #propose update
        xi_hat =  np.sqrt(1-beta**2) * xi + beta * np.random.standard_normal(xi.shape)
        proposal_hat = PriorOp(xi_hat)
        u_hat = ObservationOp(proposal_hat)
        #evaluate update
        log_prob = min(phi(u, measurement)-phi(u_hat, measurement), 0)#anti overflow
        if np.random.rand() <= np.exp(log_prob):
            return xi_hat, proposal_hat, u_hat, log_prob, True
        return xi, proposal, u, log_prob, False

    #initialize values
    #xi
    if not xi: xi = np.random.standard_normal(sample_shape)
    
    #lengthscale
    if not xi2: xi2 = np.random.standard_normal(sample_shape)
    PriorOp_0 = PriorOp
    
    #beta
    # update_beta_until = 10000
    # beta_0 = .9
    sigma = 20
    # betaPdf = lambda x, theta: stats.truncnorm.pdf((x-theta)*sigma, a=-theta*sigma, b=sigma*(1-theta))*sigma
    betaRV = lambda theta: stats.truncnorm.rvs(-sigma*theta, sigma*(1-theta)) / sigma + theta
    
    #u - the observations
    proposal = PriorOp(xi)
    u = ObservationOp(proposal)
    
    #initialize arrays in which samples are stored
    samples = []
    accepted = []
    
    #discount rate
    discount_rate = .9
    discount_scal = discount_rate/(1-discount_rate)
    disc_acc = 0
    #debug code
    if debug:
        av_acc = 0
        log_probs = []
        betas = []
    
    #the loop
    for i in range(n_iter):
        
        #chain for target distribution on image space
        xi, proposal, u, log_prob, u_acc = _update_main(xi, proposal, u, beta, phi, PriorOp, ObservationOp)

        #update lengthscale - WIP and ridiculously inefficient!!!
        xi2_hat = np.sqrt((1-beta**2)) * xi2 + beta * np.random.standard_normal(xi2.shape)
        amplitude_hat = PriorOp(xi2_hat)
        PriorOp_hat = partial(PriorTransform, T=T, amplitude=amplitude_hat)
        log_prob2 = np.sum(PriorOp(xi2_hat)**2 - PriorOp_hat(xi2)**2 + PriorOp_0(xi2_hat)**2 - PriorOp_0(xi2)**2)
        if np.random.rand() <= np.exp(log_prob + log_prob2):
            xi2 = xi2_hat
            PriorOp = PriorOp_hat
        
        #beta update
        disc_acc = discount_rate * disc_acc + u_acc
        beta = _update_beta(beta, disc_acc * discount_scal, betaRV)
        betas.append(beta)

        #store samples
        if i >= n_iter*burnin_ratio:
            samples.append(proposal)
            accepted.append(u_acc)

        #debug code
        if debug:
            av_acc = av_acc + (u_acc-av_acc)/(i+1) # discounted acceptance rate
            log_probs.append(log_prob)
            #debug messages
            if i%1000 == 0 or i==1:
                print('%s Beta: %s, LogProb: %s, AccProb: %s'%(i, beta, log_prob, disc_acc))
    #debug code
    if debug:
        return samples, accepted, av_acc, xi, log_probs, betas
   
    return samples, accepted


if __name__=='__main__':
    #load image
    image = imread('data/phantom.png', as_gray=True)
    image = rescale(image, scale=.05, mode='reflect', multichannel=False)


    
    #size of sample area - the unit square is a solid choice for the most part, be aware of image dimensions
    sample_area = [(0, 1)]*image.ndim
    
    #find pixel locations
    xx, yy = tuple(
        np.linspace(
            sample_area[i][0],sample_area[i][1],image.shape[i]
        ) for i in range(image.ndim)
    )

    #coordinate list from coodinate vectors
    xx, yy = np.meshgrid(xx, yy)
    X = np.vstack([xx.ravel(), yy.ravel()]).T

    #calculate distance matrix with lengthscale
    lengthScale = 15 * np.ones_like(X.shape)
    X *= lengthScale
    dists = cdist(X,X)

    #make kernel for RKHS of Prior space
    T = sqrtm(
        matern_cov(
            dists=dists,
            nu = 1.5
        )
    ) * np.sqrt(2)
    
    #setup prior operator
    PriorOp = partial(PriorTransform, T=T)
   
    #normalize function values to [-1,1]
    fX, _, _ = center_scale(image)
    #transform back to meshgird format 
    Xp, Yp, Zp = plotting.plot_contour(X[:,0], X[:,1], np.ravel(fX))
    
    #make contour plot - it's easy to spot errors this way
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
    #error functional on the measurements
    _phi = lambda x,y,dx: np.sum((x-y)**2) * dx
    phi = partial(_phi, dx=delta)
    
    #setup measurement operator - this is bad for performance, due to function call overhead, but really good for development
    ObservationOp = partial(radon, theta=theta, circle=False)
    
    #run the w_pcn chain
    samples, accepted, av_acc, xi, log_probs, betas = w_pCN(sinogram, ObservationOp, PriorOp, fX.shape, X, 50000, .5, phi, T, burnin_ratio=.2, debug=True)

    #evaluate markovchain
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