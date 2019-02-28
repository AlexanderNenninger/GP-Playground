from typing import Callable
import numpy as np
import scipy.spatial as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.special as special
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import Kernel


dim = 100
sigma = .1
#Make Data
####################
X = np.matrix(np.linspace(0,2*np.pi, dim)).T
y = np.round(np.sin(X)) + sigma * np.random.standard_normal(X.shape)
##################



length_scale = np.ones(1)


#define required functions
phi = lambda x, y: np.linalg.norm(x-y)
ker = Matern(length_scale=length_scale)
ker.nu = 1.5

def calc_cov_matrix(X_train: np.array, X_eval: np.array, ker: Kernel):
    '''uses kernels form scipy.gaussian_process.kernels to calculate a cov matrix'''
    return ker(X_train, X_eval)


def w_pCN_step(xi: np.array, y: np.array, beta: float, T: np.array, phi: Callable, dim: int=-1):
    '''defines one step of the w-pCN algorithm as found in https://arxiv.org/pdf/1803.03344.pdf Section 3.1
    xi : current location of markov chain
    y : sample data
    beta: jump parameter
    T: To center the coordinates
    dim : dimensionality of the input. 
    '''
    #setup vars
    accepted = False
    if dim <0: dim = len(xi)
    
    #propose update
    xi_hat = np.sqrt(1-beta**2) * xi + beta * np.random.standard_normal(dim)
    
    #check if update improves upon xi
    log_prob = min(0, phi(T@xi, y) - phi(T@xi_hat,y)) # min to combat overflow
    if  np.exp(log_prob) >= np.random.rand():
        xi = xi_hat
        accepted = True
    return xi, accepted

#Calculate the Covariance Matrix and its Cholesky Decomposition
C = calc_cov_matrix(X,X, ker)
T = C #np.linalg.cholesky(C).conj().T

#Run w-pCN
n_iter = 10000
xi = np.random.standard_normal(dim)
accepted = 0
path = []
alpha = .5
beta = .5

for k in range(n_iter):
    
    ###pcn step
    xi, acc = w_pCN_step(xi, y, beta, T, phi)
    accepted +=acc
    path.append(T@xi)
    ##gibbs step
    beta_hat =  np.sqrt(1-alpha)**2 * beta + alpha * np.random.rand()
    



print('Acceptance probability: {0}%'.format(accepted/n_iter*100))


#Viz
path = np.asarray(path)
mean = np.mean(path[-int(n_iter*.5):-1], axis=0)
std = np.sqrt(np.var(path[-int(n_iter*.5):-1], axis=0))

plt.plot(X,y, 'g.')
plt.plot(X,np.clip(mean, -10,10))
plt.plot(X,np.clip(mean+std,-10,10), 'b--')
plt.plot(X,np.clip(mean-std,-10,10), 'b--')
plt.show()



