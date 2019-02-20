from typing import Callable
import numpy as np
import scipy.spatial as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.special as special
from sklearn.gaussian_process.kernels import Matern



dim = 100
sigma = .1
#Make Data
####################
X = np.linspace(0,2*np.pi, dim)
y = np.sin(X) + sigma * np.random.standard_normal(dim)
##################



length_scale = 2


sq_exp = lambda r, _: np.exp(-(r)**2/(2*length_scale**2))

#define required functions
phi = lambda x, y: np.linalg.norm(x-y)
mat = Matern(length_scale=length_scale)
mat.nu = 2.5

def calc_cov_matrix(x, ufunc):
    '''Calculates the covariance matrix for a given covariance function'''
    C = np.zeros((*x.shape,*x.shape))
    for i in range(dim):
        for j in range(dim):
            C[i,j] = ufunc((x[i]- x[j])**2, 1, 2, .5)
    return C

def w_pCN_step(xi: np.array, y: np.array, beta: float, T: np.array, phi: Callable, dim: int=-1):
    '''defines one step of the w-pCN algorithm as found in https://arxiv.org/pdf/1803.03344.pdf Section 3.1
    xi : current location of markov chain
    y : sample data
    beta: jump parameter
    T: Q
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
C = mat.__call__(np.array([X,]).T)
T = C #np.linalg.cholesky(C).conj().T

#Run w-pCN
n_iter = 10000
xi = np.random.standard_normal(dim)
accepted = 0
path = []
for k in range(n_iter):
    beta = .1
    xi, acc = w_pCN_step(xi, y, beta, T, phi)
    accepted +=acc
    path.append(T@xi)

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