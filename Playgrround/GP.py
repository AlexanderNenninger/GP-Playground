import numpy as np
import scipy.spatial as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.special as special
from sklearn.gaussian_process.kernels import Matern



dim = 256
sigma = .01
#Make Data
####################
X = np.linspace(0,2*np.pi, dim)
y = np.sin(X) + sigma * np.random.standard_normal(dim)
##################



l = 1.5


sq_exp = lambda r, _: np.exp(-(r)**2/(2*l**2))

#define required finctions
phi = lambda x, y: np.linalg.norm(x-y)
mat = Matern(length_scale=.1)
mat.nu = 2.5

def calc_cov_matrix(x, ufunc):
#Calculates the covariance matrix for a given covariance function    
    C = np.zeros((*x.shape,*x.shape))
    for i in range(dim):
        for j in range(dim):
            C[i,j] = ufunc((x[i]- x[j])**2, 1, 2, .5)
    return C

#Calculate the Covariance Matrix and its Cholesky Decomposition
C = mat.__call__(np.array([X,]).T)
T = np.linalg.cholesky(C).conj().T

#Run w-pCN
n_iter = 100000
xi = np.random.standard_normal(dim)
accepted = 0
path = []
for k in range(n_iter):
    beta = .5
    xi_hat = np.sqrt(1-beta**2) * xi + beta * np.random.standard_normal(dim)
    
    log_prob = phi(T@xi, y) - phi(T@xi_hat,y)
    if  np.exp(log_prob) >= np.random.rand():
        xi = xi_hat
        accepted += 1
    path.append(T@xi)
print('Acceptance probability: {0}%'.format(accepted/n_iter*100))

path = np.asarray(path)
mean = np.mean(path[-int(n_iter*.5):-1], axis=0)
plt.plot(X,y)
plt.plot(X,mean)
plt.show()