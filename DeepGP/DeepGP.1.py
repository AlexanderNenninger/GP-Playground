from typing import Callable
import numpy as np
import scipy.spatial as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.special as special
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import Kernel
from mayavi import mlab

dim = 50
n_samples = 100
sigma = .02
#Make Data
####################
X_train = np.matrix([np.linspace(0,2*np.pi, dim)]*n_samples).T
y = np.round(np.sin(X_train),0) + sigma * np.random.standard_normal(X_train.shape)
##################
X_eval = np.matrix(np.linspace(0, 2*np.pi, dim)).T

l = np.ones(n_samples)
print('L: {1} -> {0}'.format(X_eval.shape, X_train.shape))
#define required functions
phi = lambda x, y: np.linalg.norm(x-y)
ker = Matern()
ker.nu = 1.5
ker.length_scale = l

def calculate_cov_matrix(X_eval: np.array, X_train: np.array, ker: Kernel, eval_gradient=False):
    #allows for possibly additional functionality
    return ker(X_eval, X_train, eval_gradient=eval_gradient)


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
    xi_hat = np.sqrt(1-beta**2) * xi + beta * np.random.standard_normal(xi.shape)
    u = T@xi
    u_hat = T@xi_hat
    #check if update improves upon xi
    log_prob = min(0, phi(u, y) - phi(u_hat,y)) # min to combat overflow
    if  np.exp(log_prob) >= np.random.rand():
        xi = xi_hat
        u = u_hat
        accepted = True
    return xi, u, accepted, log_prob

C = calculate_cov_matrix(X_train, X_eval, ker, eval_gradient=False)

#Calculate the Covariance Matrix and its Cholesky Decomposition


    #    """Return the kernel k(X, Y) and optionally its gradient.

    #     Parameters
    #     ----------
    #     X : array, shape (n_samples_X, n_features)
    #         Left argument of the returned kernel k(X, Y)

    #     Y : array, shape (n_samples_Y, n_features), (optional, default=None)
    #         Right argument of the returned kernel k(X, Y). If None, k(X, X)
    #         if evaluated instead.

    #     eval_gradient : bool (optional, default=False)
    #         Determines whether the gradient with respect to the kernel
    #         hyperparameter is determined.

    #     Returns
    #     -------
    #     K : array, shape (n_samples_X, n_samples_Y)
    #         Kernel k(X, Y)

    #     K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
    #         The gradient of the kernel k(X, X) with respect to the
    #         hyperparameter of the kernel. Only returned when eval_gradient
    #         is True.
    #     """
        # from \sklearn\gaussian_process\kernels.py
T = C #np.linalg.cholesky(C).conj().T

#Run w-pCN
n_iter = 100000
xi = np.random.standard_normal(X_eval.shape)
accepted = 0
path = []
alpha = 1
beta = .05
gamma = 1.7

for k in range(n_iter):
    ###pcn step
    xi, u, acc, log_prob = w_pCN_step(xi, y, beta, T, phi)
    accepted +=acc
    path.append(u)
   
    # ##gibbs step
    # beta_hat = alpha * np.random.rand()**gamma
    # if np.exp(log_prob) * beta**gamma-1 / beta_hat**gamma-1 >= np.random.rand():
    #     beta = beta_hat
 



print('Acceptance probability: {0}%'.format(accepted/n_iter*100))


#Viz
path = np.asarray(path)
mean = np.squeeze(np.mean(path[-int(n_iter*.5):-1], axis=(0)))
std = np.squeeze(np.sqrt(np.var(path[-int(n_iter*.5):-1], axis=(0))))


# for m in range(mean.shape[0]):
#     print('mean[{0}]={1} +-{2}'.format(m, mean[m], std[m]))

plt.plot(X_train, y, 'rx')



plt.plot(X_eval, mean)
plt.plot(X_eval, np.clip(mean+std,-10,10), 'b--')
plt.plot(X_eval, np.clip(mean-std,-10,10), 'b--')
plt.show()



