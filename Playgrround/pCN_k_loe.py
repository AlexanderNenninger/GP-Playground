import tensorflow as tf
import numpy as np
import scipy.linalg as lina
import scipy.stats as stats
import matplotlib.pyplot as plt

dim = 256
sigma = .1
#############generate data
x = np.linspace(0, 2*np.pi, dim)
y = np.sin(x) + sigma*np.random.standard_normal(dim) + 1

plt.plot(x,y)
####################pcn
# Function Definitions:

def calculate_cov(x, ufunc, l = .1):
    N = len(x)
    C = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            C[i,j] = np.exp(ufunc(x[i], x[j])**2/(2*l))
    return C


phi = lambda x, y: np.linalg.norm(x-y)

C = calculate_cov(x, phi)
beta = 0.01
u = np.zeros_like(x)

path = []
for k in range(100):
    u_hat = np.sqrt(1-beta**2)*u + beta* C@np.random.standard_normal(dim)

    if np.exp(phi(u,y) - phi(u_hat, y)) >= np.random.uniform():
        u = u_hat
    path.append(u)
    
path = np.array(path)

for p in path[-10:-1]:
    plt.plot(x,p, 'x')

plt.show()
