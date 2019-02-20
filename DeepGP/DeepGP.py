from sklearn import gaussian_process as gp
import numpy as np


dim = 100
sigma = .1
#Make Data
####################
X = np.linspace(0,2*np.pi, dim)
y = np.sin(X) + sigma * np.random.standard_normal(dim)
##################

kernel = gp.kernels.Matern()

print(kernel.hyperparameters)

reg = gp.GaussianProcessRegressor()

