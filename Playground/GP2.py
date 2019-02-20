import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

dim = 256
X = np.array([np.linspace(0,1, dim)]).T
y = X.T[0]

estimator = GaussianProcessRegressor(Matern(length_scale=.1, nu=2.5))
#estimator.fit(X,y)
for i in range(2):
    y = estimator.sample_y(X, random_state=np.random.randint(0,100))
    plt.plot(X,y)
plt.show()