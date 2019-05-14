import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from functools import partial

support = np.linspace(0,1,99)

sigma = 10
betaPdf = lambda x, theta, sigma: stats.truncnorm.pdf((x-theta)*sigma, a=-theta*sigma, b=sigma*(1-theta))*sigma
betaRvs = lambda theta, sigma: stats.truncnorm.rvs(-sigma*theta, sigma*(1-theta)) / sigma + theta

X = np.linspace(-.1,1.1, 100)
Beta = np.linspace(0,1,5)
fig, ax = plt.subplots(nrows=5, ncols=1,)
plt.subplots_adjust(hspace=.5)
for i, beta in enumerate(Beta):
    samples = [betaRvs(beta, sigma) for i in range(10000)]
    ax[i].plot(X, betaPdf(X, beta, sigma))
    ax[i].hist(samples, density=True)
plt.show()