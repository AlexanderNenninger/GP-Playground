'''
This is the NICE code accompanying my bachelor's thesis. I was able to cut out a lot of unnecessary stuff.

Problems:
    1. The jump parameter beta always converges to the area around beta_0, the peak of the prior. Something is wrong there!

Ideas:
    1. Adapt beta based on acceptance probability every 100 iterations - Done
    2. Adapt the length scale parameters during the iteration, maybe for the first 10,000 iterations, then fix or reduce update rate
    3. Implement more layers, but I need a powerful server for that and need to switch to an ONB representation of the function space
    4. Performance Metrics - Need to decide with Tim which ones he wants to see

Next Steps:
    1. Modularize the code more, so functionality is easily added - done
    2. Implement idea Nr.2 - done
    3. Implement Idea Nr.4 - done
    5. Document Code - done
    
After Thesis (maybe): 
    1. Implement different Priors - Besov is an option!; Might need to switch to basis representations of the space then.
    2. Write performance critical components in C++/C
    3. Optimize regularization parameter
    4. Use a coarser tiling for deeper layers
'''
import copy
from pathlib import Path
import time
from datetime import datetime
import pickle

import numpy as np
from matplotlib import pyplot as plt
#from scipy.stats import lognorm

from operators import mOp, CovOp, mDevice, RadonTransform
from logPdfs import lognorm_pdf, exp_pdf, log_acceptance_ratio
import dataLoading
import plotting

def update_beta(beta, acc_prob, target):
	'small function for updating beta'
	beta += .01*(acc_prob-target)
	return np.clip(beta, 2**(-15), 1-2**(-15))

class wpCN(object):
	def phi(self, x, y):
		return np.sum((x-y)**2) * self.dx

	def __init__(self, ndim, size, Covariance: CovOp, T: mDevice):
		self.data = data
		self.ndim = ndim
		self.size = size
		self.shape = (size,)*ndim
		self.C = Covariance
		self.C_hat = copy.deepcopy(self.C)
		self.C_2 = CovOp(1,1,1,.1)
		self.T = T

		self.dx = 1/T.len
		self.xi = np.random.standard_normal(shape)
		self.u = C(self.xi)
		self.m = T(self.u)

		self.Temperature = 1

		self.xi_1 = np.random.standard_normal(1)

		self.beta = (.5, .5)
		self.samples = []
		self.probs = []
		self.betas  = []
		self.data = []

	
	def _0_layer(self, data):
		#base layer
		xi_hat = np.sqrt(1 - self.beta[0]**2) * self.xi + self.beta[0] * np.random.standard_normal(self.shape)
		u_hat = self.C(xi_hat)
		m_hat = self.T(u_hat)
		logProb = min(self.phi(self.m, data) - self.phi(m_hat, data), 0) * self.Temperature
		if np.random.rand() <= np.exp(logProb):
			self.xi = xi_hat
			self.u = u_hat
			self.m = m_hat
		self.beta = update_beta(self.beta[0], np.exp(logProb), .23) , self.beta[1]
	
	def _1_layer(self, data):
		#second layer
		xi_hat = np.sqrt(1 - self.beta[1]**2) * self.xi_1 + self.beta[1] * np.random.standard_normal(1)
		h_hat = np.exp(xi_hat)
		self.C_hat.sigma = h_hat
		
		logProb = self.phi(self.T(self.C(self.xi)), data) - self.phi(self.T(self.C_hat(self.xi)), data)
		logProb += log_acceptance_ratio(self.xi_1, xi_hat, self.beta[1], self.C_2)
		logProb = min(logProb, 0)  * self.Temperature
		if np.random.rand() <= np.exp(logProb):
			self.xi_1 = xi_hat
			h = h_hat
			self.C.sigma = h
		self.beta = self.beta[0], update_beta(self.beta[1], np.exp(logProb), .23)

	def sample(self, data, niter = 10000):
		t_start = time.time()
		i=0
		while i <= niter:
			i+=1
			self._0_layer(data)
			self._1_layer(data)
			self.samples.append((self.u, self.C.sigma))
			self.betas.append(self.beta)
			# Kill chain if beta leaves a sensible region
			if i%1000==0:
				if min(self.beta) < 0.05:
					self.Temperature /= 2
					self.samples = []
					self.betas = []
					self.beta = (.5,.5)
					print('Temperature decreased to: ', self.Temperature)
					i = 0
				if max(self.beta) > 0.95:
					self.Temperature *= 2
					self.samples = []
					self.betas = []
					self.beta = (.5,.5)
					print('Temperature increased to: ', self.Temperature)
					i = 0
				print(i, 'Beta: ', self.beta)
		t_end = time.time()
		self.t_delta = (t_end - t_start)
		print('#Samples:%s, Time: %ss'%(niter, self.t_delta))
		self.reconstruction = np.mean([s[0] for s in self.samples], axis = 0)
		self.var = np.var([s[0] for s in self.samples], axis = 0)
		self.heightscale = np.mean([s[1] for s in self.samples], axis = 0)


if __name__=='__main__':	
	image_path = Path('data/phantom.png')
	size = 15
	image = dataLoading.import_image(image_path, size=size)

	ndim = image.ndim
	shape = (size,)*ndim

	C = CovOp(ndim, size, sigma=np.ones((size,)*ndim), ro=.02)

	means = np.array([
		(.25, .25),	(.25, .5), (.25, .75),
		(.5, .25), (.5, .5), (.5, .75),
		(.75, .25), (.75, .5), (.75, .75),
	])
	T = RadonTransform(ndim, size, np.linspace(0, 180, 10))

	data = T(image)
	data += .1 * np.random.standard_normal(data.shape)
	fbp = T.inv(data)
	print('Shape of Data:', data.shape, end='	\n')

	chain = wpCN(ndim, size, C, T)

	n_iter = 1000
	chain.sample(data, n_iter)

	f_name = '%s_n.pkl'%datetime.now().replace(microsecond=0).isoformat().replace(':','-') + str(n_iter)
	with open(f_name, 'wb') as f:
		pickle.dump(chain, f)

	plotting.plot_result_2d(image, chain, means, C, size, data, fbp)
	plt.plot([b[0] for b in chain.betas])
	plt.plot([b[1] for b in chain.betas])
	plt.show()