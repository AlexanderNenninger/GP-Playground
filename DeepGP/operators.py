import numpy as np
'''
Define Operators. mOp is for taking measurements, CovOp is for mapping to the right function space. 
'''
class mOp(object):
	'L_p(r) -> R'
	g = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)
	m = 0
	s = 1
	
	def __init__(self, dim, mean=m, sigma=s):
		self.sigma = sigma
		self.mean = mean
		
		self.f = lambda x, mean: np.exp(-(x-mean)**2/sigma**2/2) / np.sqrt(2*np.pi) / sigma
		
		self.filter = self.f(
			np.linspace(0,1,dim), 
			mean
		)
		self.filter/=self.filter.sum()

	def __call__(self, x):
		return np.sum(self.filter*x)/x.shape[0]
		
class CovOp(object):
	'L_p[0,1]->L_p[0,1]'
	s = 1
	r = 1
	k = np.zeros((1,1))
	g = lambda x: np.exp(-x/r)
	
	def __init__(self, shape, sigma=s, ro=r):
		self.ker = np.zeros(shape)
		self.ro = ro * shape[0]
		self.sigma = sigma
		self.f = lambda r: np.exp(-r/self.ro)
		
		for i in range(shape[0]):
			for j in range(shape[1]):
				self.ker[i,j] = self.f(np.abs(i-j))
		self.ker = np.linalg.cholesky(self.ker)
	
	def __call__(self, x):
		return self.sigma * np.dot(self.ker,x)