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
    'L_p[0,1]^ndim->L_p[0,1]^ndim'
    def __init__(self, ndim, size, sigma=1, ro=1):
        self.ndim = ndim
        self.size = size
        self.shape = (size,)*2**ndim
        self.C = np.zeros(self.shape)
        self.ro = ro * size
        self.sigma = sigma
        self.f = lambda r: (1 + np.sqrt(3)*r / self.ro) * np.exp(-np.sqrt(3) * r / self.ro)
        self.update_tensor()
	
    def __call__(self, x):
        return self.sigma * np.tensordot(self.C, x)
    
    def update_tensor(self):
        it = np.nditer(self.C, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = np.array(it.multi_index)
            d = np.linalg.norm(idx[:idx.shape[0]//2] - idx[idx.shape[0]//2:])
            it[0] = self.f(d)
            it.iternext()

if __name__=='__main__':
    import matplotlib.pyplot as plt
    size = 10
    ndim = 2
    x = np.random.standard_normal((size,)*ndim)
    C = CovOp(ndim, size, 10, .2)
    plt.imshow(C(x))
    plt.show()
    