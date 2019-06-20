import numpy as np
from matplotlib import pyplot as plt

from operators import mOp, CovOp
from logPdfs import lognorm_pdf

	

'small function for updating beta'
def update_beta(beta, acc_prob, target):
	beta += .0005*(acc_prob-target)
	return np.clip(beta, 2**(-10), 1-2**(-10))


def measure(T, x):
	m = [t(x) for t in T]
	return np.array(m)

dim = 200

eta = np.random.standard_normal()
h = np.exp(eta)

C = CovOp((dim,dim), h, ro = 500)
C_hat = CovOp((dim, dim), h, ro = 500)

T = mOp(dim, .1, .1)
dm = 1

xi = np.random.standard_normal(dim)
u = C(xi)

plt.plot(u)
plt.plot(T.filter)
plt.show()
plt.clf()

m = T(u)
print(m)

beta_0 = .5
beta_1 = .5

samples = []
probs = []
betas  = []

phi = lambda x, y, dx: np.sum((x-y)**2) * dx

y = 1000
	
for i in range(100000):
	
	#base layer
	xi_hat = np.sqrt(1-beta_0**2)*xi + beta_0*np.random.standard_normal(dim)

	u_hat = C(xi_hat)
	m_hat = T(u_hat)
	
	logProb_0 = min(phi(m,y,dm) - phi(m_hat,y,dm), 0)

	if np.random.rand() <= np.exp(logProb_0):
		xi = xi_hat
		u = u_hat
		m = m_hat
	
	#second layer
	eta_hat = np.sqrt(1-beta_1**2)*eta + beta_1*np.random.standard_normal()
	
	h_hat = np.exp(eta_hat)
	C_hat.sigma = h_hat
	
	logProb_1 = phi(T(C(xi)), y, dm) - phi(T(C_hat(xi)), y, dm)
	logProb_1 += lognorm_pdf(h_hat,h) - lognorm_pdf(h,h_hat) 
	logProb_1 += lognorm_pdf(h_hat,1) - lognorm_pdf(h,1)
	logProb_1 = min(logProb_1, 0)
	
	if np.random.rand() <= np.exp(logProb_1):
		eta = eta_hat
		h = h_hat
	#C.sigma = h
	
	beta_0 = update_beta(
		beta_0,
		np.exp(logProb_0),
		.23
	)
	
	beta_1 = update_beta(
		beta_1,
		np.exp(logProb_1),
		.23
	)
		
	samples.append((u, C.sigma - h))
	probs.append(
		(np.exp(logProb_0), np.exp(logProb_1))
	)
	betas.append((beta_0, beta_1))

print('acc prob ', np.mean(probs, axis=0))

mean = np.mean([x[0] for x in samples], axis=0)
std = np.std([x[0] for x in samples], axis=0)

plt.plot(mean)
plt.plot(mean+std, 'g--')
plt.plot(mean-std, 'g--')
plt.show()
plt.clf()

plt.hist([x[1] for x in samples], density=True)
plt.title('sigma')
plt.show()
plt.clf()

plt.plot([b[0] for b in betas])
plt.show()