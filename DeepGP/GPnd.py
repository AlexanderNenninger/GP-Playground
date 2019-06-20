import numpy as np
from matplotlib import pyplot as plt

from operators import mOp, CovOp
from logPdfs import lognorm_pdf

'small function for updating beta'
def update_beta(beta, acc_prob, target):
	beta += .001*(acc_prob-target)
	return np.clip(beta, 2**(-15), 1-2**(-15))

ndim = 1
size = 100
shape = (size,)*ndim

eta = np.random.standard_normal()
h = np.exp(eta)

C = CovOp(ndim, size, sigma=10, ro=.1)
C_hat = CovOp(ndim, size, sigma=1, ro=.1)

mean = (.5,)*ndim
T = mOp(ndim, size, mean, sigma=.02)

xi = np.random.standard_normal(shape)
u = C(xi)

fig, ax = plt.subplots(1,2)
ax[0].plot(u)
ax[1].plot(T.F)
plt.show()

m = T(u)
print('Measurement: <%s> %s'%(mean, m))

beta_0 = .5
beta_1 = .5

samples = []
probs = []
betas  = []

dx = 1/size**ndim
phi = lambda x, y, dx: np.sum((x-y)**2) * dx

y = -100
	
for i in range(50000):
	#base layer
	xi_hat = np.sqrt(1-beta_0**2)*xi + beta_0*np.random.standard_normal(shape)

	u_hat = C(xi_hat)
	m_hat = T(u_hat)
	
	logProb_0 = min(phi(m, y, dx) - phi(m_hat, y, dx), 0)

	if np.random.rand() <= np.exp(logProb_0):
		xi = xi_hat
		u = u_hat
		m = m_hat
	
	#second layer
	eta_hat = np.sqrt(1 - beta_1**2) * eta + beta_1 * np.random.standard_normal()
	
	h_hat = np.exp(eta_hat)
	C_hat.sigma = h_hat
	
	logProb_1 = phi(T(C(xi)), y, dx) - phi(T(C_hat(xi)), y, dx)
	logProb_1 += lognorm_pdf(h_hat,h) - lognorm_pdf(h,h_hat) 
	logProb_1 += lognorm_pdf(h_hat,1) - lognorm_pdf(h,1)
	logProb_1 = min(logProb_1, 0)
	
	if np.random.rand() <= np.exp(logProb_1):
		eta = eta_hat
		h = h_hat
	C.sigma = h
	
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
		
	samples.append((u, C.sigma))
	probs.append(
		(np.exp(logProb_0), np.exp(logProb_1))
	)
	betas.append((beta_0, beta_1))

print('acc prob ', np.mean(probs, axis=0))
print('2nd Layer:', np.mean([x[1] for x in samples]), end='	')

mean = np.mean([x[0] for x in samples], axis=0)
std = np.std([x[0] for x in samples], axis=0)

fig, ax = plt.subplots(2, 2)

ax[0,0].plot(mean)
ax[0,0].plot(mean + std, 'g--')
ax[0,0].plot(mean - std, 'g--')

ax[0,1].plot(betas)

ax[1,0].plot([x[1] for x in samples])
ax[1,1].hist([x[1] for x in samples])

plt.show()