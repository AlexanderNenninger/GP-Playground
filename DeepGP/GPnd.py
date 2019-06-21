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

import numpy as np
from matplotlib import pyplot as plt

from operators import mOp, CovOp
from logPdfs import lognorm_pdf
import dataLoading

'small function for updating beta'
def update_beta(beta, acc_prob, target):
	beta += .001*(acc_prob-target)
	return np.clip(beta, 2**(-15), 1-2**(-15))

def measure(T, x):
	'Makes measureing multiple values easy'
	if type(T) == mOp:
		return T(x)
	m = [t(x) for t in T]
	return np.array(m)

size = 100

# image = dataLoading.import_image(size=size)

ndim = 1
shape = (size,)*ndim

eta = np.random.standard_normal()
h = np.exp(eta)

C = CovOp(ndim, size, sigma=2, ro=.1)
C_hat = CovOp(ndim, size, sigma=2, ro=.1)


means = np.array([
	(.5, .5)#,	(.25, .5), (.25, .75),
	# (.5, .25), (.5, .5), (.5, .75),
	# (.75, .25), (.75, .5), (.75, .75),
])

T = [mOp(ndim, size, mean, sigma=.1) for mean in means]

y = 15 #measure(T, image)
print('Data:', y, end='	\n')

xi = np.random.standard_normal(shape)
u = C(xi)

fig, ax = plt.subplots(2,2)
# ax[0,0].imshow(image, cmap='Greys_r')
ax[1,0].plot(u)
# ax[0,1].scatter(means[:,0], means[:,1], c=y)
plt.show()

m = measure(T, u)
print('Measurement:\n <%s> \n %s'%(means, m))

beta_0 = .5
beta_1 = .5

samples = []
probs = []
betas  = []

dx = 1/size**ndim
phi = lambda x, y, dx: np.sum((x-y)**2) * dx * 100

	
for i in range(50000):
	#base layer
	xi_hat = np.sqrt(1-beta_0**2)*xi + beta_0*np.random.standard_normal(shape)

	u_hat = C(xi_hat)
	m_hat = measure(T, u_hat)
	
	logProb_0 = min(phi(m, y, dx) - phi(m_hat, y, dx), 0)

	if np.random.rand() <= np.exp(logProb_0):
		xi = xi_hat
		u = u_hat
		m = m_hat
	
	#second layer
	eta_hat = np.sqrt(1 - beta_1**2) * eta + beta_1 * np.random.standard_normal()
	
	h_hat = np.exp(eta_hat)
	C_hat.sigma = h_hat
	
	logProb_1 = phi(measure(T, C(xi)), y, dx) - phi(measure(T, C_hat(xi)), y, dx)
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