'Define log pdfs here'
import numpy as np

def lognorm_pdf(x, s):
	val = -np.log(x)**2/(2*s**2) - np.log(s*x*np.sqrt(2*np.pi))
	return val

def exp_pdf(x, beta):
    return -x/beta - np.log(beta)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(2**-10,10, 1000)
    plt.plot(x, np.exp(lognorm_pdf(x,.5)))
    plt.show()