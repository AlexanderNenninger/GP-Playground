'Define log pdfs here'
import numpy as np

def lognorm_pdf(x, s):
	if x<=0 or s<=0:
		raise Exception('Only positive RVs can be distributed ~ lognorm')
	val = -np.log(x)**2/(2*s**2) - np.log(s*x*np.sqrt(2*np.pi))
	return val

def exp_pdf(x, beta):
    if x<=0 or beta<=0:
        raise Exception('Only positive RVs can be distributed ~ exp')
    return -x/beta - np.log(beta)


    1